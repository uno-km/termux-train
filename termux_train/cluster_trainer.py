"""
termux_train.cluster_trainer
============================
AMEVA Disaggregated Virtual RAM Pooling Distributed Pipeline Training Engine.
Enables heterogeneous mobile devices (S25, S21, S20, A53, A35) to aggregate
their LPDDR RAM into a single unified 44GB virtual memory pool for sharded
deep learning and on-device LoRA / Transformer training across CPU & GPU.
Open-Source under Apache License 2.0.
"""

import sys
import os
import time
import socket
import select
import json
import struct
import threading
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

import termux_train as tt
from termux_train import Tensor, get_backend, set_backend, available_backends
import termux_train.nn as nn
import termux_train.optim as optim
from termux_train.cluster import VirtualRAMPool, VirtualNodeInfo, DEFAULT_GUARD_BAND_MB
from termux_train.exceptions import ClusterConnectionError, ClusterConfigurationError

logger = logging.getLogger(__name__)


def _send_msg(sock: socket.socket, payload: Dict[str, Any]) -> None:
    raw = json.dumps(payload).encode("utf-8")
    sock.sendall(struct.pack(">I", len(raw)) + raw)


def _recv_msg(sock: socket.socket, timeout: float = 10.0) -> Dict[str, Any]:
    sock.settimeout(timeout)
    hdr = sock.recv(4)
    if not hdr or len(hdr) < 4:
        raise ConnectionResetError("Connection closed while reading packet header.")
    msg_len = struct.unpack(">I", hdr)[0]
    chunks = []
    received = 0
    while received < msg_len:
        chunk = sock.recv(min(65536, msg_len - received))
        if not chunk:
            raise ConnectionResetError("Connection closed while reading packet body.")
        chunks.append(chunk)
        received += len(chunk)
    body = b"".join(chunks).decode("utf-8")
    return json.loads(body)


class ClusterWorkerServer:
    """
    Symmetric On-Device RPC Worker Server.
    Can be run as a daemon on any Android Termux node to contribute
    its local LPDDR RAM and compute (CPU/Vulkan/AMUDA) to the cluster.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50052,
        backend: str = "auto",
        guard_band_mb: int = DEFAULT_GUARD_BAND_MB,
    ):
        self.host = host
        self.port = port
        self.guard_band_mb = guard_band_mb
        self.backend_name = backend
        self._server_sock: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Shard compute state
        self.layers: List[nn.Module] = []
        self.optimizer: Optional[optim.AdamW] = None
        self.last_inputs: List[Tensor] = []
        self.last_outputs: List[Tensor] = []

        if backend and backend != "auto":
            set_backend(backend)

    def start(self) -> None:
        if self._running:
            return
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(5)
        self._server_sock.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        logger.info("[WORKER] RPC Worker Server started on %s:%d (Backend=%s)", self.host, self.port, get_backend().name)

    def stop(self) -> None:
        self._running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("[WORKER] RPC Worker Server stopped.")

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, addr = self._server_sock.accept()
                t = threading.Thread(target=self._client_handler, args=(conn, addr), daemon=True)
                t.start()
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                logger.error("[WORKER] Error in accept loop: %s", e)
                break

    def _client_handler(self, conn: socket.socket, addr: Tuple[str, int]) -> None:
        b = get_backend()
        while self._running:
            try:
                msg = _recv_msg(conn)
                cmd = msg.get("cmd")

                if cmd == "PING" or cmd == "PROBE":
                    # Report memory and device status
                    import psutil
                    try:
                        vm = psutil.virtual_memory()
                        total_ram = int(vm.total / (1024 * 1024))
                        avail_ram = int(vm.available / (1024 * 1024))
                    except Exception:
                        total_ram = 8192
                        avail_ram = 4096

                    resp = {
                        "status": "ok",
                        "node_id": f"node-{self.port}",
                        "total_ram_mb": total_ram,
                        "mem_available_mb": avail_ram,
                        "safe_usable_vram_mb": max(64, avail_ram - self.guard_band_mb),
                        "backend": b.name,
                    }
                    _send_msg(conn, resp)

                elif cmd == "INIT_SHARD":
                    # Initialize module layers on this worker
                    configs = msg.get("layers", [])
                    lr = float(msg.get("lr", 0.001))
                    self.layers = []
                    for cfg in configs:
                        ltype = cfg.get("type")
                        if ltype == "linear":
                            in_f = cfg["in_features"]
                            out_f = cfg["out_features"]
                            bias = cfg.get("bias", True)
                            layer = nn.Linear(in_f, out_f, bias=bias, backend=b)
                            if "weight" in cfg:
                                layer.weight = Tensor(b.from_data(cfg["weight"], dtype="float32"), dtype="float32", backend=b)
                            if bias and "bias_val" in cfg and cfg["bias_val"] is not None:
                                layer.bias = Tensor(b.from_data(cfg["bias_val"], dtype="float32"), dtype="float32", backend=b)
                            self.layers.append(layer)
                        elif ltype == "relu":
                            self.layers.append(nn.ReLU())
                        elif ltype == "layernorm":
                            dim = cfg["normalized_shape"]
                            self.layers.append(nn.LayerNorm(dim))

                    params = []
                    for layer in self.layers:
                        params.extend(layer.parameters())
                    if params:
                        self.optimizer = optim.AdamW(params, lr=lr)
                    _send_msg(conn, {"status": "ok", "layers_initialized": len(self.layers)})

                elif cmd == "FORWARD":
                    # Forward pass through this shard
                    x_list = msg.get("x")
                    shape = tuple(msg.get("shape", [len(x_list)]))
                    x = Tensor(b.from_data(x_list, dtype="float32"), dtype="float32", requires_grad=True, backend=b).reshape(*shape)
                    
                    self.last_inputs = [x]
                    curr = x
                    for layer in self.layers:
                        curr = layer(curr)
                    self.last_outputs = [curr]

                    out_flat = b.to_flat_list(curr._data)
                    _send_msg(conn, {
                        "status": "ok",
                        "y": out_flat,
                        "shape": list(curr.shape),
                    })

                elif cmd == "BACKWARD":
                    # Backward pass through this shard
                    grad_list = msg.get("grad_output")
                    shape = tuple(msg.get("shape", [len(grad_list)]))
                    grad_output = Tensor(b.from_data(grad_list, dtype="float32"), dtype="float32", backend=b).reshape(*shape)

                    # Compute gradient w.r.t input
                    if self.last_outputs:
                        out = self.last_outputs[-1]
                        out.backward(grad_output)
                        in_grad = self.last_inputs[0].grad
                        in_grad_flat = b.to_flat_list(in_grad._data) if in_grad is not None else []
                        _send_msg(conn, {
                            "status": "ok",
                            "grad_input": in_grad_flat,
                            "shape": list(in_grad.shape) if in_grad is not None else [],
                        })
                    else:
                        _send_msg(conn, {"status": "error", "message": "No active forward activation cached for backward."})

                elif cmd == "STEP":
                    # Optimizer step & zero grad
                    if self.optimizer:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    _send_msg(conn, {"status": "ok"})

                elif cmd == "GET_WEIGHTS":
                    # Dump state dict
                    weights = {}
                    for i, l in enumerate(self.layers):
                        for name, p in l.named_parameters():
                            weights[f"layer_{i}.{name}"] = b.to_flat_list(p._data)
                    _send_msg(conn, {"status": "ok", "weights": weights})

                elif cmd == "CLOSE" or cmd == "SHUTDOWN":
                    _send_msg(conn, {"status": "closing"})
                    break

            except (socket.timeout, ConnectionResetError, BrokenPipeError):
                break
            except Exception as e:
                logger.error("[WORKER-HANDLER-ERROR] %s", e)
                try:
                    _send_msg(conn, {"status": "error", "message": str(e)})
                except Exception:
                    pass
                break
        try:
            conn.close()
        except Exception:
            pass


class ClusterPipelineSession:
    """
    AMEVA Cluster Pipeline Session Orchestrator.
    Manages end-to-end distributed pipeline training across the pooled 44GB fleet memory.
    """

    def __init__(
        self,
        pool: VirtualRAMPool,
        rpc_endpoints: List[str],
        lr: float = 0.001,
        backend: str = "auto",
    ):
        self.pool = pool
        self.rpc_endpoints = rpc_endpoints
        self.lr = lr
        self.backend = backend
        self._sockets: Dict[str, socket.socket] = {}

    def connect_all(self, timeout: float = 3.0) -> None:
        """Connect to all RPC workers and verify reachability (Zero-Silent-Fallback)."""
        failed = []
        for ep in self.rpc_endpoints:
            host, port_str = ep.split(":")
            port = int(port_str)
            try:
                s = socket.create_connection((host, port), timeout=timeout)
                self._sockets[ep] = s
                # Probe node telemetry to register into pool
                _send_msg(s, {"cmd": "PROBE"})
                resp = _recv_msg(s, timeout=timeout)
                self.pool.add_node(
                    endpoint=ep,
                    total_ram_mb=resp.get("total_ram_mb", 8192),
                    mem_available_mb=resp.get("mem_available_mb", 4096),
                    backend=resp.get("backend", "auto"),
                )
            except Exception as exc:
                failed.append((ep, str(exc)))

        if failed:
            self.close()
            details = "\n".join([f"  - {ep}: {err}" for ep, err in failed])
            raise ClusterConnectionError(
                f"[FAIL-FAST] ClusterPipelineSession could not connect to {len(failed)} worker(s):\n{details}"
            )

    def init_shards(self, layer_definitions: List[Dict[str, Any]]) -> List[Tuple[str, int, int]]:
        """
        Shards the specified layer architecture across the virtual RAM pool.
        """
        shards = self.pool.calculate_shards(len(layer_definitions))
        for ep, start_idx, end_idx in shards:
            assigned_layers = layer_definitions[start_idx:end_idx]
            s = self._sockets[ep]
            _send_msg(s, {
                "cmd": "INIT_SHARD",
                "layers": assigned_layers,
                "lr": self.lr,
            })
            resp = _recv_msg(s)
            if resp.get("status") != "ok":
                raise RuntimeError(f"Failed to initialize shard on {ep}: {resp.get('message')}")
        return shards

    def train_step(
        self,
        x: Tensor,
        y: Tensor,
        shards: List[Tuple[str, int, int]],
        criterion: nn.Module,
    ) -> float:
        """
        Executes a distributed pipeline forward-backward step across the fleet.
        """
        b = get_backend()

        # 1. Pipeline Forward
        curr_x = b.to_flat_list(x._data)
        curr_shape = list(x.shape)

        for ep, _, _ in shards:
            s = self._sockets[ep]
            _send_msg(s, {
                "cmd": "FORWARD",
                "x": curr_x,
                "shape": curr_shape,
            })
            resp = _recv_msg(s)
            if resp.get("status") != "ok":
                raise RuntimeError(f"Forward failed on node {ep}: {resp.get('message')}")
            curr_x = resp["y"]
            curr_shape = resp["shape"]

        # 2. Master Loss Computation
        pred_tensor = Tensor(b.from_data(curr_x, dtype="float32"), dtype="float32", requires_grad=True, backend=b).reshape(*curr_shape)
        loss = criterion(pred_tensor, y)
        loss_val = float(loss.item())

        # 3. Master Loss Backward w.r.t Final Output
        loss.backward()
        final_grad = pred_tensor.grad
        curr_grad = b.to_flat_list(final_grad._data)
        grad_shape = list(final_grad.shape)

        # 4. Pipeline Backward (Reverse Order)
        for ep, _, _ in reversed(shards):
            s = self._sockets[ep]
            _send_msg(s, {
                "cmd": "BACKWARD",
                "grad_output": curr_grad,
                "shape": grad_shape,
            })
            resp = _recv_msg(s)
            if resp.get("status") != "ok":
                raise RuntimeError(f"Backward failed on node {ep}: {resp.get('message')}")
            curr_grad = resp["grad_input"]
            grad_shape = resp["shape"]

        # 5. Optimizer Step on All Shards
        for ep, _, _ in shards:
            s = self._sockets[ep]
            _send_msg(s, {"cmd": "STEP"})
            resp = _recv_msg(s)
            if resp.get("status") != "ok":
                raise RuntimeError(f"Optimizer step failed on node {ep}: {resp.get('message')}")

        return loss_val

    def close(self) -> None:
        """Cleanly close all socket connections."""
        for ep, s in list(self._sockets.items()):
            try:
                _send_msg(s, {"cmd": "CLOSE"})
                s.close()
            except Exception:
                pass
        self._sockets.clear()

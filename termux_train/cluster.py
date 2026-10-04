"""
AMEVA Unified Distributed RPC Cluster Orchestration Utility.
Strict Protocol: Zero-Silent-Fallback & Zero-Deception ABI.
Component: [TRAIN-CLUSTER]
"""

import socket
import logging
from typing import List, Optional, Union, Dict, Any, Tuple

from termux_train.exceptions import (
    ClusterConnectionError,
    ClusterConfigurationError,
    ClusterLicenseRequiredError,
)

logger = logging.getLogger(__name__)


def check_cluster_license(license_key: Optional[str] = None) -> None:
    """Enforces exclusive AMEVA-Cluster runtime license requirement.
    Fail-Fast: Raises ClusterLicenseRequiredError if ameva_cluster is not installed
    or license verification fails.
    """
    try:
        from ameva_cluster.guard import verify_cluster_license
    except ImportError:
        raise ClusterLicenseRequiredError()

    if not verify_cluster_license(license_key):
        raise ClusterLicenseRequiredError(
            "\n================================================================================\n"
            "[AMEVA-CLUSTER] CLUSTER LICENSE INVALID (E403)\n"
            "================================================================================\n"
            "The provided AMEVA Cluster license token is invalid or expired.\n"
            "Please check your AMEVA_CLUSTER_LICENSE environment variable.\n"
            "================================================================================"
        )


def setup_cluster_guard_tunnels(
    servers: List[str],
    secret_key: Optional[str] = None
) -> Tuple[List[str], List[Any]]:
    """Establishes authenticated loopback MasterTunnels for remote RPC nodes.

    Transforms remote endpoints into local authenticated tunnel endpoints (127.0.0.1:PORT)
    which transparently inject the AMEVA Guard cryptographic handshake.

    Returns:
        Tuple of (tunnel_server_endpoints, active_tunnel_instances)
    """
    if not servers:
        return [], []

    try:
        from ameva_cluster.guard import MasterTunnel
    except ImportError:
        raise ClusterLicenseRequiredError()

    tunnel_endpoints: List[str] = []
    active_tunnels: List[Any] = []

    for endpoint in servers:
        host, port_str = endpoint.split(":")
        port = int(port_str)

        # Local loopback addresses do not require an extra proxy hop
        if host in ("127.0.0.1", "localhost", "::1"):
            tunnel_endpoints.append(endpoint)
            continue

        tunnel = MasterTunnel(remote_host=host, remote_port=port, secret_key=secret_key)
        tunnel.start()
        active_tunnels.append(tunnel)
        local_ep = f"127.0.0.1:{tunnel.local_port}"
        tunnel_endpoints.append(local_ep)
        logger.info(
            "[AMEVA-TRAIN-CLUSTER] Authenticated MasterTunnel active: %s -> %s",
            local_ep,
            endpoint,
        )

    return tunnel_endpoints, active_tunnels



def parse_cluster_rpc_spec(rpc_input: Optional[Union[str, List[str]]]) -> List[str]:
    """Parse and normalize distributed RPC server endpoints.

    Supports comma-separated strings or list of host:port strings.
    Validates port ranges and host syntax strictly without silent normalization.
    """
    if not rpc_input:
        return []

    if isinstance(rpc_input, str):
        raw_servers = [s.strip() for s in rpc_input.split(",") if s.strip()]
    elif isinstance(rpc_input, (list, tuple)):
        raw_servers = [str(s).strip() for s in rpc_input if str(s).strip()]
    else:
        raise ClusterConfigurationError(
            f"Invalid cluster_rpc_servers specification type: {type(rpc_input).__name__}. "
            f"Expected comma-separated string or list of endpoints."
        )

    parsed_servers: List[str] = []
    for entry in raw_servers:
        if ":" not in entry:
            raise ClusterConfigurationError(
                f"[FAIL-FAST] Invalid RPC address '{entry}': Must be in 'host:port' format."
            )
        parts = entry.split(":")
        if len(parts) != 2:
            raise ClusterConfigurationError(
                f"[FAIL-FAST] Malformed RPC endpoint '{entry}'. Exactly one colon separator required."
            )
        host, port_str = parts[0].strip(), parts[1].strip()
        if not host:
            raise ClusterConfigurationError(f"[FAIL-FAST] Empty hostname/IP in RPC endpoint '{entry}'.")
        try:
            port = int(port_str)
            if port < 1 or port > 65535:
                raise ValueError()
        except ValueError:
            raise ClusterConfigurationError(
                f"[FAIL-FAST] Invalid port number '{port_str}' in RPC endpoint '{entry}'. Must be integer 1..65535."
            )
        parsed_servers.append(f"{host}:{port}")

    return parsed_servers


def verify_rpc_cluster_nodes(servers: List[str], timeout: float = 3.0) -> None:
    """Strictly verify TCP reachability of each RPC server prior to initiating training graph.

    Zero-Silent-Fallback: If ANY specified RPC worker is unreachable, this function raises
    ClusterConnectionError immediately to prevent silent local CPU/GPU fallback degradation.
    """
    if not servers:
        return

    failed_nodes = []
    for endpoint in servers:
        host, port_str = endpoint.split(":")
        port = int(port_str)
        try:
            with socket.create_connection((host, port), timeout=timeout):
                logger.info("[AMEVA-TRAIN-CLUSTER] RPC Worker verified reachable: %s", endpoint)
        except (socket.timeout, ConnectionRefusedError, OSError) as exc:
            failed_nodes.append((endpoint, str(exc)))

    if failed_nodes:
        err_details = "\n".join([f"  - {ep}: {err}" for ep, err in failed_nodes])
        raise ClusterConnectionError(
            f"Distributed RPC Cluster Pre-Flight Check FAILED for {len(failed_nodes)}/{len(servers)} worker(s):\n"
            f"{err_details}\n"
            f"Zero-Silent-Fallback Violation Prevented: Training terminated to avoid silent execution degradation."
        )


def verify_rpc_cluster_health(servers: List[str], timeout: float = 3.0) -> Dict[str, Any]:
    """Inspect and report cluster connectivity status without throwing, returning granular diagnostics."""
    status: Dict[str, Any] = {"all_healthy": True, "nodes": {}}
    for endpoint in servers:
        host, port_str = endpoint.split(":")
        port = int(port_str)
        try:
            with socket.create_connection((host, port), timeout=timeout):
                status["nodes"][endpoint] = {"reachable": True, "error": None}
        except Exception as exc:
            status["all_healthy"] = False
            status["nodes"][endpoint] = {"reachable": False, "error": str(exc)}
    return status


DEFAULT_GUARD_BAND_MB = 300


class VirtualNodeInfo:
    """Telemetry and capacity metadata for a cluster compute/memory node."""

    def __init__(
        self,
        endpoint: str,
        total_ram_mb: int,
        mem_available_mb: int,
        guard_band_mb: int = DEFAULT_GUARD_BAND_MB,
        is_local: bool = False,
        soc: str = "Unknown",
        gpu: str = "Unknown",
        backend: str = "auto",
    ):
        self.endpoint = endpoint
        parts = endpoint.split(":")
        self.host = parts[0]
        self.port = int(parts[1]) if len(parts) > 1 else 50052
        self.total_ram_mb = max(1, total_ram_mb)
        self.mem_available_mb = max(0, mem_available_mb)
        self.guard_band_mb = max(0, guard_band_mb)
        self.is_local = is_local
        self.soc = soc
        self.gpu = gpu
        self.backend = backend

    @property
    def safe_usable_vram_mb(self) -> int:
        """Safe allocatable RAM after deducting safety guard-band (까치밥 메모리)."""
        return max(64, self.mem_available_mb - self.guard_band_mb)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "host": self.host,
            "port": self.port,
            "total_ram_mb": self.total_ram_mb,
            "mem_available_mb": self.mem_available_mb,
            "guard_band_mb": self.guard_band_mb,
            "safe_usable_vram_mb": self.safe_usable_vram_mb,
            "is_local": self.is_local,
            "soc": self.soc,
            "gpu": self.gpu,
            "backend": self.backend,
        }


class VirtualRAMPool:
    """
    AMEVA Virtual RAM Pooling Orchestrator.
    Binds heterogeneous mobile fleet memories (S25, S21, S20, A53, A35)
    into a unified virtual memory space (e.g. 44GB physical RAM pool).
    """

    def __init__(self, guard_band_mb: int = DEFAULT_GUARD_BAND_MB):
        self.guard_band_mb = guard_band_mb
        self.nodes: Dict[str, VirtualNodeInfo] = {}

    def add_node(
        self,
        endpoint: str,
        total_ram_mb: int,
        mem_available_mb: int,
        is_local: bool = False,
        soc: str = "Unknown",
        gpu: str = "Unknown",
        backend: str = "auto",
    ) -> VirtualNodeInfo:
        node = VirtualNodeInfo(
            endpoint=endpoint,
            total_ram_mb=total_ram_mb,
            mem_available_mb=mem_available_mb,
            guard_band_mb=self.guard_band_mb,
            is_local=is_local,
            soc=soc,
            gpu=gpu,
            backend=backend,
        )
        self.nodes[endpoint] = node
        return node

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def total_pooled_ram_mb(self) -> int:
        """Total raw physical RAM aggregated across all active fleet nodes."""
        return sum(n.total_ram_mb for n in self.nodes.values())

    @property
    def total_safe_vram_mb(self) -> int:
        """Total safe usable memory space after guard-band deduction."""
        return sum(n.safe_usable_vram_mb for n in self.nodes.values())

    def calculate_shards(
        self, num_layers: int, strategy: str = "proportional"
    ) -> List[Tuple[str, int, int]]:
        """
        Partitions layers across active cluster nodes according to memory capacity.
        Returns list of tuples: (endpoint, start_layer_inclusive, end_layer_exclusive).
        """
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not self.nodes:
            raise ClusterConfigurationError("Cannot calculate shards: VirtualRAMPool has no registered nodes.")

        node_list = list(self.nodes.values())
        k = len(node_list)

        if k == 1 or num_layers <= k:
            # Simple uniform distribution
            base = num_layers // k
            rem = num_layers % k
            allocations = [base + (1 if i < rem else 0) for i in range(k)]
        elif strategy == "uniform":
            base = num_layers // k
            rem = num_layers % k
            allocations = [base + (1 if i < rem else 0) for i in range(k)]
        else:
            # Proportional to safe usable memory
            total_mem = max(1, self.total_safe_vram_mb)
            allocations = [max(1, round(n.safe_usable_vram_mb / total_mem * num_layers)) for n in node_list]
            diff = num_layers - sum(allocations)
            if diff != 0:
                # Adjust highest capacity node
                max_idx = max(range(k), key=lambda i: node_list[i].safe_usable_vram_mb)
                allocations[max_idx] = max(1, allocations[max_idx] + diff)

        shards = []
        curr = 0
        for i, count in enumerate(allocations):
            end = min(curr + count, num_layers)
            shards.append((node_list[i].endpoint, curr, end))
            curr = end

        # If any layers remain unassigned due to rounding, assign to last node
        if curr < num_layers and shards:
            ep, s, _ = shards[-1]
            shards[-1] = (ep, s, num_layers)

        return shards

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_nodes": self.node_count,
            "total_pooled_ram_mb": self.total_pooled_ram_mb,
            "total_safe_vram_mb": self.total_safe_vram_mb,
            "guard_band_per_node_mb": self.guard_band_mb,
            "nodes": [n.to_dict() for n in self.nodes.values()],
        }

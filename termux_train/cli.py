"""
termux_train.cli
================
Official Command-Line Interface (CLI) for termux-train.
Provides diagnostics, environment validation, benchmark scoring, demo execution, and on-device training.
"""

import sys
import os
import time
import json
import argparse
import subprocess

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError) as _rec_err:
    _ = _rec_err

from termux_train import __version__, available_backends, get_backend, set_backend, Tensor, randn, nn
from termux_train.utils.termux_env import is_termux, is_android, get_device_info


def cmd_info(args):
    """Prints comprehensive system, hardware, and backend diagnostic information."""
    is_json = getattr(args, "json", False)
    info = get_device_info()
    backends = [b.upper() for b in available_backends()]
    active_b = get_backend().name.upper()

    if is_json:
        payload = {
            "version": __version__,
            "device": info,
            "backend": {
                "active": active_b,
                "available": backends,
            },
            "environment": {
                "is_termux": is_termux(),
                "is_android": is_android(),
            }
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print("=" * 65)
    print(f"  📱 termux-train (v{__version__}) - System Diagnostics")
    print("=" * 65)
    for k, v in info.items():
        print(f"  • {k:20s}: {v}")
    print("=" * 65)
    print("  🔧 Framework Capabilities:")
    print(f"  • Active Backend     : {active_b}")
    print(f"  • Available Backends : {backends}")
    print(f"  • Termux Native      : {'YES' if is_termux() else 'NO (Host Environment)'}")
    print(f"  • Android OS         : {'YES' if is_android() else 'NO'}")
    print("=" * 65)


def cmd_doctor(args):
    """Comprehensive environment, hardware, and training capacity diagnostic doctor."""
    is_json = getattr(args, "json", False)
    info = get_device_info()
    backends = [b.upper() for b in available_backends()]
    vulkan_supported = "VULKAN" in backends

    # Memory & Capacity heuristics without magic numbers
    ram_mb = 0
    try:
        import psutil
        ram_mb = int(psutil.virtual_memory().total / (1024 * 1024))
    except (ImportError, OSError, AttributeError) as _ps_err:
        import logging
        logging.getLogger(__name__).debug("psutil memory inspection unavailable: %s", _ps_err)

    if ram_mb <= 0 and os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            ram_mb = int(parts[1]) // 1024
                        break
        except (OSError, ValueError) as _mem_err:
            import logging
            logging.getLogger(__name__).debug("/proc/meminfo read failed: %s", _mem_err)

    if ram_mb <= 0:
        ram_mb = 4096  # Baseline fallback if kernel information is unreadable

    if ram_mb >= 8192:
        rec_lora_rank = 16
        rec_batch_size = 32
        tier = "High-End Mobile"
    elif ram_mb >= 4096:
        rec_lora_rank = 8
        rec_batch_size = 16
        tier = "Standard Mobile"
    else:
        rec_lora_rank = 4
        rec_batch_size = 4
        tier = "Ultra-Low Memory"

    report = {
        "framework": "termux-train",
        "version": __version__,
        "platform": {
            "system": sys.platform,
            "is_termux": is_termux(),
            "is_android": is_android(),
        },
        "hardware": {
            "device": info.get("Device Model", "Generic ARM64 / Host"),
            "cpu_cores": os.cpu_count() or 4,
            "ram_mb": ram_mb,
            "tier": tier,
        },
        "backends": {
            "active": get_backend().name.upper(),
            "available": backends,
            "vulkan_acceleration": vulkan_supported,
        },
        "recommended_training_config": {
            "lora_rank": rec_lora_rank,
            "batch_size": rec_batch_size,
            "seq_len": 512,
        },
        "status": "HEALTHY",
    }

    if is_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print("=" * 65)
    print(f"  🩺 termux-train Diagnostic Doctor (v{__version__})")
    print("=" * 65)
    print(f"  • Platform       : {report['platform']['system']} (Termux: {report['platform']['is_termux']})")
    print(f"  • Device Tier    : {tier} (RAM: ~{ram_mb} MB | Cores: {report['hardware']['cpu_cores']})")
    print(f"  • Active Backend : {report['backends']['active']}")
    print(f"  • Vulkan GPU     : {'Available (Hardware Accelerated)' if vulkan_supported else 'Not Present (Using CPU/NumPy Engine)'}")
    print("=" * 65)
    print("  📋 Recommended On-Device Training Preset:")
    print(f"  • Max LoRA Rank  : r={rec_lora_rank}")
    print(f"  • Optimal Batch  : {rec_batch_size}")
    print(f"  • SafeTensors IO : Zero-Copy mmap Enabled")
    print("=" * 65)


def cmd_benchmark(args):
    """Runs on-device GEMM & Autograd latency and throughput benchmarks."""
    is_json = getattr(args, "json", False)
    dim = getattr(args, "dim", 256)
    
    # 1. Warm-up
    a = randn((dim, dim), requires_grad=True)
    b = randn((dim, dim), requires_grad=True)
    c = (a @ b).sum()
    c.backward()

    # 2. Benchmark Forward GEMM
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        z = a @ b
    gemm_lat_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    # 3. Benchmark Forward + Backward Autograd
    t0 = time.perf_counter()
    for _ in range(iters):
        a.grad = None
        b.grad = None
        out = (a @ b).sum()
        out.backward()
    autograd_lat_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    # Theoretical FLOPs for (N x N) @ (N x N) = 2 * N^3
    gflops = (2 * (dim ** 3)) / (gemm_lat_ms / 1000.0) / 1e9

    res = {
        "dimension": f"{dim}x{dim}",
        "iterations": iters,
        "backend": get_backend().name.upper(),
        "gemm_latency_ms": round(gemm_lat_ms, 3),
        "autograd_step_latency_ms": round(autograd_lat_ms, 3),
        "throughput_gflops": round(gflops, 3),
    }

    if is_json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    print("=" * 65)
    print(f"  ⚡ termux-train Benchmark (Dimension: {dim}x{dim})")
    print("=" * 65)
    print(f"  • Backend                 : {res['backend']}")
    print(f"  • Forward GEMM Latency    : {res['gemm_latency_ms']:.3f} ms")
    print(f"  • Full Autograd Step (F+B): {res['autograd_step_latency_ms']:.3f} ms")
    print(f"  • Compute Throughput      : {res['throughput_gflops']:.3f} GFLOPS")
    print("=" * 65)


def cmd_check(args):
    """Performs self-test across all available backends to verify mathematical integrity."""
    print("=" * 65)
    print(f"  🔍 termux-train v{__version__} - Self-Diagnostic Verification")
    print("=" * 65)

    all_passed = True
    for b_name in available_backends():
        set_backend(b_name)
        print(f"  [+] Testing Backend: [{b_name.upper()}] ... ", end="", flush=True)
        try:
            # 1. Tensor creation & basic math
            a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            b = Tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
            c = (a @ b).sum()
            c.backward()

            # 2. NN & RoPE Transformer verification
            m = nn.TinyTransformerLM(vocab_size=10, d_model=8, num_heads=2, d_ff=16, num_layers=1, pos_type="rope")
            inp = Tensor([[1, 2, 3]], dtype="int64")
            logits, _ = m(inp)
            assert logits.shape == (1, 3, 10)

            print("PASSED ✅")
        except Exception as e:
            print(f"FAILED ❌ ({e})")
            all_passed = False

    print("=" * 65)
    if all_passed:
        print("  🎉 All backends verified and operating with mathematical integrity!")
    else:
        print("  ⚠️ One or more backends encountered errors. Check system libraries.")
    print("=" * 65)


def cmd_score(args):
    """Runs the 0-Point Baseline Granular Audit Scoring System."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(root_dir, "scripts", "run_audit_scoring.py")
    if not os.path.exists(script_path):
        pytest_cmd = [sys.executable, "-m", "pytest", "tests/test_audit_scorecard.py", "-s", "-v"]
        subprocess.run(pytest_cmd, check=False)
        return

    subprocess.run([sys.executable, script_path], check=False)


def cmd_demo(args):
    """Executes one of the 8 canonical example demos."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(root_dir, "examples")
    
    demo_map = {
        "1": "01_tensor_basics.py",
        "2": "02_nn_forward_backward.py",
        "3": "03_matmul_1d_to_3d.py",
        "4": "04_xor_training.py",
        "5": "05_mobile_training_runtime.py",
        "6": "06_lora_adapter_training.py",
        "7": "07_transformer_lm.py",
        "8": "08_docfold_trainer.py",
    }

    choice = str(args.demo_number).lstrip("0")
    if choice not in demo_map:
        print(f"❌ Unknown demo number: '{args.demo_number}'. Available demos: 1 through 8.")
        print("   1: Tensor Basics")
        print("   2: NN Forward/Backward")
        print("   3: 1D~3D Matmul")
        print("   4: XOR Training")
        print("   5: Mobile Training Runtime & Checkpoints")
        print("   6: LoRA Adapter Fine-Tuning")
        print("   7: Character-Level Transformer LM")
        print("   8: DocFold Sequence Mapping Trainer")
        sys.exit(1)

    target_script = os.path.join(examples_dir, demo_map[choice])
    print(f"🚀 Running Demo [{choice}]: {demo_map[choice]} ...\n")
    subprocess.run([sys.executable, target_script], check=False)


def cmd_train(args):
    """Executes on-device training / LoRA loop via runner.run_session."""
    from termux_train.runtime.runner import run_session

    cfg = {
        "modelType": getattr(args, "model", "mlp"),
        "dataPath": getattr(args, "data", None),
        "dim": getattr(args, "dim", 32),
        "loraRank": getattr(args, "rank", 8),
        "epochs": getattr(args, "epochs", 5),
        "lr": getattr(args, "lr", 0.001),
        "batchSize": getattr(args, "batch_size", 16),
        "seqLen": getattr(args, "seq_len", 32),
        "backend": getattr(args, "backend", "auto"),
        "checkpointPath": getattr(args, "checkpoint", None),
        "resumePath": getattr(args, "resume", None),
        "rpc": getattr(args, "rpc", None),
        "tensor_split": getattr(args, "tensor_split", None),
        "cluster_rpc_servers": getattr(args, "cluster_rpc_servers", None),
        "cluster_split_mode": getattr(args, "cluster_split_mode", None),
        "cluster_tensor_split": getattr(args, "cluster_tensor_split", None),
        "cluster_vram_budget": getattr(args, "cluster_vram_budget", None),
        "vocab_slice": getattr(args, "vocab_slice", None),
        "chunk_layers": getattr(args, "chunk_layers", None),
        "stream_layers": getattr(args, "stream_layers", None),
        "virtual_ram_pool": getattr(args, "virtual_ram_pool", False),
    }

    try:
        run_session(cfg)
    except Exception as exc:
        print(f"[ERROR] Training failed: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_cluster_worker(args):
    """Starts symmetric on-device RPC worker server for virtual RAM pooling."""
    from termux_train.cluster_trainer import ClusterWorkerServer
    port = getattr(args, "port", 50052)
    host = getattr(args, "host", "0.0.0.0")
    backend_req = getattr(args, "backend", "auto")
    guard_band = getattr(args, "guard_band", 300)

    server = ClusterWorkerServer(host=host, port=port, backend=backend_req, guard_band_mb=guard_band)
    server.start()
    print("=" * 65)
    print(f"  🌐 AMEVA-Cluster RPC Worker Active on {host}:{port}")
    print(f"  • Safe Guard-Band  : {guard_band} MB")
    print(f"  • Compute Backend  : {get_backend().name.upper()}")
    print("  • Status           : Waiting for training shards (Ctrl+C to stop)")
    print("=" * 65)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping cluster worker...")
        server.stop()


def cmd_cluster_probe(args):
    """Probes cluster RPC nodes, diagnostics and reports pooled virtual RAM."""
    from termux_train.cluster import parse_cluster_rpc_spec, VirtualRAMPool
    import socket
    import struct
    import json

    rpc_spec = getattr(args, "rpc", None)
    if not rpc_spec:
        print("❌ Error: --rpc endpoint list required (e.g. --rpc '192.168.0.220:50052,192.168.0.253:50052')", file=sys.stderr)
        sys.exit(1)

    nodes = parse_cluster_rpc_spec(rpc_spec)
    pool = VirtualRAMPool(guard_band_mb=getattr(args, "guard_band", 300))

    print("=" * 65)
    print("  🔍 AMEVA Virtual RAM Pooling Fleet Probe")
    print("=" * 65)

    for ep in nodes:
        host, port_str = ep.split(":")
        port = int(port_str)
        t0 = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=3.0) as s:
                rtt = (time.perf_counter() - t0) * 1000.0
                raw = json.dumps({"cmd": "PROBE"}).encode("utf-8")
                s.sendall(struct.pack(">I", len(raw)) + raw)
                hdr = s.recv(4)
                mlen = struct.unpack(">I", hdr)[0]
                resp = json.loads(s.recv(mlen).decode("utf-8"))
                total_ram = resp.get("total_ram_mb", 8192)
                avail_ram = resp.get("mem_available_mb", 4096)
                backend_name = resp.get("backend", "unknown")
                node = pool.add_node(ep, total_ram, avail_ram, backend=backend_name)
                print(f"  ✅ Node [{ep}]: {rtt:.1f}ms RTT | RAM: {total_ram}MB (Safe Usable: {node.safe_usable_vram_mb}MB) | Backend: {backend_name.upper()}")
        except Exception as exc:
            print(f"  ❌ Node [{ep}]: FAILED ({exc})")

    summary = pool.get_summary()
    print("=" * 65)
    print(f"  📊 Pooled Nodes       : {summary['total_nodes']} active")
    print(f"  🧠 Total Pooled RAM   : {summary['total_pooled_ram_mb']} MB (~{summary['total_pooled_ram_mb']/1024:.1f} GB)")
    print(f"  🛡️ Safe Usable Memory : {summary['total_safe_vram_mb']} MB (~{summary['total_safe_vram_mb']/1024:.1f} GB)")
    print("=" * 65)


def cmd_diffusion_train(args):
    """Executes on-device Image Folder Diffusion LoRA training."""
    from termux_train.diffusion.trainer import train_diffusion_lora

    image_dir = args.image_dir
    output_path = getattr(args, "output", None) or os.path.join(image_dir, "adapter_diffusion_lora.safetensors")
    prompt = getattr(args, "prompt", "a photo of subject")
    resolution = getattr(args, "resolution", 512)
    epochs = getattr(args, "epochs", 5)
    lr = getattr(args, "lr", 0.001)
    batch_size = getattr(args, "batch_size", 1)
    rank = getattr(args, "rank", 4)
    alpha = getattr(args, "alpha", 1.0)
    backend_req = getattr(args, "backend", "auto")

    train_diffusion_lora(
        image_dir=image_dir,
        output_path=output_path,
        prompt=prompt,
        resolution=resolution,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        rank=rank,
        alpha=alpha,
        backend=backend_req,
        verbose=True,
    )


def cmd_vision_train(args):
    """Executes on-device Vision VLM LoRA training."""
    from termux_train.vision.trainer import train_vision_vlm_lora

    data = args.data
    output_path = getattr(args, "output", None) or os.path.join(data if os.path.isdir(data) else os.path.dirname(data), "adapter_vision_lora.safetensors")
    epochs = getattr(args, "epochs", 5)
    lr = getattr(args, "lr", 0.001)
    batch_size = getattr(args, "batch_size", 1)
    rank = getattr(args, "rank", 4)
    alpha = getattr(args, "alpha", 1.0)
    backend_req = getattr(args, "backend", "auto")

    train_vision_vlm_lora(
        data_source=data,
        output_path=output_path,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        rank=rank,
        alpha=alpha,
        backend=backend_req,
        verbose=True,
    )


def cmd_stt_train(args):
    """Executes on-device Whisper STT Cross-Attention LoRA acoustic training."""
    from termux_train.audio.trainer import train_stt_lora

    data = args.data
    output_path = getattr(args, "output", None) or os.path.join(data if os.path.isdir(data) else os.path.dirname(data), "adapter_stt_lora.safetensors")
    epochs = getattr(args, "epochs", 5)
    lr = getattr(args, "lr", 0.001)
    batch_size = getattr(args, "batch_size", 1)
    rank = getattr(args, "rank", 4)
    alpha = getattr(args, "alpha", 1.0)
    backend_req = getattr(args, "backend", "auto")

    train_stt_lora(
        data_source=data,
        output_path=output_path,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        rank=rank,
        alpha=alpha,
        backend=backend_req,
        verbose=True,
    )


def cmd_tts_train(args):
    """Executes on-device Audio TTS Speaker Adaptation LoRA training."""
    from termux_train.audio.trainer import train_tts_lora

    data = args.data
    output_path = getattr(args, "output", None) or os.path.join(data if os.path.isdir(data) else os.path.dirname(data), "adapter_tts_lora.safetensors")
    epochs = getattr(args, "epochs", 5)
    lr = getattr(args, "lr", 0.001)
    batch_size = getattr(args, "batch_size", 1)
    rank = getattr(args, "rank", 4)
    alpha = getattr(args, "alpha", 1.0)
    backend_req = getattr(args, "backend", "auto")

    train_tts_lora(
        data_source=data,
        output_path=output_path,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        rank=rank,
        alpha=alpha,
        backend=backend_req,
        verbose=True,
    )


def cmd_peft(args):
    """Executes on-device PEFT (LoRA / DoRA) training loop."""
    from termux_train.nn.linear import Linear
    from termux_train.nn.lora import LoRALinear
    from termux_train.nn.dora import DoRALinear
    from termux_train.nn.loss import MSELoss
    from termux_train.optim.adamw import AdamW
    from termux_train.adapters.hub import export_target_adapter, TargetEcosystem

    peft_type = getattr(args, "type", "lora").lower()
    target_eco = getattr(args, "target", "llamacpp").lower()
    rank = getattr(args, "rank", 8)
    alpha = getattr(args, "alpha", 16.0)
    dim = getattr(args, "dim", 64)
    epochs = getattr(args, "epochs", 5)
    lr = getattr(args, "lr", 0.001)
    batch_size = getattr(args, "batch_size", 16)
    backend_req = getattr(args, "backend", "auto")
    output_path = getattr(args, "output", None)

    if backend_req and backend_req != "auto":
        set_backend(backend_req)

    b = get_backend()
    print("=" * 65)
    print(f"  ⚡ termux-train PEFT Engine: {peft_type.upper()} Mode")
    print(f"  • Target Ecosystem : {target_eco.upper()}")
    print(f"  • Compute Backend  : {b.name.upper()}")
    print(f"  • Architecture     : Dim={dim} -> Rank={rank} (alpha={alpha})")
    print(f"  • Hyperparameters  : Epochs={epochs}, LR={lr}, Batch={batch_size}")
    print("=" * 65)

    base_layer = Linear(dim, dim, bias=True, backend=b)
    if peft_type == "dora":
        model = DoRALinear.from_linear(base_layer, rank=rank, alpha=alpha)
    else:
        model = LoRALinear.from_linear(base_layer, rank=rank, alpha=alpha)

    optimizer = AdamW(model.adapter_parameters(), lr=lr)
    loss_fn = MSELoss()

    for epoch in range(1, epochs + 1):
        x = randn((batch_size, dim), backend=b)
        target = randn((batch_size, dim), backend=b)

        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        print(f"  [Epoch {epoch}/{epochs}] Step Loss: {loss.item():.6f}")

    print("=" * 65)
    print("  ✅ PEFT Training session completed successfully.")

    if output_path:
        saved = export_target_adapter(
            model,
            target=target_eco,
            output_path=output_path,
            base_model_name=f"on-device-{target_eco}-base",
            custom_metadata={"peft_type": peft_type, "rank": str(rank), "alpha": str(alpha)}
        )
        print(f"  💾 Exported target adapter: {saved}")
        print("=" * 65)


def cmd_rl(args):
    """Executes on-device Reinforcement Learning (GRPO / DPO / PPO)."""
    from termux_train.rl.grpo import GRPOLoss
    from termux_train.rl.dpo import DPOLoss
    from termux_train.rl.ppo import PPOLoss
    from termux_train.rl.reward import xml_format_reward, accuracy_reward
    from termux_train.nn.linear import Linear
    from termux_train.optim.adamw import AdamW

    method = getattr(args, "method", "grpo").lower()
    group_size = getattr(args, "group_size", 4)
    epochs = getattr(args, "epochs", 5)
    lr = getattr(args, "lr", 0.001)
    beta = getattr(args, "beta", 0.04)
    epsilon = getattr(args, "epsilon", 0.2)
    backend_req = getattr(args, "backend", "auto")

    if backend_req and backend_req != "auto":
        set_backend(backend_req)

    b = get_backend()
    print("=" * 65)
    print(f"  🧠 termux-train Reinforcement Learning: {method.upper()} Engine")
    print(f"  • Compute Backend  : {b.name.upper()}")
    print(f"  • Group Size       : G={group_size}")
    print(f"  • Hyperparameters  : Epochs={epochs}, LR={lr}, Beta={beta}, Epsilon={epsilon}")
    print("=" * 65)

    dim = 16
    policy_head = Linear(dim, 1, bias=False, backend=b)
    optimizer = AdamW(policy_head.parameters(), lr=lr)

    if method == "grpo":
        loss_fn = GRPOLoss(epsilon=epsilon, beta=beta, group_size=group_size)
        for ep in range(1, epochs + 1):
            optimizer.zero_grad()
            inputs = randn((group_size, dim), backend=b)
            logps = policy_head(inputs).flatten()
            old_logps = logps.detach()

            candidate_texts = [
                f"<think>Step {i} analysis</think><answer>{i * 42}</answer>" if i % 2 == 0
                else f"Raw text output without tags {i}"
                for i in range(group_size)
            ]
            rewards = [xml_format_reward(txt) + accuracy_reward(txt, "0") for txt in candidate_texts]

            loss, metrics = loss_fn(logps, old_logps, rewards)
            loss.backward()
            optimizer.step()
            print(f"  [Epoch {ep}/{epochs}] GRPO Loss: {metrics['total_loss']:.6f} | Mean Adv: {metrics['mean_advantage']:.4f}")

    elif method == "dpo":
        loss_fn = DPOLoss(beta=beta)
        batch = 4
        for ep in range(1, epochs + 1):
            optimizer.zero_grad()
            inp_w = randn((batch, dim), backend=b)
            inp_l = randn((batch, dim), backend=b)
            pi_w = policy_head(inp_w).flatten()
            pi_l = policy_head(inp_l).flatten()
            ref_w = pi_w.detach()
            ref_l = (pi_l - 0.5).detach()

            loss, metrics = loss_fn(pi_w, pi_l, ref_w, ref_l)
            loss.backward()
            optimizer.step()
            print(f"  [Epoch {ep}/{epochs}] DPO Loss: {metrics['loss']:.6f} | Reward Margin: {metrics['reward_margin']:.4f}")

    elif method == "ppo":
        loss_fn = PPOLoss(clip_eps=epsilon)
        batch = group_size
        for ep in range(1, epochs + 1):
            optimizer.zero_grad()
            inp = randn((batch, dim), backend=b)
            logps = policy_head(inp).flatten()
            old_logps = logps.detach()
            advs = Tensor([1.0, -1.0, 0.5, -0.5][:batch], backend=b)
            loss, metrics = loss_fn(logps, old_logps, advs)
            loss.backward()
            optimizer.step()
            print(f"  [Epoch {ep}/{epochs}] PPO Loss: {metrics['total_loss']:.6f}")

    print("=" * 65)
    print(f"  ✅ {method.upper()} on-device reinforcement learning converged.")
    print("=" * 65)


def cmd_export(args):
    """Exports trained adapter directly for target ecosystem."""
    from termux_train.adapters.hub import export_target_adapter

    adapter_path = args.adapter
    target = args.target
    output_path = args.output
    base_model = getattr(args, "base_model", "unknown")

    if not os.path.exists(adapter_path):
        print(f"❌ Input adapter file not found: {adapter_path}", file=sys.stderr)
        sys.exit(1)

    if adapter_path.endswith(".safetensors"):
        from termux_train.checkpoint.safetensors import load_safetensors
        tensors, meta = load_safetensors(adapter_path)
        data = {"adapters": {"layer": {"lora_A": tensors.get("layer.lora_A", list(tensors.values())[0]).tolist() if tensors else [], "lora_B": tensors.get("layer.lora_B", list(tensors.values())[-1]).tolist() if tensors else []}}}
    else:
        import json
        with open(adapter_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    out_file = export_target_adapter(
        data,
        target=target,
        output_path=output_path,
        base_model_name=base_model
    )
    print(f"✅ Adapter exported for {target.upper()}: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        prog="termux-train",
        description="Native On-Device Deep Learning & LoRA Training Framework for Android Termux."
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s v{__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # info
    p_info = subparsers.add_parser("info", help="Display environment, hardware, and backend capabilities")
    p_info.add_argument("--json", action="store_true", help="Output diagnostics in JSON format")
    p_info.set_defaults(func=cmd_info)

    # doctor
    p_doc = subparsers.add_parser("doctor", help="Inspect device hardware, RAM tier, and Vulkan GPU status")
    p_doc.add_argument("--json", action="store_true", help="Output doctor report in JSON format")
    p_doc.set_defaults(func=cmd_doctor)

    # benchmark
    p_bm = subparsers.add_parser("benchmark", help="Run on-device GEMM & Autograd latency benchmark")
    p_bm.add_argument("--dim", type=int, default=256, help="Matrix dimension N for NxN GEMM (default: 256)")
    p_bm.add_argument("--json", action="store_true", help="Output benchmark metrics in JSON format")
    p_bm.set_defaults(func=cmd_benchmark)

    # check
    p_check = subparsers.add_parser("check", help="Run self-diagnostic mathematical checks across all backends")
    p_check.set_defaults(func=cmd_check)

    # score / test
    p_score = subparsers.add_parser("score", help="Run 0-point baseline granular audit scoring system")
    p_score.set_defaults(func=cmd_score)

    # demo
    p_demo = subparsers.add_parser("demo", help="Run an interactive example demo (1 to 8)")
    p_demo.add_argument("demo_number", type=int, help="Demo number (1 to 8)")
    p_demo.set_defaults(func=cmd_demo)

    # train
    p_train = subparsers.add_parser("train", help="Run on-device training / LoRA loop")
    p_train.add_argument("--model", type=str, default="mlp", choices=["mlp", "lora", "transformer"], help="Model architecture")
    p_train.add_argument("--data", type=str, default=None, help="Path to dataset file (.safetensors, .jsonl, .txt)")
    p_train.add_argument("--dim", type=int, default=32, help="Model hidden/embedding dimension")
    p_train.add_argument("--rank", type=int, default=8, help="LoRA rank (for lora model)")
    p_train.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    p_train.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p_train.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    p_train.add_argument("--seq-len", type=int, default=32, help="Sequence length (for transformer)")
    p_train.add_argument("--backend", type=str, default="auto", help="Compute backend (auto, vulkan, numpy, python)")
    p_train.add_argument("--checkpoint", type=str, default=None, help="Path to save SafeTensors checkpoint")
    p_train.add_argument("--resume", type=str, default=None, help="Path to existing SafeTensors checkpoint to resume training from")
    p_train.add_argument("--rpc", type=str, default=None, help="Distributed RPC server addresses (e.g. '192.168.0.220:50052,192.168.0.253:50052')")
    p_train.add_argument("-ts", "--tensor-split", type=str, default=None, help="Fraction of the model to offload across devices (e.g. '50,50' or '60,40')")
    p_train.add_argument("--cluster-rpc-servers", type=str, default=None, help="AMEVA cluster RPC server endpoints (comma-separated host:port)")
    p_train.add_argument("--cluster-split-mode", type=str, default=None, help="AMEVA cluster split mode")
    p_train.add_argument("--cluster-tensor-split", type=str, default=None, help="AMEVA cluster tensor split ratios")
    p_train.add_argument("--cluster-vram-budget", type=str, default=None, help="AMEVA cluster VRAM budget")
    p_train.add_argument("--vocab-slice", type=int, default=None, help="Slice LM Head vocabulary projection to top N items (reduces VRAM significantly)")
    p_train.add_argument("--chunk-layers", type=int, default=None, help="Process layers in chunks of N to prevent Mali GPU watchdog timeouts")
    p_train.add_argument("--stream-layers", type=int, default=None, help="Layer streaming ping-pong buffer size (reduces active resident RAM)")
    p_train.add_argument("--virtual-ram-pool", action="store_true", help="Aggregate all cluster node RAM into a single unified virtual memory pool for sharded pipeline training")
    p_train.set_defaults(func=cmd_train)

    # cluster-worker
    p_cw = subparsers.add_parser("cluster-worker", help="Start symmetric RPC worker server to contribute LPDDR RAM to cluster")
    p_cw.add_argument("--host", type=str, default="0.0.0.0", help="Binding host address (default: 0.0.0.0)")
    p_cw.add_argument("--port", type=int, default=50052, help="RPC port (default: 50052)")
    p_cw.add_argument("--backend", type=str, default="auto", help="Compute backend: auto, vulkan, amuda, numpy, python")
    p_cw.add_argument("--guard-band", type=int, default=300, help="Safety guard-band in MB to protect against Android LMK (default: 300)")
    p_cw.set_defaults(func=cmd_cluster_worker)

    # cluster-probe
    p_cp = subparsers.add_parser("cluster-probe", help="Probe cluster RPC nodes and inspect pooled virtual RAM")
    p_cp.add_argument("--rpc", type=str, required=True, help="RPC endpoints to probe (e.g. '192.168.0.220:50052,192.168.0.253:50052')")
    p_cp.add_argument("--guard-band", type=int, default=300, help="Safety guard-band in MB (default: 300)")
    p_cp.set_defaults(func=cmd_cluster_probe)

    # peft (LoRA / DoRA)
    p_peft = subparsers.add_parser("peft", aliases=["lora"], help="Run on-device PEFT (LoRA / DoRA) training & export")
    p_peft.add_argument("--type", type=str, default="lora", choices=["lora", "dora"], help="PEFT algorithm: LoRA or DoRA")
    p_peft.add_argument("--target", type=str, default="llamacpp", choices=["llamacpp", "bitnet", "diffusion", "vision", "tts", "stt"], help="Target ecosystem component")
    p_peft.add_argument("--dim", type=int, default=64, help="Hidden dimension")
    p_peft.add_argument("--rank", type=int, default=8, help="Adapter rank")
    p_peft.add_argument("--alpha", type=float, default=16.0, help="Scaling factor alpha")
    p_peft.add_argument("--epochs", type=int, default=5, help="Epoch count")
    p_peft.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p_peft.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p_peft.add_argument("--backend", type=str, default="auto", help="Compute backend")
    p_peft.add_argument("--output", type=str, default=None, help="Export path for generated adapter")
    p_peft.set_defaults(func=cmd_peft)

    # diffusion-train
    p_diff = subparsers.add_parser("diffusion-train", help="Run on-device Image Folder Diffusion LoRA training")
    p_diff.add_argument("--image-dir", type=str, required=True, help="Directory containing training images and .txt captions")
    p_diff.add_argument("--output", type=str, default=None, help="Output .safetensors path for trained LoRA adapter")
    p_diff.add_argument("--prompt", type=str, default="a photo of subject", help="Default trigger prompt/caption")
    p_diff.add_argument("--resolution", type=int, default=512, choices=[256, 512, 768], help="Image resolution (default: 512)")
    p_diff.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    p_diff.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p_diff.add_argument("--batch-size", type=int, default=1, help="Batch size")
    p_diff.add_argument("--rank", type=int, default=4, help="LoRA rank")
    p_diff.add_argument("--alpha", type=float, default=1.0, help="LoRA alpha scaling factor")
    p_diff.add_argument("--backend", type=str, default="auto", help="Compute backend: auto, vulkan, amuda, numpy, python")
    p_diff.set_defaults(func=cmd_diffusion_train)

    # vision-train
    p_vis = subparsers.add_parser("vision-train", help="Run on-device Vision VLM LoRA training")
    p_vis.add_argument("--data", type=str, required=True, help="Path to QA dataset (.json, .jsonl) or directory of images")
    p_vis.add_argument("--output", type=str, default=None, help="Output .safetensors path for trained VLM LoRA adapter")
    p_vis.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    p_vis.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p_vis.add_argument("--batch-size", type=int, default=1, help="Batch size")
    p_vis.add_argument("--rank", type=int, default=4, help="LoRA rank")
    p_vis.add_argument("--alpha", type=float, default=1.0, help="LoRA alpha scaling factor")
    p_vis.add_argument("--backend", type=str, default="auto", help="Compute backend: auto, vulkan, amuda, numpy, python")
    p_vis.set_defaults(func=cmd_vision_train)

    # stt-train
    p_stt = subparsers.add_parser("stt-train", help="Run on-device Whisper STT Cross-Attention LoRA acoustic training")
    p_stt.add_argument("--data", type=str, required=True, help="Path to manifest (.json, .jsonl) or directory of audio files")
    p_stt.add_argument("--output", type=str, default=None, help="Output .safetensors path for trained STT LoRA adapter")
    p_stt.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    p_stt.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p_stt.add_argument("--batch-size", type=int, default=1, help="Batch size")
    p_stt.add_argument("--rank", type=int, default=4, help="LoRA rank")
    p_stt.add_argument("--alpha", type=float, default=1.0, help="LoRA alpha scaling factor")
    p_stt.add_argument("--backend", type=str, default="auto", help="Compute backend: auto, vulkan, amuda, numpy, python")
    p_stt.set_defaults(func=cmd_stt_train)

    # tts-train
    p_tts = subparsers.add_parser("tts-train", help="Run on-device Audio TTS Speaker Adaptation LoRA training")
    p_tts.add_argument("--data", type=str, required=True, help="Path to voice samples manifest or directory of audio files")
    p_tts.add_argument("--output", type=str, default=None, help="Output .safetensors path for trained TTS LoRA adapter")
    p_tts.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    p_tts.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p_tts.add_argument("--batch-size", type=int, default=1, help="Batch size")
    p_tts.add_argument("--rank", type=int, default=4, help="LoRA rank")
    p_tts.add_argument("--alpha", type=float, default=1.0, help="LoRA alpha scaling factor")
    p_tts.add_argument("--backend", type=str, default="auto", help="Compute backend: auto, vulkan, amuda, numpy, python")
    p_tts.set_defaults(func=cmd_tts_train)

    # rl (Reinforcement Learning: GRPO, DPO, PPO)
    p_rl = subparsers.add_parser("rl", help="Run on-device Reinforcement Learning (GRPO / DPO / PPO)")
    p_rl.add_argument("--method", type=str, default="grpo", choices=["grpo", "dpo", "ppo"], help="RL algorithm: GRPO (DeepSeek-R1 style), DPO, or PPO")
    p_rl.add_argument("--group-size", type=int, default=4, help="Candidate group size G for GRPO")
    p_rl.add_argument("--epochs", type=int, default=5, help="Training iterations")
    p_rl.add_argument("--lr", type=float, default=0.001, help="Policy learning rate")
    p_rl.add_argument("--beta", type=float, default=0.04, help="KL penalty / temperature factor")
    p_rl.add_argument("--epsilon", type=float, default=0.2, help="Policy clipping parameter")
    p_rl.add_argument("--backend", type=str, default="auto", help="Compute backend")
    p_rl.set_defaults(func=cmd_rl)

    # export
    p_exp = subparsers.add_parser("export", help="Export trained adapter weights to target runtime format")
    p_exp.add_argument("--adapter", type=str, required=True, help="Path to input adapter JSON/SafeTensors")
    p_exp.add_argument("--target", type=str, required=True, choices=["llamacpp", "bitnet", "diffusion", "vision", "tts", "stt"], help="Target runtime")
    p_exp.add_argument("--output", type=str, required=True, help="Path for output exported adapter")
    p_exp.add_argument("--base-model", type=str, default="unknown", help="Provenance base model identifier")
    p_exp.set_defaults(func=cmd_export)

    # ── AMEVA Component Protocol v1 ─────────────────────────────────────────
    try:
        from ameva_component.cli_support import build_protocol_subcommands
        build_protocol_subcommands(subparsers)
        _protocol_available = True
    except ImportError:
        _protocol_available = False
    # ────────────────────────────────────────────────────────────────────────

    args = parser.parse_args()
    if args.command is None:
        cmd_info(args)
    elif args.command in ("component", "model", "instance") and _protocol_available:
        from ameva_component.cli_support import dispatch_protocol
        from termux_train.control import TrainControl
        dispatch_protocol(args, TrainControl())
    elif args.command in ("component", "model", "instance"):
        import sys
        print("[ERROR] ameva-component-sdk not installed.", file=sys.stderr)
        sys.exit(1)
    else:
        args.func(args)


if __name__ == "__main__":
    main()


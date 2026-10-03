"""
AMEVA Heterogeneous Mobile Virtual RAM Pooling Distributed Training Tests.
Validates 44GB Virtual RAM Pooling across Fleet (S25, S21, S20, A53, A35),
Proportional & Uniform Layer Sharding, and Distributed Pipeline Sessions.
Component: [TRAIN-VIRTUAL-RAM-POOLING]
"""

import sys
import time
import socket
import pytest
from unittest.mock import patch, MagicMock

import termux_train as tt
from termux_train import Tensor, randn
from termux_train.cluster import (
    VirtualNodeInfo,
    VirtualRAMPool,
    DEFAULT_GUARD_BAND_MB,
    ClusterConnectionError,
    ClusterConfigurationError,
)
from termux_train.cluster_trainer import ClusterWorkerServer, ClusterPipelineSession
import termux_train.nn as nn
from termux_train.cli import main


def test_virtual_node_info_guard_band():
    """Verify safety guard-band (까치밥 메모리 300MB) is deducted to protect against Android LMK."""
    node = VirtualNodeInfo(
        endpoint="100.83.82.60:50052",
        total_ram_mb=8192,
        mem_available_mb=4500,
        guard_band_mb=300,
        soc="Exynos 2100",
        gpu="Mali-G78",
    )
    assert node.total_ram_mb == 8192
    assert node.mem_available_mb == 4500
    assert node.guard_band_mb == 300
    assert node.safe_usable_vram_mb == 4200  # 4500 - 300


def test_virtual_ram_pool_fleet_aggregation():
    """Verify 44GB physical RAM pool aggregation across 5-node mobile fleet."""
    pool = VirtualRAMPool(guard_band_mb=300)

    # 1. Galaxy S25 (12GB)
    pool.add_node("100.77.47.37:50052", total_ram_mb=12288, mem_available_mb=8000, soc="Snapdragon 8 Elite", gpu="Adreno 830")
    # 2. Galaxy S21 (8GB)
    pool.add_node("100.83.82.60:50052", total_ram_mb=8192, mem_available_mb=4500, soc="Exynos 2100", gpu="Mali-G78")
    # 3. Galaxy S20 (12GB)
    pool.add_node("100.106.99.81:50052", total_ram_mb=12288, mem_available_mb=7000, soc="Snapdragon 865", gpu="Adreno 650")
    # 4. Galaxy A53 (6GB)
    pool.add_node("100.106.183.109:50052", total_ram_mb=6144, mem_available_mb=3200, soc="Exynos 1280", gpu="Mali-G68")
    # 5. Galaxy A35 (6GB)
    pool.add_node("100.106.251.21:50052", total_ram_mb=6144, mem_available_mb=3400, soc="Exynos 1380", gpu="Mali-G68")

    assert pool.node_count == 5
    # Total raw physical RAM: 12GB + 8GB + 12GB + 6GB + 6GB = 44GB (45,056 MB)
    assert pool.total_pooled_ram_mb == 45056
    # Total safe usable memory: (8000-300) + (4500-300) + (7000-300) + (3200-300) + (3400-300) = 7700 + 4200 + 6700 + 2900 + 3100 = 24600 MB
    assert pool.total_safe_vram_mb == 24600

    summary = pool.get_summary()
    assert summary["total_nodes"] == 5
    assert summary["total_pooled_ram_mb"] == 45056


def test_virtual_ram_pool_sharding_calculation():
    """Verify layer allocation across nodes under proportional and uniform strategies."""
    pool = VirtualRAMPool()
    pool.add_node("node1:50052", total_ram_mb=12000, mem_available_mb=6300)  # Safe: 6000MB
    pool.add_node("node2:50052", total_ram_mb=6000, mem_available_mb=3300)   # Safe: 3000MB

    # Proportional sharding for 6 layers: 2:1 ratio -> 4 layers and 2 layers
    shards = pool.calculate_shards(num_layers=6, strategy="proportional")
    assert len(shards) == 2
    assert shards[0] == ("node1:50052", 0, 4)
    assert shards[1] == ("node2:50052", 4, 6)

    # Uniform sharding for 6 layers: 3 layers each
    shards_uni = pool.calculate_shards(num_layers=6, strategy="uniform")
    assert shards_uni[0] == ("node1:50052", 0, 3)
    assert shards_uni[1] == ("node2:50052", 3, 6)


def test_distributed_pipeline_session_convergence():
    """
    Spawns 2 lightweight loopback worker servers, partitions layers,
    and validates end-to-end forward/backward pipeline convergence.
    """
    port1 = 59121
    port2 = 59122

    server1 = ClusterWorkerServer(host="127.0.0.1", port=port1, backend="python")
    server2 = ClusterWorkerServer(host="127.0.0.1", port=port2, backend="python")

    server1.start()
    server2.start()
    time.sleep(0.1)

    try:
        pool = VirtualRAMPool()
        endpoints = [f"127.0.0.1:{port1}", f"127.0.0.1:{port2}"]
        session = ClusterPipelineSession(pool=pool, rpc_endpoints=endpoints, lr=0.05, backend="python")
        session.connect_all(timeout=3.0)

        # 2-layer MLP architecture across 2 workers: (16 -> 8) -> (8 -> 4)
        layer_specs = [
            {"type": "linear", "in_features": 16, "out_features": 8, "bias": False},
            {"type": "linear", "in_features": 8, "out_features": 4, "bias": False},
        ]
        shards = session.init_shards(layer_specs)
        assert len(shards) == 2

        criterion = nn.MSELoss()
        x_batch = randn((4, 16))
        y_batch = randn((4, 4))

        # Run 3 pipeline steps and observe loss reduction
        initial_loss = session.train_step(x_batch, y_batch, shards, criterion)
        step2_loss = session.train_step(x_batch, y_batch, shards, criterion)
        step3_loss = session.train_step(x_batch, y_batch, shards, criterion)

        assert step3_loss < initial_loss
        session.close()

    finally:
        server1.stop()
        server2.stop()


def test_distributed_pipeline_fail_fast_unreachable():
    """Verify Zero-Silent-Fallback: Unreachable worker raises ClusterConnectionError immediately."""
    pool = VirtualRAMPool()
    unreachable_endpoints = ["127.0.0.1:59999"]
    session = ClusterPipelineSession(pool=pool, rpc_endpoints=unreachable_endpoints)

    with pytest.raises(ClusterConnectionError) as exc_info:
        session.connect_all(timeout=0.5)

    assert "ClusterPipelineSession could not connect to 1 worker" in str(exc_info.value)


def test_cli_cluster_probe_args(monkeypatch):
    """Verify CLI accepts cluster-probe subcommand."""
    test_args = [
        "termux-train",
        "cluster-probe",
        "--rpc",
        "100.77.47.37:50052,100.83.82.60:50052",
        "--guard-band",
        "250",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("termux_train.cli.cmd_cluster_probe") as mock_probe:
        try:
            main()
        except SystemExit as e:
            assert e.code == 0 or e.code is None
        assert mock_probe.call_count == 1
        args = mock_probe.call_args[0][0]
        assert args.rpc == "100.77.47.37:50052,100.83.82.60:50052"
        assert args.guard_band == 250

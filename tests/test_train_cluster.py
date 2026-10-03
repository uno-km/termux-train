"""
AMEVA Unified Distributed RPC Cluster Orchestration Cold Tests for termux-train.
Strict Protocol: Zero-Silent-Fallback & Fail-Fast Verification.
Component: [TRAIN-CLUSTER]
"""

import socket
import sys
import json
from unittest.mock import patch, MagicMock
import pytest

import termux_train as tt
from termux_train.exceptions import (
    ClusterConnectionError,
    ClusterConfigurationError,
)
from termux_train.cluster import (
    parse_cluster_rpc_spec,
    verify_rpc_cluster_nodes,
    verify_rpc_cluster_health,
)
from termux_train.runtime.runner import run_session
from termux_train.cli import main


def test_parse_cluster_rpc_spec_valid():
    """Verify parsing and normalization of valid RPC specs."""
    servers_str = "100.77.47.37:50052, 100.106.99.81:50052"
    parsed = parse_cluster_rpc_spec(servers_str)
    assert parsed == ["100.77.47.37:50052", "100.106.99.81:50052"]

    servers_list = ["100.77.47.37:50052", "100.106.99.81:50052"]
    parsed2 = parse_cluster_rpc_spec(servers_list)
    assert parsed2 == ["100.77.47.37:50052", "100.106.99.81:50052"]

    assert parse_cluster_rpc_spec(None) == []
    assert parse_cluster_rpc_spec("") == []


def test_parse_cluster_rpc_spec_invalid():
    """Verify invalid RPC formats strictly raise ClusterConfigurationError."""
    with pytest.raises(ClusterConfigurationError):
        parse_cluster_rpc_spec("invalid_host_no_port")

    with pytest.raises(ClusterConfigurationError):
        parse_cluster_rpc_spec("host:not_a_number")

    with pytest.raises(ClusterConfigurationError):
        parse_cluster_rpc_spec("host:999999")  # Port out of range

    with pytest.raises(ClusterConfigurationError):
        parse_cluster_rpc_spec(12345)  # Invalid type


def test_verify_rpc_cluster_nodes_success():
    """Verify pre-flight check succeeds when all RPC nodes accept TCP connection."""
    servers = ["100.77.47.37:50052", "100.106.99.81:50052"]
    with patch("socket.create_connection") as mock_conn:
        mock_conn.return_value.__enter__.return_value = MagicMock()
        verify_rpc_cluster_nodes(servers, timeout=1.0)
        assert mock_conn.call_count == 2


def test_verify_rpc_cluster_nodes_fail_fast():
    """Verify Zero-Silent-Fallback: Unreachable node raises ClusterConnectionError immediately."""
    servers = ["100.77.47.37:50052", "100.106.99.81:50052"]

    def fake_connect(addr, timeout=None):
        host, port = addr
        if host == "100.106.99.81":
            raise ConnectionRefusedError("Connection refused by test worker")
        return MagicMock()

    with patch("socket.create_connection", side_effect=fake_connect):
        with pytest.raises(ClusterConnectionError) as exc_info:
            verify_rpc_cluster_nodes(servers, timeout=1.0)
        assert "Zero-Silent-Fallback Violation Prevented" in str(exc_info.value)
        assert "100.106.99.81:50052" in str(exc_info.value)


def test_verify_rpc_cluster_health_reporting():
    """Verify granular diagnostics from verify_rpc_cluster_health."""
    servers = ["100.77.47.37:50052", "100.106.99.81:50052"]

    def fake_connect(addr, timeout=None):
        host, port = addr
        if host == "100.106.99.81":
            raise socket.timeout("Timed out")
        return MagicMock()

    with patch("socket.create_connection", side_effect=fake_connect):
        health = verify_rpc_cluster_health(servers, timeout=1.0)
        assert health["all_healthy"] is False
        assert health["nodes"]["100.77.47.37:50052"]["reachable"] is True
        assert health["nodes"]["100.106.99.81:50052"]["reachable"] is False


def test_train_runner_cluster_spec_and_preflight(capsys):
    """Verify run_session verifies nodes and emits cluster metrics."""
    cfg = {
        "modelType": "mlp",
        "dim": 16,
        "epochs": 1,
        "lr": 0.01,
        "batchSize": 4,
        "cluster_rpc_servers": "100.77.47.37:50052,100.106.99.81:50052",
        "cluster_tensor_split": "60,40",
        "cluster_split_mode": "tensor",
        "cluster_vram_budget": "4000,4000",
    }

    with patch("termux_train.runtime.runner.verify_rpc_cluster_nodes") as mock_verify:
        run_session(cfg)
        assert mock_verify.called
        assert mock_verify.call_args[0][0] == ["100.77.47.37:50052", "100.106.99.81:50052"]

    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")

    metric_found = False
    for line in lines:
        if line.startswith("__METRICS__:"):
            metric_data = json.loads(line.replace("__METRICS__:", ""))
            if metric_data.get("event") == "step":
                assert metric_data.get("distributed") is True
                assert metric_data.get("rpc_nodes") == ["100.77.47.37:50052", "100.106.99.81:50052"]
                assert metric_data.get("tensor_split") == [60.0, 40.0]
                assert metric_data.get("split_mode") == "tensor"
                assert metric_data.get("vram_budget") == "4000,4000"
                metric_found = True
                break

    assert metric_found, "Distributed metrics event was not emitted in run_session"


def test_train_runner_cluster_unreachable_node_fail_fast():
    """Verify run_session immediately terminates if a cluster node is unreachable."""
    cfg = {
        "modelType": "mlp",
        "dim": 16,
        "epochs": 1,
        "lr": 0.01,
        "batchSize": 4,
        "cluster_rpc_servers": "192.168.0.220:50052,192.168.0.253:50052",
    }

    def fake_connect(addr, timeout=None):
        raise ConnectionRefusedError("Node connection refused")

    with patch("socket.create_connection", side_effect=fake_connect):
        with pytest.raises(ClusterConnectionError):
            run_session(cfg)


def test_cli_train_cluster_flags_parsing(monkeypatch):
    """Verify CLI parses cluster arguments and passes to cmd_train."""
    test_args = [
        "termux-train",
        "train",
        "--model",
        "mlp",
        "--epochs",
        "1",
        "--cluster-rpc-servers",
        "100.77.47.37:50052,100.106.99.81:50052",
        "--cluster-tensor-split",
        "50,50",
        "--cluster-split-mode",
        "tensor",
        "--cluster-vram-budget",
        "4096,4096",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("termux_train.cli.cmd_train") as mock_cmd_train:
        try:
            main()
        except SystemExit as e:
            assert e.code == 0 or e.code is None

        assert mock_cmd_train.call_count == 1
        args = mock_cmd_train.call_args[0][0]
        assert args.cluster_rpc_servers == "100.77.47.37:50052,100.106.99.81:50052"
        assert args.cluster_tensor_split == "50,50"
        assert args.cluster_split_mode == "tensor"
        assert args.cluster_vram_budget == "4096,4096"

import sys
import json
from unittest.mock import patch, MagicMock
import pytest

from termux_train.cli import main
from termux_train.runtime.runner import run_session


def test_train_cli_rpc_arg_parsing(monkeypatch):
    test_args = [
        "termux-train",
        "train",
        "--model",
        "mlp",
        "--epochs",
        "1",
        "--rpc",
        "192.168.0.220:50052,192.168.0.253:50052",
        "-ts",
        "50,50",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("termux_train.cli.cmd_train") as mock_cmd_train:
        # Prevent actual execution but verify arg parsing
        try:
            main()
        except SystemExit as e:
            assert e.code == 0 or e.code is None

        assert mock_cmd_train.call_count == 1
        args = mock_cmd_train.call_args[0][0]
        assert args.rpc == "192.168.0.220:50052,192.168.0.253:50052"
        assert args.tensor_split == "50,50"


def test_train_runner_rpc_session_metrics(capsys):
    cfg = {
        "modelType": "mlp",
        "dim": 16,
        "epochs": 1,
        "lr": 0.01,
        "batchSize": 4,
        "rpc": "192.168.0.220:50052,192.168.0.253:50052",
        "tensorSplit": "60,40",
    }

    with patch("termux_train.runtime.runner.verify_rpc_cluster_nodes"):
        run_session(cfg)

    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")
    
    # Check if __METRICS__ contains distributed metadata
    metric_found = False
    for line in lines:
        if line.startswith("__METRICS__:"):
            metric_data = json.loads(line.replace("__METRICS__:", ""))
            if metric_data.get("event") == "step":
                assert metric_data.get("distributed") is True
                assert metric_data.get("rpc_nodes") == ["192.168.0.220:50052", "192.168.0.253:50052"]
                assert metric_data.get("tensor_split") == [60.0, 40.0]
                assert metric_data.get("node_count") == 2
                metric_found = True
                break

    assert metric_found, "Distributed metrics event was not emitted in run_session"


def test_train_runner_rpc_fail_fast_invalid_format():
    # 1. Invalid node format without port
    cfg_invalid_node = {
        "modelType": "mlp",
        "epochs": 1,
        "rpc": "192.168.0.220,192.168.0.253",
    }
    with pytest.raises(ValueError, match=r"\[FAIL-FAST\] Invalid RPC address"):
        run_session(cfg_invalid_node)

    # 2. Split count mismatch
    cfg_mismatch_split = {
        "modelType": "mlp",
        "epochs": 1,
        "rpc": "192.168.0.220:50052,192.168.0.253:50052",
        "tensorSplit": "50,25,25,10",  # 4 splits for 2 nodes
    }
    with pytest.raises(ValueError, match=r"\[FAIL-FAST\] Tensor split count mismatch"):
        run_session(cfg_mismatch_split)

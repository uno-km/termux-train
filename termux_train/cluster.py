"""
AMEVA Unified Distributed RPC Cluster Orchestration Utility.
Strict Protocol: Zero-Silent-Fallback & Zero-Deception ABI.
Component: [TRAIN-CLUSTER]
"""

import socket
import logging
from typing import List, Optional, Union, Dict, Any

from termux_train.exceptions import (
    ClusterConnectionError,
    ClusterConfigurationError,
)

logger = logging.getLogger(__name__)


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

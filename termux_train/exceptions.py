"""
Unified Cluster & Distributed Execution Exception Hierarchy for termux-train.
Enforces Zero-Silent-Fallback & Fail-Fast Engineering Principles.
"""


class ClusterConfigurationError(ValueError):
    """Raised when cluster RPC parameters, topology, or tensor split specifications are malformed."""
    pass


class ClusterConnectionError(RuntimeError):
    """Raised when one or more cluster RPC nodes cannot be reached or fail health checks."""
    pass

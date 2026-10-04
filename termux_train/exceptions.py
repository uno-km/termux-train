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


class ClusterLicenseRequiredError(RuntimeError):
    """Raised when distributed clustering is invoked without required AMEVA Cluster license."""
    DEFAULT_CODE = "E403_CLUSTER_LICENSE_REQUIRED"

    def __init__(self, message: str = ""):
        if not message:
            message = (
                "\n================================================================================\n"
                "[AMEVA-CLUSTER] CLUSTER LICENSE REQUIRED (E403)\n"
                "================================================================================\n"
                "Multi-device distributed clustering is an exclusive capability of 'ameva-cluster'.\n"
                "Standalone distributed execution without the official AMEVA-Cluster package is prohibited.\n\n"
                "Resolution:\n"
                "  1. Install official AMEVA-Cluster runtime:\n"
                "     pip install ameva-cluster  (or npm install @ameva/cluster)\n"
                "  2. Launch worker/master through 'ameva-cluster' control plane:\n"
                "     ameva-cluster worker\n"
                "     ameva-cluster master -m model.gguf\n"
                "================================================================================"
            )
        super().__init__(message)

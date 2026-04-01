"""Re-export local worker. Cloud worker is not included in vimin-core."""
from vimin_core.core.local_worker import LocalWorker

__all__ = ["LocalWorker"]

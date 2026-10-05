import logging
import os

from datetime import datetime

from from_root import from_root


def _resolve_project_root() -> str:
    """
    from_root anchors on a .git or .project-root marker. Container images are built
    without the .git directory, so the current working directory is used as a fallback.
    """
    try:
        return str(from_root())
    except Exception:
        return os.getcwd()


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_dir = os.path.join(_resolve_project_root(), "logs")

os.makedirs(log_dir, exist_ok=True)

logs_path = os.path.join(log_dir, LOG_FILE)


logging.basicConfig(
    filename=logs_path,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

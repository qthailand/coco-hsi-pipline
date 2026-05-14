import sys
import logging
from pathlib import Path

# Must be added BEFORE any pipeline imports so bare imports inside GUI.py work
# (e.g. `from feature_extraction import ...`)
_PIPELINE_DIR = str(Path(__file__).parent / "pipeline")
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from PyQt5.QtWidgets import QApplication
from GUI import HSIPipelineWindow

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy third-party loggers
logging.getLogger("spectral").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = HSIPipelineWindow()
    window.show()
    sys.exit(app.exec_())

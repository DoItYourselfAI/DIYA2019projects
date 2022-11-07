from pathlib import Path

# IO directories
PROJECT_ROOT = Path(__file__).resolve().parent.as_posix()
RES_DIR = 'res'
TMP_DIR = 'tmp'
LOG_DIR = 'logs'
PTH_DIR = 'checkpoints'

# Computation options
EPS = 1e-5
INF = 1e5

# Logging levels
LOG_LEVELS = {
    'DEBUG': {'lvl': 10, 'color': 'cyan'},
    'INFO': {'lvl': 20, 'color': 'white'},
    'WARNING': {'lvl': 30, 'color': 'yellow'},
    'ERROR': {'lvl': 40, 'color': 'red'},
    'CRITICAL': {'lvl': 50, 'color': 'red'},
}

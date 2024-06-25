import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format=(
        "<green><level>GBMLGG:" " </level></green><blue><level>{message}</level></blue>"
    ),
)

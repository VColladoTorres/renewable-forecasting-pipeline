"""Standardised logger."""
import logging, sys
FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
def init_logger(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=FORMAT,
                        handlers=[logging.StreamHandler(sys.stdout)])

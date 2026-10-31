# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import sys
from logging import Filter, Formatter, Logger, LogRecord, StreamHandler
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

FORMATTER = Formatter("[phyagi] [%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s")
LOG_LEVEL = logging.DEBUG

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
MPI_RANK = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", None)
if MPI_RANK is not None:
    LOCAL_RANK = int(MPI_RANK)


class _RankFilter(Filter):
    def __init__(self, rank: int) -> None:
        self.rank = rank

    def filter(self, record: LogRecord) -> bool:
        return self.rank == 0


def _get_console_handler() -> StreamHandler:
    console_handler = StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)

    return console_handler


def _get_timed_file_handler(file_path: Optional[str] = None) -> TimedRotatingFileHandler:
    file_path = file_path or "phyagi.log"

    file_handler = TimedRotatingFileHandler(file_path, delay=True, when="midnight", encoding="utf-8")
    file_handler.setFormatter(FORMATTER)

    return file_handler


def get_logger(logger_name: str, file_path: Optional[str] = None, rank_filter: bool = True) -> Logger:
    """Get a logger with the specified name and default settings.

    Args:
        logger_name: Name of the logger.
        file_path: Path to the file to which log messages will be written.
        rank_filter: Whether to apply a filter that prints only to the GPU with ``local_rank=0``.

    Returns:
        A :class:`logging.Logger` instance with the specified name and default settings.

    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(LOG_LEVEL)

    logger.addHandler(_get_console_handler())
    logger.addHandler(_get_timed_file_handler(file_path=file_path))

    if rank_filter:
        logger.addFilter(_RankFilter(LOCAL_RANK))

    logger.propagate = False

    return logger

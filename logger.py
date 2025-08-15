import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from rich.console import Console
from rich.json import JSON as RichJSON

from config import config


console = Console()


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "filename": record.filename,
            "lineno": record.lineno,
            "module": record.module,
            "funcName": record.funcName,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record, indent=4)


class ColourJSONFormatter(logging.Formatter):
    """Formatter for colorized JSON logs (human-readable)."""

    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "filename": record.filename,
            "lineno": record.lineno,
            "module": record.module,
            "funcName": record.funcName,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        console.print(RichJSON(json.dumps(log_record, indent=4)))
        return ""


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.makedirs(os.path.dirname(config.log_file_path), exist_ok=True)
file_handler = RotatingFileHandler(
    config.log_file_path, maxBytes=1024 * 1024, backupCount=5
)
json_formatter = JSONFormatter()
file_handler.setFormatter(json_formatter)

stream_handler = logging.StreamHandler(sys.stdout)
colour_json_formatter = ColourJSONFormatter()
stream_handler.setFormatter(colour_json_formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

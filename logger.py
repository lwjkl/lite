import json
import logging
import sys
from logging.handlers import RotatingFileHandler

from settings import settings


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
        return json.dumps(log_record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(settings.log_file_path, maxBytes=1024*1024, backupCount=5)
stream_handler = logging.StreamHandler(sys.stdout)

json_formatter = JSONFormatter()
file_handler.setFormatter(json_formatter)
stream_handler.setFormatter(json_formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

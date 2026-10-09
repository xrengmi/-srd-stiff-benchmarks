import logging
import sys
import json

class StructLogAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        if kwargs:
            extra_str = " ".join([f"{k}={v}" for k, v in kwargs.items() if k not in ["exc_info", "stack_info", "extra"]])
            msg = f"{msg} | {extra_str}"
            # Sadece standart logging kwargs'larını sakla
            kwargs = {k: v for k, v in kwargs.items() if k in ["exc_info", "stack_info", "extra"]}
        return msg, kwargs

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
    return StructLogAdapter(logger, {})


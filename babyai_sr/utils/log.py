import os
import sys
import logging

import babyai.utils


def get_log_dir(log_name):
    return os.path.join(babyai.utils.storage_dir(), "logs", log_name)


def get_log_path(log_name):
    return os.path.join(get_log_dir(log_name), "log.log")


def configure_logging(log_name, stream=False):
    path = get_log_path(log_name)
    babyai.utils.create_folders_if_necessary(path)
    
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(name)s: %(asctime)s: %(message)s")
    
    file_handler = logging.FileHandler(filename=path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：config.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''

import os
import logging
from dotenv import load_dotenv
from utils.loggers.logger import *

load_dotenv()

level = logging.INFO
leve_str = os.getenv("LOG_LEVEL") # DEBUG, INFO, WARNING, ERROR, CRITICAL
if leve_str == 'DEBUG':
    level = logging.DEBUG
elif leve_str == 'INFO':
    pass
elif leve_str == 'WARNING':
    level = logging.WARNING
elif leve_str == 'ERROR':
    level = logging.ERROR
elif leve_str == 'CRITICAL':
    level = logging.CRITICAL
else:
    level = logging.DEBUG

initTimedRotateLog(os.getenv("LOG_PATH"), level)


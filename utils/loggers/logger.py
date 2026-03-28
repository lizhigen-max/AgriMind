#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：logger.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''

__author__ = 'Zhigen.li'

import os
import logging
from logging.handlers import TimedRotatingFileHandler


def initLogWithHandler(handler, level):
    FORMAT = "%(asctime)s %(levelname)s %(thread)d %(filename)s %(funcName)s %(message)s"

    logging.basicConfig(format=FORMAT, level=level)
    logging.getLogger().setLevel(level)

    handler.setLevel(level)
    formatter = logging.Formatter(FORMAT)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

def initTimedRotateLog(logFile, level=logging.INFO):
    """
    初始化日志系统，日志系统按天生成日志
    :param logFile: 包含全路径的文件名
    :return:无返回，出错则抛出异常
    """
    logPath = os.path.dirname(logFile)
    if not os.path.exists(logPath):
        os.makedirs(logPath)

    handler = TimedRotatingFileHandler(logFile, when='MIDNIGHT', backupCount=30, encoding='utf-8')
    initLogWithHandler(handler, level)





#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：__init__.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''

"""
RAG全自动化评估框架

基于Ragas指标的专业RAG系统评估解决方案
"""

from .RAGAS_Evaluator import (RAGASConfig, RAGAS_Evaluator, example_usage)

__version__ = "1.0.0"
__author__ = "Zhigen.li"

__all__ = [
    'RAGASConfig',
    'RAGAS_Evaluator',
    'example_usage'
]

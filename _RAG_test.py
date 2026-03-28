#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：_RAG_test.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''

from utils.RAG.RAGSaver import RAGSaver, create_index_from_folder
from config import *


create_index_from_folder('./Document', db_type='milvus')
# RAGSaver.test_search('葡萄冬季怎么修剪')
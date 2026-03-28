#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：MetadataExtractor.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''
"""
元数据提取模块 - 使用LLM提取结构化过滤条件
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
import os
import logging

from .RAGConfig import RAGConfig



class KeyWord(BaseModel):
    """结构化关键字信息"""
    file_name: Optional[str] = Field(None, description="文件名称")
    department: Optional[str] = Field(None, description="部门")
    year: Optional[int] = Field(None, description="年份")
    doc_type: Optional[str] = Field(None, description="文件类型：技术文档、生产文档等")
    author: Optional[str] = Field(None, description="作者")


class MetadataExtractor:
    """元数据提取器"""

    def __init__(self, model=None, config: RAGConfig=RAGConfig()):
        self.config = config
        self._model = model

        self._init_model()

    def _init_model(self):
        """初始化LLM模型"""
        try:
            if not self._model:
                self._model = init_chat_model(
                    "deepseek:deepseek-chat",
                    api_key=self.config.deepseek_api_key
                )
                self._structured_llm = self._model.with_structured_output(KeyWord)
            logging.info("✓ 已初始化DeepSeek模型用于元数据提取")
        except Exception as e:
            logging.error(f"初始化LLM失败: {e}")

    def extract_filters(self, query: str) -> Dict[str, Any]:
        """
        从查询中提取过滤条件（同步）

        Args:
            query: 用户查询

        Returns:
            过滤条件字典
        """
        if not self._model:
            logging.warning("LLM未初始化，跳过元数据提取")
            return {}

        try:
            result = self._structured_llm.invoke(
                f"从用户查询中提取文件名称、部门、年份、文档类型、作者，查询：{query}"
            )

            logging.info(f"提取结果: file_name={result.file_name}, "
                        f"department={result.department}, year={result.year}")

            filters = {}
            if result.file_name:
                filters["file_name"] = result.file_name
            if result.department:
                filters["department"] = result.department
            if result.year:
                filters["year"] = result.year
            if result.doc_type:
                filters["doc_type"] = result.doc_type
            if result.author:
                filters["author"] = result.author

            return filters

        except Exception as e:
            logging.error(f"元数据提取失败: {e}")
            return {}

    async def aextract_filters(self, query: str) -> Dict[str, Any]:
        """异步提取过滤条件"""
        if not self._model:
            return {}

        try:
            result = await self._structured_llm.ainvoke(
                f"从用户查询中提取文件名称、部门、年份、文档类型、作者，查询：{query}"
            )

            filters = {}
            if result.file_name:
                filters["file_name"] = result.file_name
            if result.department:
                filters["department"] = result.department
            if result.year:
                filters["year"] = result.year
            if result.doc_type:
                filters["doc_type"] = result.doc_type
            if result.author:
                filters["author"] = result.author

            return filters

        except Exception as e:
            logging.error(f"异步元数据提取失败: {e}")
            return {}
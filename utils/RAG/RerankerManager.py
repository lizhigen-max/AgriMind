#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：RerankerManager.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''
"""
重排序管理模块
"""
import torch
from typing import List, Tuple
from langchain_core.documents import Document
from modelscope import AutoModelForSequenceClassification, AutoTokenizer
import logging
import asyncio

from .RAGConfig import RerankerConfig


class RerankerManager:
    """重排序管理器"""

    def __init__(self, config: RerankerConfig, tokenizer=None, model=None):
        self.config = config
        self._tokenizer = tokenizer
        self._model = model

        if config.enabled:
            self._load_model()

    def _load_model(self):
        """加载重排序模型"""
        try:
            if not self._tokenizer:
                self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            if not self._model:
                self._model = AutoModelForSequenceClassification.from_pretrained(self.config.model_path)
                self._model.eval()
                logging.info(f"✓ 已加载重排序模型: {self.config.model_path}")
        except Exception as e:
            logging.error(f"加载重排序模型失败: {e}")
            raise

    def rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        重排序文档（同步）

        Args:
            query: 查询文本
            documents: 待排序文档列表

        Returns:
            (文档, 分数)列表，按分数降序排列
        """
        if not self.config.enabled or not documents:
            return [(doc, 0.0) for doc in documents]

        pairs = [[query, doc.page_content] for doc in documents]

        with torch.no_grad():
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.config.max_length
            )
            logits = self._model(**inputs, return_dict=True).logits.view(-1).float()
            scores = torch.sigmoid(logits)

        # 筛选并排序
        results = []
        for doc, score in zip(documents, scores):
            score_val = score.item()
            if score_val > self.config.threshold:
                new_metadata = doc.metadata.copy() if doc.metadata else {}
                new_metadata["rerank_score"] = score_val

                new_doc = Document(
                    page_content=doc.page_content,
                    metadata=new_metadata
                )
                results.append((new_doc, score_val))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    async def arerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """异步重排序"""
        if not self.config.enabled or not documents:
            return [(doc, 0.0) for doc in documents]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.rerank, query, documents)

    def rerank_and_filter(self, query: str, documents: List[Document]) -> List[Document]:
        """
        重排序并过滤低分文档（同步）

        Returns:
            过滤后的Document列表
        """
        results = self.rerank(query, documents)
        return [doc for doc, _ in results]

    async def arerank_and_filter(self, query: str, documents: List[Document]) -> List[Document]:
        """异步重排序并过滤"""
        results = await self.arerank(query, documents)
        return [doc for doc, _ in results]
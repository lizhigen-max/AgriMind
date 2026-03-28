#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：RAGProcessor.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''
"""
RAG处理器主模块 - 统一入口，支持所有检索模式
"""
from typing import List, Optional, Dict, Any, Union
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import asyncio
import logging

from .RAGConfig import RAGConfig, VectorModelConfig, VectorDBConfig, CacheConfig, RerankerConfig, RetrievalConfig
from .VectorStoreFactory import VectorStoreFactory
from .CacheManager import CacheManager, cached_retrieval
from .RerankerManager import RerankerManager
from .MetadataExtractor import MetadataExtractor


class RAGProcessor:
    """
    RAG处理器 - 统一检索接口

    支持:
    - 向量检索 (FAISS/Chroma)
    - BM25关键词检索
    - 混合检索 (向量+BM25)
    - 重排序
    - 标签过滤
    - 本地缓存
    - 同步/异步操作
    """

    def __init__(self, llm_model=None, config: Optional[RAGConfig] = None, reranker_tokenizer=None, reranker_model=None, vector_model=None):
        self.llm_model= llm_model
        self.reranker_tokenizer = reranker_tokenizer
        self.reranker_model = reranker_model
        self.vector_model = vector_model
        self.config = config or RAGConfig()
        self.vector_store_factory: Optional[VectorStoreFactory] = None
        self.cache_manager: Optional[CacheManager] = None
        self.reranker_manager: Optional[RerankerManager] = None
        self.metadata_extractor: Optional[MetadataExtractor] = None
        self._bm25_retriever: Optional[BM25Retriever] = None

        self._initialize_components()

    def _initialize_components(self):
        """初始化所有组件"""
        # 缓存管理器
        self.cache_manager = CacheManager(self.config.cache)

        # 重排序管理器
        self.reranker_manager = RerankerManager(self.config.reranker, self.reranker_tokenizer, self.reranker_model)

        # 元数据提取器
        self.metadata_extractor = MetadataExtractor(self.llm_model, self.config)

        logging.info("✓ RAG处理器初始化完成")

    def load_vector_store(self, db_config: Optional[VectorDBConfig] = None,
                          model_config: Optional[VectorModelConfig] = None) -> None:
        """
        加载向量存储

        Args:
            db_config: 向量数据库配置，默认使用self.config.vector_db
            model_config: 向量模型配置，默认使用self.config.vector_model
        """
        db_config = db_config or self.config.vector_db
        model_config = model_config or self.config.vector_model

        self.vector_store_factory = VectorStoreFactory(db_config, model_config, self.vector_model)
        self.vector_store_factory.load_vector_store()

        # 初始化BM25检索器
        self._init_bm25_retriever()

        logging.info(f"✓ 已加载向量存储: {db_config.db_type.value}")

    def create_vector_store(self, documents: List[Document],
                            db_config: Optional[VectorDBConfig] = None,
                            model_config: Optional[VectorModelConfig] = None) -> None:
        """
        创建新的向量存储

        Args:
            documents: 初始文档列表
            db_config: 向量数据库配置
            model_config: 向量模型配置
        """
        db_config = db_config or self.config.vector_db
        model_config = model_config or self.config.vector_model

        self.vector_store_factory = VectorStoreFactory(db_config, model_config)
        self.vector_store_factory.create_vector_store(documents)
        self.vector_store_factory.save()

        # 初始化BM25
        self._init_bm25_retriever()

        logging.info(f"✓ 已创建向量存储: {db_config.db_type.value}")

    def _init_bm25_retriever(self):
        """初始化BM25检索器"""
        if self.vector_store_factory is None:
            raise RuntimeError("向量存储未初始化")

        all_docs = self.vector_store_factory.get_all_documents()
        self._bm25_retriever = BM25Retriever.from_documents(all_docs)
        self._bm25_retriever.k = self.config.retrieval.bm25_k
        logging.info(f"✓ BM25检索器已初始化，文档数: {len(all_docs)}")

    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到现有存储"""
        if self.vector_store_factory is None:
            raise RuntimeError("向量存储未初始化")

        self.vector_store_factory.add_documents(documents)
        self.vector_store_factory.save()

        # 重新初始化BM25
        self._init_bm25_retriever()

    # ==================== 检索方法 ====================

    @cached_retrieval(cache_attr_name='cache_manager')
    def vector_search(self, query: str, k: Optional[int] = None,
                      filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        纯向量检索（同步）

        Args:
            query: 查询文本
            k: 返回数量，默认使用配置
            filters: 元数据过滤条件

        Returns:
            Document列表
        """
        if self.vector_store_factory is None:
            raise RuntimeError("向量存储未初始化")

        k = k or self.config.retrieval.vector_k
        search_kwargs = {"k": k}
        if filters:
            search_kwargs["filter"] = filters

        retriever = self.vector_store_factory.get_retriever(search_kwargs)
        return retriever.invoke(query)

    async def avector_search(self, query: str, k: Optional[int] = None,
                             filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """纯向量检索（异步）"""
        if self.vector_store_factory is None:
            raise RuntimeError("向量存储未初始化")

        # 检查缓存
        if self.config.cache.enabled:
            cached = await self.cache_manager.aget_cached_results(query)
            if cached is not None:
                return cached

        k = k or self.config.retrieval.vector_k
        search_kwargs = {"k": k}
        if filters:
            search_kwargs["filter"] = filters

        retriever = self.vector_store_factory.get_retriever(search_kwargs)

        # 异步执行检索
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(None, retriever.invoke, query)

        # 缓存结果
        if self.config.cache.enabled:
            await self.cache_manager.acache_results(query, docs)

        return docs

    @cached_retrieval(cache_attr_name='cache_manager')
    def bm25_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        BM25关键词检索（同步）

        Args:
            query: 查询文本
            k: 返回数量

        Returns:
            Document列表
        """
        if self._bm25_retriever is None:
            raise RuntimeError("BM25检索器未初始化")

        k = k or self.config.retrieval.bm25_k
        self._bm25_retriever.k = k
        return self._bm25_retriever.invoke(query)

    async def abm25_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """BM25关键词检索（异步）"""
        if self._bm25_retriever is None:
            raise RuntimeError("BM25检索器未初始化")

        # 检查缓存
        if self.config.cache.enabled:
            cached = await self.cache_manager.aget_cached_results(query)
            if cached is not None:
                return cached

        k = k or self.config.retrieval.bm25_k
        self._bm25_retriever.k = k

        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(None, self._bm25_retriever.invoke, query)

        if self.config.cache.enabled:
            await self.cache_manager.acache_results(query, docs)

        return docs

    # @cached_retrieval(cache_attr_name='cache_manager')
    def ensemble_search(self, query: str, k: Optional[int] = None,
                        filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        混合检索：向量 + BM25（同步）

        Args:
            query: 查询文本
            k: 返回数量
            filters: 向量检索的过滤条件

        Returns:
            Document列表
        """
        if self.vector_store_factory is None or self._bm25_retriever is None:
            raise RuntimeError("检索器未初始化")

        k = k or self.config.retrieval.vector_k

        # 配置检索器
        vector_retriever = self.vector_store_factory.get_retriever({
            "k": k,
            "filter": filters
        })
        self._bm25_retriever.k = k

        # 混合检索
        ensemble = EnsembleRetriever(
            retrievers=[self._bm25_retriever, vector_retriever],
            weights=self.config.retrieval.ensemble_weights
        )

        return ensemble.invoke(query)

    async def aensemble_search(self, query: str, k: Optional[int] = None,
                               filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """混合检索（异步）"""
        # 检查缓存
        if self.config.cache.enabled:
            cached = await self.cache_manager.aget_cached_results(query)
            if cached is not None:
                return cached

        # 并行执行两种检索
        k = k or self.config.retrieval.vector_k

        async def vector_task():
            search_kwargs = {"k": k}
            if filters:
                search_kwargs["filter"] = filters
            retriever = self.vector_store_factory.get_retriever(search_kwargs)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, retriever.invoke, query)

        async def bm25_task():
            self._bm25_retriever.k = k
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._bm25_retriever.invoke, query)

        vector_docs, bm25_docs = await asyncio.gather(vector_task(), bm25_task())

        # 合并结果（不去重，后续重排序会处理）
        all_docs = vector_docs + bm25_docs

        if self.config.cache.enabled:
            await self.cache_manager.acache_results(query, all_docs)

        return all_docs

    @cached_retrieval(cache_attr_name='cache_manager')
    def ensemble_search_with_rerank(self, query: str, k: Optional[int] = None,
                                    filters: Optional[Dict[str, Any]] = None,
                                    threshold: Optional[float] = None) -> List[Document]:
        """
        混合检索 + 重排序（同步）

        Args:
            query: 查询文本
            k: 返回数量
            filters: 过滤条件
            threshold: 重排序阈值，默认使用配置

        Returns:
            重排序后的Document列表
        """
        # 先执行混合检索
        docs = self.ensemble_search(query, k=k, filters=filters)
        logging.info('已完成混合检索，检索到{}个片段'.format(len(docs)))

        if not docs:
            return []

        # 重排序
        threshold = threshold or self.config.reranker.threshold
        reranked = self.reranker_manager.rerank(query, docs)
        logging.info('已完成重排序')

        # 过滤并返回
        return [doc for doc, score in reranked if score > threshold]

    async def aensemble_search_with_rerank(self, query: str, k: Optional[int] = None,
                                           filters: Optional[Dict[str, Any]] = None,
                                           threshold: Optional[float] = None,
                                           streaming_callback = None) -> List[Document]:
        """混合检索 + 重排序（异步）"""
        # 检查缓存
        if self.config.cache.enabled:
            cached = await self.cache_manager.aget_cached_results(query)
            if cached is not None:
                return cached

        # 异步混合检索
        docs = await self.aensemble_search(query, k=k, filters=filters)
        if streaming_callback:
            await streaming_callback(f'混合检索到{len(docs)}个文档')

        if not docs:
            return []

        # 异步重排序
        threshold = threshold or self.config.reranker.threshold
        reranked = await self.reranker_manager.arerank(query, docs)

        result = [doc for doc, score in reranked if score > threshold]
        if streaming_callback:
            await streaming_callback(f'重排序得到{len(result)}个文档')

        if self.config.cache.enabled:
            await self.cache_manager.acache_results(query, result)

        return result

    @cached_retrieval(cache_attr_name='cache_manager')
    def filtered_search(self, query: str, k: Optional[int] = None,
                        use_reranker: bool = True) -> List[Document]:
        """
        智能过滤检索：自动提取过滤条件 + 检索 + 重排序（同步）

        Args:
            query: 查询文本
            k: 返回数量
            use_reranker: 是否使用重排序

        Returns:
            Document列表
        """
        # 提取过滤条件
        filters = self.metadata_extractor.extract_filters(query)

        if use_reranker and self.config.reranker.enabled:
            return self.ensemble_search_with_rerank(query, k=k, filters=filters)
        else:
            return self.ensemble_search(query, k=k, filters=filters)

    async def afiltered_search(self, query: str, k: Optional[int] = None,
                               use_reranker: bool = True) -> List[Document]:
        """智能过滤检索（异步）"""
        # 检查缓存
        if self.config.cache.enabled:
            cached = await self.cache_manager.aget_cached_results(query)
            if cached is not None:
                return cached

        # 异步提取过滤条件
        filters = await self.metadata_extractor.aextract_filters(query)

        if use_reranker and self.config.reranker.enabled:
            result = await self.aensemble_search_with_rerank(query, k=k, filters=filters)
        else:
            result = await self.aensemble_search(query, k=k, filters=filters)

        if self.config.cache.enabled:
            await self.cache_manager.acache_results(query, result)

        return result

    # ==================== 工具方法 ====================

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        if self.cache_manager:
            return self.cache_manager.get_stats()
        return {}

    def clear_cache(self):
        """清空缓存"""
        if self.cache_manager:
            self.cache_manager.clear_all()

    def similarity_search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """
        相似度搜索并返回分数（用于调试）

        Returns:
            (Document, score)列表
        """
        if self.vector_store_factory is None:
            raise RuntimeError("向量存储未初始化")

        return self.vector_store_factory.similarity_search(query, k=k)

    def print_results(self, results: Union[List[Document], List[tuple]],
                      show_score: bool = True) -> None:
        """
        打印检索结果（保持与原代码一致的格式）

        Args:
            results: 结果列表
            show_score: 是否显示分数
        """
        for i, item in enumerate(results):
            if isinstance(item, tuple):
                doc, score = item
            else:
                doc = item
                score = doc.metadata.get('rerank_score', 0.0) if show_score else None

            logging.info(f"【结果 {i}】", end="")
            if score is not None:
                logging.info(f"相似度: {score:.3f}")
            logging.info(f"内容: {doc.page_content[:300]}...")
            logging.info(f"来源: {doc.metadata.get('source', 'Unknown')}")
            logging.info("-" * 50)
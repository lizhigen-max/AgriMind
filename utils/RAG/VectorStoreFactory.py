#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：VectorStoreFactory.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''
"""
向量存储工厂模块 - 支持FAISS、Chroma和Milvus
"""
import math

from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma, Milvus
import os
import logging

from .RAGConfig import VectorDBConfig, VectorModelConfig, VectorDBType, MilvusConfig


class VectorStoreFactory:
    """向量存储工厂类"""

    def __init__(self, db_config: VectorDBConfig, model_config: VectorModelConfig, model=None):
        self.db_config = db_config
        self.model_config = model_config
        self._embeddings = model
        self._vector_store = None

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        """获取或创建嵌入模型"""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_config.get_model_name(),
                model_kwargs={'device': self.model_config.device},
                encode_kwargs={'normalize_embeddings': self.model_config.normalize_embeddings}
            )
        return self._embeddings

    def create_vector_store(self, documents: Optional[List[Document]] = None) -> None:
        """
        创建新的向量存储

        Args:
            documents: 初始文档列表
        """
        logging.info('正在进行向量存储，请稍候...')
        embeddings = self._get_embeddings()

        if self.db_config.db_type == VectorDBType.FAISS:
            db_path = self.db_config.get_db_path()
            os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

            if documents:
                self._vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=embeddings
                )
            else:
                raise ValueError("FAISS需要提供初始文档")

        elif self.db_config.db_type == VectorDBType.CHROMA:
            db_path = self.db_config.get_db_path()
            os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

            self._vector_store = Chroma(
                collection_name=self.db_config.collection_name,
                embedding_function=embeddings,
                persist_directory=db_path
            )
            if documents:
                self._vector_store.add_documents(documents)

        elif self.db_config.db_type == VectorDBType.MILVUS:
            milvus_cfg = self.db_config.milvus_config

            if documents:
                ephco = math.ceil(len(documents) / 10.0)
                for i in range(ephco):
                    logging.info(f'共{len(documents)}个文本块，正在向量存储第{i + 1}批次（第{i*10+1}~第{((i+1)*10) if ((i+1)*10) < len(documents) else len(documents)}个文本块），共{ephco}批次')
                    if i == 0:
                        self._vector_store = Milvus.from_documents(
                            documents=documents[:10],
                            embedding=embeddings,
                            collection_name=milvus_cfg.collection_name,
                            connection_args=milvus_cfg.connection_args,
                            index_params=milvus_cfg.index_params,
                            search_params=milvus_cfg.search_params,
                            metadata_field="metadata",
                            drop_old=True  # True: 如果集合已存在则删除重建
                        )
                    else:
                        self._vector_store.add_documents(documents[i*10: (i + 1) * 10])
            else:
                # 创建空集合
                self._vector_store = Milvus(
                    embedding_function=embeddings,
                    collection_name=milvus_cfg.collection_name,
                    connection_args=milvus_cfg.connection_args,
                    index_params=milvus_cfg.index_params,
                    search_params=milvus_cfg.search_params,
                    metadata_field="metadata",
                )
        else:
            raise ValueError(f"不支持的向量数据库类型: {self.db_config.db_type}")

        logging.info(f"✓ 已创建 {self.db_config.db_type.value} 向量存储")

    def load_vector_store(self) -> None:
        """加载已存在的向量存储"""
        embeddings = self._get_embeddings()

        if self.db_config.db_type == VectorDBType.FAISS:
            db_path = self.db_config.get_db_path()
            faiss_index_path = os.path.join(db_path, f"{self.db_config.index_name}.faiss")
            pkl_path = os.path.join(db_path, f"{self.db_config.index_name}.pkl")

            if not os.path.exists(faiss_index_path) or not os.path.exists(pkl_path):
                raise FileNotFoundError(f"FAISS索引文件不存在: {db_path}")

            self._vector_store = FAISS.load_local(
                folder_path=db_path,
                index_name=self.db_config.index_name,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )

        elif self.db_config.db_type == VectorDBType.CHROMA:
            db_path = self.db_config.get_db_path()
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Chroma数据库不存在: {db_path}")

            self._vector_store = Chroma(
                collection_name=self.db_config.collection_name,
                embedding_function=embeddings,
                persist_directory=db_path
            )

        elif self.db_config.db_type == VectorDBType.MILVUS:
            milvus_cfg = self.db_config.milvus_config

            # Milvus 不需要本地文件检查，直接连接服务器
            self._vector_store = Milvus(
                embedding_function=embeddings,
                collection_name=milvus_cfg.collection_name,
                connection_args=milvus_cfg.connection_args,
                index_params=milvus_cfg.index_params,
                search_params=milvus_cfg.search_params
            )

            # 检查集合是否存在
            try:
                from pymilvus import utility
                if not utility.has_collection(milvus_cfg.collection_name, using=milvus_cfg.connection_args):
                    logging.warning(f"Milvus集合 '{milvus_cfg.collection_name}' 不存在，请先创建向量存储")
            except Exception as e:
                logging.warning(f"无法验证Milvus集合状态: {e}")

        else:
            raise ValueError(f"不支持的向量数据库类型: {self.db_config.db_type}")

        logging.info(f"✓ 已加载 {self.db_config.db_type.value} 向量存储")

    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到向量存储"""
        if self._vector_store is None:
            raise RuntimeError("向量存储未初始化")
        self._vector_store.add_documents(documents)
        logging.info(f"✓ 已添加 {len(documents)} 个文档")

    def save(self) -> None:
        """保存向量存储"""
        if self._vector_store is None:
            raise RuntimeError("向量存储未初始化")

        if self.db_config.db_type == VectorDBType.FAISS:
            self._vector_store.save_local(
                self.db_config.get_db_path(),
                index_name=self.db_config.index_name
            )
        # Chroma 和 Milvus 会自动保存

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """获取检索器"""
        if self._vector_store is None:
            raise RuntimeError("向量存储未初始化")

        kwargs = search_kwargs or {"k": 8}

        # Milvus 需要特殊处理搜索参数
        if self.db_config.db_type == VectorDBType.MILVUS:
            milvus_cfg = self.db_config.milvus_config
            kwargs["search_params"] = milvus_cfg.search_params

        return self._vector_store.as_retriever(search_kwargs=kwargs)

    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[tuple]:
        """相似度搜索"""
        if self._vector_store is None:
            raise RuntimeError("向量存储未初始化")

        if self.db_config.db_type == VectorDBType.FAISS:
            return self._vector_store.similarity_search_with_relevance_scores(query, k=k, **kwargs)

        elif self.db_config.db_type == VectorDBType.MILVUS:
            # Milvus 支持带分数的搜索
            milvus_cfg = self.db_config.milvus_config
            search_kwargs = kwargs.copy()
            search_kwargs.setdefault("param", milvus_cfg.search_params)
            return self._vector_store.similarity_search_with_score(query, k=k, **search_kwargs)

        else:
            # Chroma需要单独处理分数
            docs = self._vector_store.similarity_search(query, k=k, **kwargs)
            return [(doc, 0.0) for doc in docs]  # Chroma不直接返回分数

    def get_all_documents(self) -> List[Document]:
        """获取所有文档（用于BM25）"""
        if self._vector_store is None:
            raise RuntimeError("向量存储未初始化")

        # FAISS
        if hasattr(self._vector_store, 'docstore') and hasattr(self._vector_store.docstore, '_dict'):
            return list(self._vector_store.docstore._dict.values())

        # Chroma - 需要特殊处理
        if self.db_config.db_type == VectorDBType.CHROMA:
            collection = self._vector_store._collection
            result = collection.get(include=["documents", "metadatas"])
            documents = []
            for doc, metadata in zip(result["documents"], result["metadatas"]):
                documents.append(Document(page_content=doc, metadata=metadata or {}))
            return documents

        # Milvus - 获取所有数据
        if self.db_config.db_type == VectorDBType.MILVUS:
            try:
                # 使用 Milvus 客户端获取所有数据
                from pymilvus import Collection, connections
                # 直接访问底层 Collection 对象
                collection = self._vector_store.col

                if collection is None:
                    raise ValueError("Collection 未初始化")

                # 加载集合
                collection.load()

                # 查询所有数据
                results = collection.query(
                    expr="pk >= 0",  # 永真表达式
                    output_fields=["text", "metadata", "pk"]
                )

                documents = []
                for result in results:
                    doc_content = result.get("text", "")
                    doc_metadata = result.get("metadata", {})
                    documents.append(Document(page_content=doc_content, metadata=doc_metadata))

                return documents
            except Exception as e:
                logging.error(f"从 Milvus 获取文档失败: {e}")
                raise NotImplementedError(f"无法从 Milvus 获取所有文档: {e}")

        raise NotImplementedError("当前向量存储类型不支持获取所有文档")

    def delete_collection(self) -> bool:
        """删除集合/索引（Milvus专用）"""
        if self.db_config.db_type == VectorDBType.MILVUS:
            try:
                from pymilvus import utility
                milvus_cfg = self.db_config.milvus_config
                if utility.has_collection(milvus_cfg.collection_name, using=milvus_cfg.connection_args):
                    utility.drop_collection(milvus_cfg.collection_name, using=milvus_cfg.connection_args)
                    logging.info(f"✓ 已删除 Milvus 集合: {milvus_cfg.collection_name}")
                    return True
                return False
            except Exception as e:
                logging.error(f"删除 Milvus 集合失败: {e}")
                return False
        else:
            logging.warning("delete_collection 仅支持 Milvus 数据库")
            return False

    def collection_exists(self) -> bool:
        """检查集合是否存在（Milvus专用）"""
        if self.db_config.db_type == VectorDBType.MILVUS:
            try:
                from pymilvus import utility
                milvus_cfg = self.db_config.milvus_config
                return utility.has_collection(milvus_cfg.collection_name, using=milvus_cfg.connection_args)
            except Exception as e:
                logging.error(f"检查 Milvus 集合状态失败: {e}")
                return False
        return False

    @property
    def vector_store(self):
        return self._vector_store
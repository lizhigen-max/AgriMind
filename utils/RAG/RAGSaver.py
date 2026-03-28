#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：RAGSaver.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''
"""
文档索引模块 - 用于构建和保存向量数据库
"""
from typing import List, Optional, Union
from pathlib import Path
from langchain_core.documents import Document
import logging

from .RAGConfig import RAGConfig, VectorModelConfig, VectorDBConfig
from .VectorStoreFactory import VectorStoreFactory
from .DocumentChunk.DocumentProcessor import DocumentProcessor


class RAGSaver:
    """
    RAG文档索引器

    用于:
    - 从文件夹批量处理文档
    - 构建向量索引
    - 保存到FAISS或Chroma
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.vector_store_factory: Optional[VectorStoreFactory] = None

    def process_documents(self, document_folder: Union[str, List[str]],
                          chunk_size: int = 500,
                          chunk_overlap: int = 50) -> List[Document]:
        """
        处理文档文件夹

        Args:
            document_folder: 文件夹路径或路径列表
            chunk_size: 分块大小
            chunk_overlap: 重叠大小

        Returns:
            Document列表
        """
        try:
            processor = DocumentProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                splitter_type="recursive"
            )

            chunks = processor.batch_process(document_folder, show_progress=True)
            logging.info(f"✓ 已处理文档，共 {len(chunks)} 个块")
            return chunks

        except ImportError:
            logging.error("DocumentProcessor模块未找到，请确保已实现该模块")
            raise

    def create_index(self,
                     documents: Union[str, List[str], List[Document]],
                     db_config: Optional[VectorDBConfig] = None,
                     model_config: Optional[VectorModelConfig] = None,
                     chunk_size: int = 500,
                     chunk_overlap: int = 50) -> None:
        """
        创建向量索引

        Args:
            documents: 文件夹路径、路径列表或Document列表
            db_config: 向量数据库配置
            model_config: 向量模型配置
            chunk_size: 分块大小
            chunk_overlap: 重叠大小
        """
        db_config = db_config or self.config.vector_db
        model_config = model_config or self.config.vector_model

        # 处理输入
        if isinstance(documents, (str, list)) and documents and isinstance(
                documents[0] if isinstance(documents, list) else documents, str):
            # 是路径
            chunks = self.process_documents(documents, chunk_size, chunk_overlap)
        else:
            # 已经是Document列表
            chunks = documents

        # 创建向量存储
        self.vector_store_factory = VectorStoreFactory(db_config, model_config)
        self.vector_store_factory.create_vector_store(chunks)
        self.vector_store_factory.save()

        logging.info(f"✓ 索引已保存到: {db_config.get_db_path()}")

    def update_index(self,
                     new_documents: Union[str, List[str], List[Document]],
                     db_config: Optional[VectorDBConfig] = None,
                     model_config: Optional[VectorModelConfig] = None,
                     chunk_size: int = 500,
                     chunk_overlap: int = 50) -> None:
        """
        更新现有索引（添加新文档）

        Args:
            new_documents: 新文档
            db_config: 数据库配置
            model_config: 模型配置
            chunk_size: 分块大小
            chunk_overlap: 重叠大小
        """
        db_config = db_config or self.config.vector_db
        model_config = model_config or self.config.vector_model

        # 加载现有索引
        self.vector_store_factory = VectorStoreFactory(db_config, model_config)
        self.vector_store_factory.load_vector_store()

        # 处理新文档
        if isinstance(new_documents, (str, list)) and new_documents and isinstance(
                new_documents[0] if isinstance(new_documents, list) else new_documents, str):
            chunks = self.process_documents(new_documents, chunk_size, chunk_overlap)
        else:
            chunks = new_documents

        # 添加并保存
        self.vector_store_factory.add_documents(chunks)
        self.vector_store_factory.save()

        logging.info(f"✓ 已更新索引，新增 {len(chunks)} 个块")

    @staticmethod
    def test_search(query: str,
                    db_config: Optional[VectorDBConfig] = None,
                    model_config: Optional[VectorModelConfig] = None) -> None:
        """
        测试检索（静态方法，无需实例化）

        Args:
            query: 测试查询
            db_config: 数据库配置
            model_config: 模型配置
        """
        config = RAGConfig()
        db_config = db_config or config.vector_db
        model_config = model_config or config.vector_model

        factory = VectorStoreFactory(db_config, model_config)
        factory.load_vector_store()

        logging.info(f"[OK] 文档已索引到本地 {db_config.db_type.value}")
        logging.info(f"\n测试检索:")
        logging.info(f"  查询: {query}")

        results = factory.similarity_search(query, k=5)

        for i, (doc, score) in enumerate(results):
            logging.info(doc.metadata)
            logging.info(f"【结果 {i}】相似度: {score:.3f}")
            logging.info(f"内容: {doc.page_content[:300]}...")
            logging.info(f"来源: {doc.metadata.get('source', 'Unknown')}")
            logging.info("-" * 50)


# 便捷函数
def create_index_from_folder(folder_path: str,
                             db_type: str = "faiss",
                             model_name: str = "../models/BAAI/bge-large-zh-v1.5",
                             output_path: str = "./vector_db") -> None:
    """
    从文件夹快速创建索引

    Args:
        folder_path: 文档文件夹路径
        db_type: 数据库类型 (faiss/chroma/milvus)
        model_name: 模型名称
        output_path: 输出路径
    """
    from .RAGConfig import VectorDBType, VectorModelType

    config = RAGConfig(
        vector_db=VectorDBConfig(
            db_type=VectorDBType(db_type),
            persist_directory=output_path
        ),
        vector_model=VectorModelConfig(
            model_type=VectorModelType.BGE_LARGE_ZH if "large" in model_name else VectorModelType.BGE_M3,
            model_path=model_name if "/" not in model_name else None
        )
    )

    saver = RAGSaver(config)
    saver.create_index(folder_path)


if __name__ == '__main__':
    # 示例用法
    create_index_from_folder('../Document', db_type='milvus')
    # RAGSaver.test_search('葡萄冬季怎么修剪')
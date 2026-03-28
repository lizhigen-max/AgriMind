#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：RAGConfig.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''
"""
RAG系统配置管理模块
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import os

from dotenv import load_dotenv
load_dotenv()

class VectorModelType(Enum):
    """支持的向量模型类型"""
    BGE_LARGE_ZH = os.getenv('BGE_LARGE_ZH_MODEL_PATH')
    BGE_M3 = os.getenv('BGE_M3__MODEL_PATH')


class VectorDBType(Enum):
    """支持的向量数据库类型"""
    FAISS = "faiss"
    CHROMA = "chroma"
    MILVUS = "milvus"


@dataclass
class VectorModelConfig:
    """向量模型配置"""
    model_type: VectorModelType = VectorModelType.BGE_LARGE_ZH
    model_path: Optional[str] = None  # 本地路径，如果提供则优先使用
    device: str = os.getenv('DEVICE')
    normalize_embeddings: bool = True if os.getenv('NOR_EBDINGS') == 'true' else False

    def get_model_name(self) -> str:
        """获取实际使用的模型名称"""
        if self.model_path:
            return self.model_path
        try:
            return self.model_type.value
        except:
            return self.model_type


@dataclass
class MilvusConfig:
    """Milvus 专用配置"""
    host: str = os.getenv('MILVUS_HOST', 'localhost')
    port: int = int(os.getenv('MILVUS_PORT', '19530'))
    collection_name: str = os.getenv('MILVUS_COLLECTION_NAME', 'rag_collection')
    database_name: str = os.getenv('MILVUS_DATABASE_NAME', 'default')
    index_params: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    })
    search_params: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    })

    @property
    def connection_args(self) -> Dict[str, Any]:
        """获取 Milvus 连接参数"""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database_name
        }

@dataclass
class VectorDBConfig:
    """向量数据库配置"""
    _type_str = os.getenv('VECTORDBTYPE')
    if _type_str == 'faiss':
        db_type: VectorDBType = VectorDBType.FAISS
    elif _type_str == 'chroma':
        db_type: VectorDBType = VectorDBType.CHROMA
    elif _type_str == 'milvus':
        db_type: VectorDBType = VectorDBType.MILVUS
    else:
        raise Exception("暂不支持的向量数据库类型")
    if VectorModelConfig.model_type == VectorModelType.BGE_LARGE_ZH:
        persist_directory: str = os.getenv('VEC_DB_BGE_LARGE_ZH_PATH')
    else:
        persist_directory: str = os.getenv('VEC_DB_BGE_M3_PATH')
    index_name: str = os.getenv('VEC_INDEX')
    collection_name: str = os.getenv('COLL_NAME')  # Chroma使用

    # Milvus 专用配置
    milvus_config: MilvusConfig = field(default_factory=MilvusConfig)

    def get_db_path(self) -> str:
        """获取数据库路径"""
        if self.db_type == VectorDBType.FAISS:
            return os.path.join(self.persist_directory, self.db_type.value)
        elif self.db_type == VectorDBType.MILVUS:
            return f"milvus://{self.milvus_config.host}:{self.milvus_config.port}/{self.milvus_config.database_name}"
        return os.path.join(self.persist_directory, self.collection_name)


@dataclass
class CacheConfig:
    """缓存配置"""
    enabled: bool = True if os.getenv('ENABLED_RAG_CACHE') == 'true' else False
    db_path: str = os.getenv('RAG_CACHE_DB_PATH')
    expire_days: int = int(os.getenv('RAG_CACHE_EXPIRE_DAYS'))
    max_size_mb: int = int(os.getenv('RAG_CACHE_MAX_SIZE'))


@dataclass
class RerankerConfig:
    """重排序模型配置"""
    enabled: bool = True if os.getenv('RERANKER_ENABLED') == 'true' else False
    model_path: str = os.getenv('RERANKER_MODEL_PATH')
    threshold: float = float(os.getenv('RERANKER_THRESHOLD'))
    max_length: int = int(os.getenv('RERANKER_MAX_LEN'))


@dataclass
class RetrievalConfig:
    """检索配置"""
    vector_k: int = int(os.getenv('VECTOR_K'))
    bm25_k: int = int(os.getenv('BM25_K'))
    ensemble_weights: List[float] = field(default_factory=lambda: [float(os.getenv('ENSEMBLE_WGT_BM25')),
                                                                   float(os.getenv('ENSEMBLE_WGT_VECTOR'))])
    use_async: bool = True if os.getenv('USE_ASYNC') == 'true' else False


@dataclass
class RAGConfig:
    """RAG系统总配置"""
    vector_model: VectorModelConfig = field(default_factory=VectorModelConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    # DeepSeek API配置
    deepseek_api_key: Optional[str] = None

    def __post_init__(self):
        """初始化后处理"""
        if not self.deepseek_api_key:
            self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
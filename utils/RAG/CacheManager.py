#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：CacheManager.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''
"""
缓存管理模块 - 支持向量检索结果缓存
"""

import sqlite3
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from threading import Lock
import atexit
import logging
from functools import wraps
import asyncio

from .RAGConfig import CacheConfig


class CacheManager:
    """缓存管理器 - 线程安全"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.db_path = config.db_path
        self.expire_days = config.expire_days
        self.max_size_mb = config.max_size_mb
        self._lock = Lock()
        self._async_lock = asyncio.Lock()
        self.cleanup_interval = timedelta(hours=6)
        self.last_cleanup = datetime.now()

        if config.enabled:
            self._init_db()
            atexit.register(self.cleanup)

    def _init_db(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_results (
                    query_hash TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    documents_blob BLOB NOT NULL,
                    metadata_json TEXT NOT NULL,
                    vector_store_type TEXT NOT NULL,
                    search_k INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_stats (
                    stat_key TEXT PRIMARY KEY,
                    stat_value TEXT NOT NULL
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON vector_results(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON vector_results(last_accessed)")

            conn.execute("""
                INSERT OR IGNORE INTO cache_stats (stat_key, stat_value) 
                VALUES ('total_hits', '0'), ('total_misses', '0')
            """)
            conn.commit()

    def _get_query_hash(self, query: str) -> str:
        """生成查询哈希"""
        return hashlib.sha256(query.encode('utf-8')).hexdigest()

    def _enforce_cache_size(self):
        """强制执行缓存大小限制"""
        with sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT SUM(LENGTH(documents_blob) + LENGTH(metadata_json)) 
                FROM vector_results
            """)
            total_size = (cursor.fetchone()[0] or 0) / (1024 * 1024)

            if total_size > self.max_size_mb:
                logging.info(f"缓存大小{total_size:.2f}MB超过限制，清理中...")
                cursor.execute("""
                    DELETE FROM vector_results 
                    WHERE query_hash IN (
                        SELECT query_hash FROM vector_results 
                        ORDER BY last_accessed ASC 
                        LIMIT (SELECT COUNT(*) / 2 FROM vector_results)
                    )
                """)
                conn.commit()

    def cache_results(self, query: str, documents: List[Document],
                      k: int = 8, vector_store_type: str = "FAISS") -> None:
        """
        缓存检索结果（同步）

        Args:
            query: 查询文本
            documents: 文档列表
            k: 检索数量
            vector_store_type: 向量存储类型
        """
        if not self.config.enabled:
            return

        with self._lock:
            self._cache_results_internal(query, documents, k, vector_store_type)

    async def acache_results(self, query: str, documents: List[Document],
                             k: int = 8, vector_store_type: str = "FAISS") -> None:
        """异步缓存检索结果"""
        if not self.config.enabled:
            return

        async with self._async_lock:
            # 使用线程池执行同步操作
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._cache_results_internal,
                query, documents, k, vector_store_type
            )

    def _cache_results_internal(self, query: str, documents: List[Document],
                                k: int, vector_store_type: str) -> None:
        """内部缓存实现"""
        query_hash = self._get_query_hash(query)

        documents_data = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]
        documents_blob = pickle.dumps(documents_data)
        metadata = {
            "query_length": len(query),
            "query_hash": query_hash,
            "k": k,
            "cached_at": datetime.now().isoformat()
        }

        self._enforce_cache_size()

        with sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT query_hash FROM vector_results WHERE query_hash = ?", (query_hash,))
            exists = cursor.fetchone() is not None

            if exists:
                cursor.execute("""
                    UPDATE vector_results 
                    SET documents_blob = ?, metadata_json = ?, 
                        last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                    WHERE query_hash = ?
                """, (documents_blob, json.dumps(metadata), query_hash))
            else:
                cursor.execute("""
                    INSERT INTO vector_results 
                    (query_hash, query_text, documents_blob, metadata_json, 
                     vector_store_type, search_k, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (query_hash, query, documents_blob, json.dumps(metadata),
                      vector_store_type, k))

            stat_key = 'total_hits' if exists else 'total_misses'
            cursor.execute(f"""
                UPDATE cache_stats 
                SET stat_value = CAST(stat_value AS INTEGER) + 1 
                WHERE stat_key = '{stat_key}'
            """)
            conn.commit()

            logging.info(f"缓存{'更新' if exists else '新增'}: {query[:50]}...")

    def get_cached_results(self, query: str) -> Optional[List[Document]]:
        """
        获取缓存结果（同步）

        Returns:
            Document列表或None
        """
        if not self.config.enabled:
            return None

        with self._lock:
            return self._get_cached_results_internal(query)

    async def aget_cached_results(self, query: str) -> Optional[List[Document]]:
        """异步获取缓存结果"""
        if not self.config.enabled:
            return None

        async with self._async_lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._get_cached_results_internal, query)

    def _get_cached_results_internal(self, query: str) -> Optional[List[Document]]:
        """内部获取缓存实现"""
        self._auto_cleanup()
        query_hash = self._get_query_hash(query)
        expire_time = datetime.now() - timedelta(days=self.expire_days)

        with sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT documents_blob FROM vector_results 
                WHERE query_hash = ? AND created_at > ?
            """, (query_hash, expire_time))

            row = cursor.fetchone()
            if row:
                try:
                    cursor.execute("""
                        UPDATE vector_results 
                        SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                        WHERE query_hash = ?
                    """, (query_hash,))
                    cursor.execute("""
                        UPDATE cache_stats 
                        SET stat_value = CAST(stat_value AS INTEGER) + 1 
                        WHERE stat_key = 'total_hits'
                    """)
                    conn.commit()

                    documents_data = pickle.loads(row['documents_blob'])
                    documents = [
                        Document(page_content=d["page_content"], metadata=d["metadata"])
                        for d in documents_data
                    ]
                    logging.info(f"缓存命中: {query[:50]}...")
                    return documents

                except Exception as e:
                    logging.error(f"反序列化失败: {e}")
                    cursor.execute("DELETE FROM vector_results WHERE query_hash = ?", (query_hash,))
                    conn.commit()
                    return None
            else:
                cursor.execute("""
                    UPDATE cache_stats 
                    SET stat_value = CAST(stat_value AS INTEGER) + 1 
                    WHERE stat_key = 'total_misses'
                """)
                conn.commit()
                logging.info(f"缓存未命中: {query[:50]}...")
                return None

    def _auto_cleanup(self):
        """自动清理"""
        if datetime.now() - self.last_cleanup > self.cleanup_interval:
            self.cleanup()

    def cleanup(self, force: bool = False):
        """清理过期缓存"""
        if not force and datetime.now() - self.last_cleanup < self.cleanup_interval:
            return

        with self._lock:
            with sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False) as conn:
                cursor = conn.cursor()
                expire_time = datetime.now() - timedelta(days=self.expire_days)
                cursor.execute("DELETE FROM vector_results WHERE created_at < ?", (expire_time,))
                deleted = cursor.rowcount
                cursor.execute("VACUUM")
                conn.commit()
                self.last_cleanup = datetime.now()

                if deleted > 0:
                    logging.info(f"清理了 {deleted} 条过期缓存")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.config.enabled:
            return {"enabled": False}

        with sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT stat_key, stat_value FROM cache_stats")
            stats = {row['stat_key']: int(row['stat_value']) for row in cursor.fetchall()}

            cursor.execute("SELECT COUNT(*) as count FROM vector_results")
            stats['cache_size'] = cursor.fetchone()['count']

            cursor.execute("""
                SELECT SUM(LENGTH(documents_blob) + LENGTH(metadata_json)) as total_size 
                FROM vector_results
            """)
            size = cursor.fetchone()['total_size'] or 0
            stats['cache_mb'] = size / (1024 * 1024)

            total = stats.get('total_hits', 0) + stats.get('total_misses', 0)
            stats['hit_rate'] = (stats.get('total_hits', 0) / total * 100) if total > 0 else 0.0

            return stats

    def clear_all(self):
        """清空所有缓存"""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM vector_results")
                cursor.execute("DELETE FROM cache_stats")
                cursor.execute("""
                    INSERT INTO cache_stats (stat_key, stat_value) 
                    VALUES ('total_hits', '0'), ('total_misses', '0')
                """)
                cursor.execute("VACUUM")
                conn.commit()
                logging.info("已清空所有缓存")


def cached_retrieval(cache_attr_name: str = 'cache_manager'):
    """
    缓存装饰器工厂

    使用方式：
        @cached_retrieval('cache_manager')  # 传入属性名字符串
        def vector_search(self, query: str): ...

    Args:
        cache_attr_name: RAGProcessor实例中缓存管理器的属性名，默认'cache_manager'
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 假设第一个参数是self，第二个是query
            # 从self中获取缓存管理器实例
            cache_manager = getattr(self, cache_attr_name, None)

            # 如果缓存未启用或不存在，直接执行原函数
            if cache_manager is None or not cache_manager.config.enabled:
                return func(self, *args, **kwargs)
            query = args[0] if len(args) > 0 else kwargs.get('query')
            if not query:
                return func(self, *args, **kwargs)

            cached = cache_manager.get_cached_results(query)
            if cached is not None:
                return cached

            result = func(self, *args, **kwargs)
            if result:
                cache_manager.cache_results(query, result)
            return result

        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            cache_manager = getattr(self, cache_attr_name, None)

            # 如果缓存未启用或不存在，直接执行原函数
            if cache_manager is None or not cache_manager.config.enabled:
                return await func(self, *args, **kwargs)
            query = args[0] if len(args) > 0 else kwargs.get('query')
            if not query:
                return await func(self, *args, **kwargs)

            cached = await cache_manager.aget_cached_results(query)
            if cached is not None:
                return cached

            result = await func(self, *args, **kwargs)
            if result:
                await cache_manager.acache_results(query, result)
            return result

        # 根据函数是否是协程函数返回相应的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator
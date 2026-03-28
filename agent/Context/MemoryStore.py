#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：MemoryStore.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''

import os
import aiosqlite
import sqlite3
import logging
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
from contextlib import contextmanager
import json
from contextlib import asynccontextmanager

import aioodbc  # 改为异步ODBC
from pydantic import BaseModel, Field


class DialogueEntry(BaseModel):
    """单条对话记录"""
    id: Optional[int] = None
    user_id: str = Field(description="用户ID")
    query: str = Field(description="用户问题")
    response: str = Field(description="AI回答")
    intent: Optional[str] = Field(None, description="意图类型")
    confidence: Optional[float] = Field(None, description="置信度")
    created_at: Optional[datetime] = Field(None, description="创建时间")
    session_id: Optional[str] = Field(None, description="会话ID，用于分组连续对话")


class ConversationSummary(BaseModel):
    """对话摘要信息"""
    id: Optional[int] = None
    user_id: str = Field(description="用户ID")
    summary_text: str = Field(description="摘要内容")
    dialogue_count: int = Field(description="包含的对话轮次")
    start_time: datetime = Field(description="摘要覆盖的起始时间")
    end_time: datetime = Field(description="摘要覆盖的结束时间")
    key_points: Optional[str] = Field(None, description="关键信息点，JSON格式")
    created_at: Optional[datetime] = Field(None, description="摘要生成时间")
    updated_at: Optional[datetime] = Field(None, description="最后更新时间")


class EntityInfo(BaseModel):
    """实体与画像信息"""
    id: Optional[int] = None
    user_id: str = Field(description="用户ID")
    entity_type: str = Field(description="实体类型: person/location/project/preference/attribute等")
    entity_key: str = Field(description="实体标识/名称")
    entity_value: str = Field(description="实体值/描述")
    confidence: float = Field(1.0, description="置信度")
    source_dialogue_id: Optional[int] = Field(None, description="来源对话ID")
    created_at: Optional[datetime] = Field(None, description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="最后更新时间")


class MemoryConfig(BaseModel):
    """记忆存储配置"""
    db_type: str = Field("sqlite", description="数据库类型: sqlite 或 sqlserver")
    # SQLite 配置
    sqlite_path: str = Field("./data/memory.db", description="SQLite数据库路径")
    # SQL Server 配置
    sqlserver_host: str = Field("localhost", description="SQL Server主机")
    sqlserver_port: int = Field(1433, description="SQL Server端口")
    sqlserver_user: str = Field("", description="用户名")
    sqlserver_password: str = Field("", description="密码")
    sqlserver_database: str = Field("", description="数据库名")
    sqlserver_driver: str = Field("ODBC Driver 17 for SQL Server", description="ODBC驱动")
    # 业务配置
    max_dialogues_per_summary: int = Field(5, description="多少轮对话后生成摘要")
    recent_dialogues_limit: int = Field(5, description="每次取最近多少轮对话")
    auto_cleanup_days: int = Field(30, description="自动清理多少天前的对话")


class BaseMemoryStore(ABC):
    """记忆存储抽象基类"""

    @abstractmethod
    async def _init_tables(self) -> None:
        """初始化数据库表结构"""
        pass

    @abstractmethod
    async def save_dialogue(self, entry: DialogueEntry) -> int:
        """保存单条对话，返回记录ID"""
        pass

    @abstractmethod
    async def get_recent_dialogues(self, user_id: str, limit: int = 5) -> List[DialogueEntry]:
        """获取用户最近N轮对话"""
        pass

    @abstractmethod
    async def get_dialogue_count(self, user_id: str) -> int:
        """获取用户对话总数"""
        pass

    @abstractmethod
    async def save_summary(self, summary: ConversationSummary) -> int:
        """保存或更新摘要"""
        pass

    @abstractmethod
    async def get_latest_summary(self, user_id: str) -> Optional[ConversationSummary]:
        """获取用户最新的摘要"""
        pass

    @abstractmethod
    async def get_all_summaries(self, user_id: str) -> List[ConversationSummary]:
        """获取用户所有历史摘要"""
        pass

    @abstractmethod
    async def save_entity(self, entity: EntityInfo) -> int:
        """保存或更新实体信息（存在则更新，不存在则插入）"""
        pass

    @abstractmethod
    async def get_entities(
        self, 
        user_id: str, 
        entity_type: Optional[str] = None
    ) -> List[EntityInfo]:
        """获取用户实体信息，可按类型筛选"""
        pass

    @abstractmethod
    async def delete_old_dialogues(self, days: int) -> int:
        """删除指定天数前的对话记录，返回删除数量"""
        pass


class SQLiteMemoryStore(BaseMemoryStore):
    """SQLite 记忆存储实现"""

    def __init__(self, db_path: str = "./data/memory.db"):
        self.db_path = db_path
        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # 异步初始化
        self._initialized = False
        
    async def initialize(self):
        """异步初始化"""
        if not self._initialized:
            await self._init_tables()
            self._initialized = True
            logging.info(f"✓ SQLite记忆存储已初始化: {self.db_path}")

    @asynccontextmanager
    async def _get_connection(self):
        """获取数据库连接（异步上下文管理器）"""
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            yield conn

    async def _init_tables(self) -> None:
        """初始化表结构"""
        async with self._get_connection() as conn:
            # 对话表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS dialogues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    intent TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT
                )
            """)

            # 摘要表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    dialogue_count INTEGER NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP NOT NULL,
                    key_points TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 实体信息表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_key TEXT NOT NULL,
                    entity_value TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    source_dialogue_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, entity_type, entity_key)
                )
            """)

            # 创建索引
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dialogues_user_time 
                ON dialogues(user_id, created_at DESC)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_summaries_user_time 
                ON summaries(user_id, created_at DESC)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_user_type 
                ON entities(user_id, entity_type)
            """)
            await conn.commit()

    async def save_dialogue(self, entry: DialogueEntry) -> int:
        """保存对话记录"""
        async with self._get_connection() as conn:
            cursor = await conn.execute("""
                INSERT INTO dialogues (user_id, query, response, intent, confidence, created_at, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.user_id,
                entry.query,
                entry.response,
                entry.intent,
                entry.confidence,
                entry.created_at or datetime.now(),
                entry.session_id
            ))
            await conn.commit()
            return cursor.lastrowid

    async def get_recent_dialogues(self, user_id: str, limit: int = 5) -> List[DialogueEntry]:
        """获取最近N轮对话"""
        async with self._get_connection() as conn:
            cursor = await conn.execute("""
                SELECT * FROM dialogues 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (user_id, limit))

            rows = await cursor.fetchall()
            dialogues = [
                DialogueEntry(
                    id=row['id'],
                    user_id=row['user_id'],
                    query=row['query'],
                    response=row['response'],
                    intent=row['intent'],
                    confidence=row['confidence'],
                    created_at=row['created_at'],
                    session_id=row['session_id']
                )
                for row in rows
            ]
            return list(reversed(dialogues))

    async def get_dialogue_count(self, user_id: str) -> int:
        """获取对话总数"""
        async with self._get_connection() as conn:
            cursor = await conn.execute("""
                SELECT COUNT(*) as count FROM dialogues WHERE user_id = ?
            """, (user_id,))
            result = await cursor.fetchone()
            return result['count'] if result else 0

    async def save_summary(self, summary: ConversationSummary) -> int:
        """保存或更新摘要"""
        async with self._get_connection() as conn:
            # 查询现有记录
            cursor = await conn.execute("""
                SELECT id FROM summaries WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1
            """, (summary.user_id,))
            existing = await cursor.fetchone()

            if existing:
                # 更新现有摘要
                await conn.execute("""
                    UPDATE summaries 
                    SET summary_text = ?, dialogue_count = ?, end_time = ?, 
                        key_points = ?, updated_at = ?
                    WHERE id = ?
                """, (
                    summary.summary_text,
                    summary.dialogue_count,
                    summary.end_time,
                    summary.key_points,
                    datetime.now(),
                    existing['id']
                ))
                await conn.commit()
                return existing['id']
            else:
                # 插入新摘要
                cursor = await conn.execute("""
                    INSERT INTO summaries (user_id, summary_text, dialogue_count, start_time, end_time, key_points, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    summary.user_id,
                    summary.summary_text,
                    summary.dialogue_count,
                    summary.start_time,
                    summary.end_time,
                    summary.key_points,
                    summary.created_at or datetime.now(),
                    datetime.now()
                ))
                await conn.commit()
                return cursor.lastrowid

    async def get_latest_summary(self, user_id: str) -> Optional[ConversationSummary]:
        """获取最新摘要"""
        async with self._get_connection() as conn:
            cursor = await conn.execute("""
                SELECT * FROM summaries 
                WHERE user_id = ? 
                ORDER BY updated_at DESC 
                LIMIT 1
            """, (user_id,))

            row = await cursor.fetchone()
            if row:
                return ConversationSummary(
                    id=row['id'],
                    user_id=row['user_id'],
                    summary_text=row['summary_text'],
                    dialogue_count=row['dialogue_count'],
                    start_time=row['start_time'],
                    end_time=row['end_time'],
                    key_points=row['key_points'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            return None

    async def get_all_summaries(self, user_id: str) -> List[ConversationSummary]:
        """获取所有历史摘要"""
        async with self._get_connection() as conn:
            cursor = await conn.execute("""
                SELECT * FROM summaries 
                WHERE user_id = ? 
                ORDER BY created_at ASC
            """, (user_id,))

            rows = await cursor.fetchall()
            return [
                ConversationSummary(
                    id=row['id'],
                    user_id=row['user_id'],
                    summary_text=row['summary_text'],
                    dialogue_count=row['dialogue_count'],
                    start_time=row['start_time'],
                    end_time=row['end_time'],
                    key_points=row['key_points'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                for row in rows
            ]

    async def save_entity(self, entity: EntityInfo) -> int:
        """保存或更新实体信息（UPSERT逻辑）"""
        async with self._get_connection() as conn:
            now = datetime.now()
            
            # 尝试UPDATE，如果不存在则INSERT
            cursor = await conn.execute("""
                SELECT id FROM entities 
                WHERE user_id = ? AND entity_type = ? AND entity_key = ?
            """, (entity.user_id, entity.entity_type, entity.entity_key))
            
            existing = await cursor.fetchone()
            
            if existing:
                # 更新现有实体
                await conn.execute("""
                    UPDATE entities 
                    SET entity_value = ?, confidence = ?, updated_at = ?, source_dialogue_id = ?
                    WHERE id = ?
                """, (
                    entity.entity_value,
                    entity.confidence,
                    now,
                    entity.source_dialogue_id,
                    existing['id']
                ))
                await conn.commit()
                return existing['id']
            else:
                # 插入新实体
                cursor = await conn.execute("""
                    INSERT INTO entities 
                    (user_id, entity_type, entity_key, entity_value, confidence, source_dialogue_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity.user_id,
                    entity.entity_type,
                    entity.entity_key,
                    entity.entity_value,
                    entity.confidence,
                    entity.source_dialogue_id,
                    entity.created_at or now,
                    now
                ))
                await conn.commit()
                return cursor.lastrowid

    async def get_entities(
        self, 
        user_id: str, 
        entity_type: Optional[str] = None
    ) -> List[EntityInfo]:
        """获取用户实体信息"""
        async with self._get_connection() as conn:
            if entity_type:
                cursor = await conn.execute("""
                    SELECT * FROM entities 
                    WHERE user_id = ? AND entity_type = ?
                    ORDER BY updated_at DESC
                """, (user_id, entity_type))
            else:
                cursor = await conn.execute("""
                    SELECT * FROM entities 
                    WHERE user_id = ?
                    ORDER BY updated_at DESC
                """, (user_id,))

            rows = await cursor.fetchall()
            return [
                EntityInfo(
                    id=row['id'],
                    user_id=row['user_id'],
                    entity_type=row['entity_type'],
                    entity_key=row['entity_key'],
                    entity_value=row['entity_value'],
                    confidence=row['confidence'],
                    source_dialogue_id=row['source_dialogue_id'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                for row in rows
            ]

    async def delete_old_dialogues(self, days: int) -> int:
        """删除指定天数前的对话记录"""
        async with self._get_connection() as conn:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # 先统计要删除的数量
            cursor = await conn.execute("""
                SELECT COUNT(*) as count FROM dialogues 
                WHERE created_at < ?
            """, (cutoff_date,))
            result = await cursor.fetchone()
            count = result['count'] if result else 0
            
            # 执行删除（保留摘要，只删对话）
            await conn.execute("""
                DELETE FROM dialogues 
                WHERE created_at < ?
            """, (cutoff_date,))
            
            await conn.commit()
            
            if count > 0:
                logging.info(f"🗑️ 自动清理: 删除了 {count} 条 {days} 天前的对话记录")
            return count


class SQLServerMemoryStore(BaseMemoryStore):
    """SQL Server 异步记忆存储实现 (使用aioodbc)"""

    def __init__(self, host: str, port: int, user: str, password: str, database: str, driver: str = "ODBC Driver 17 for SQL Server"):
        self.connection_string = (
            f"DRIVER={{{driver}}};"
            f"SERVER={host},{port};"
            f"DATABASE={database};"
            f"UID={user};"
            f"PWD={password};"
            f"charset=UTF-8;"
        )
        self._initialized = False

    async def initialize(self):
        """异步初始化"""
        if not self._initialized:
            await self._init_tables()
            self._initialized = True
            logging.info(f"✓ SQL Server记忆存储已初始化")

    @asynccontextmanager
    async def _get_connection(self):
        """获取数据库连接（异步上下文管理器）"""
        conn = await aioodbc.connect(dsn=self.connection_string)
        try:
            yield conn
        finally:
            await conn.close()

    async def _init_tables(self) -> None:
        """初始化表结构"""
        async with self._get_connection() as conn:
            cursor = await conn.cursor()

            # 对话记录表
            await cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='dialogues' AND xtype='U')
                CREATE TABLE dialogues (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    user_id NVARCHAR(255) NOT NULL,
                    query NVARCHAR(MAX) NOT NULL,
                    response NVARCHAR(MAX) NOT NULL,
                    intent NVARCHAR(50),
                    confidence FLOAT,
                    created_at DATETIME DEFAULT GETDATE(),
                    session_id NVARCHAR(255)
                )
            """)

            # 摘要表
            await cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='summaries' AND xtype='U')
                CREATE TABLE summaries (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    user_id NVARCHAR(255) NOT NULL,
                    summary_text NVARCHAR(MAX) NOT NULL,
                    dialogue_count INT NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME NOT NULL,
                    key_points NVARCHAR(MAX),
                    created_at DATETIME DEFAULT GETDATE(),
                    updated_at DATETIME DEFAULT GETDATE()
                )
            """)

            # 实体信息表
            await cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='entities' AND xtype='U')
                CREATE TABLE entities (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    user_id NVARCHAR(255) NOT NULL,
                    entity_type NVARCHAR(50) NOT NULL,
                    entity_key NVARCHAR(255) NOT NULL,
                    entity_value NVARCHAR(MAX) NOT NULL,
                    confidence FLOAT DEFAULT 1.0,
                    source_dialogue_id INT,
                    created_at DATETIME DEFAULT GETDATE(),
                    updated_at DATETIME DEFAULT GETDATE(),
                    CONSTRAINT UQ_entities_user_type_key UNIQUE(user_id, entity_type, entity_key)
                )
            """)

            # 创建索引
            await cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_dialogues_user_time')
                CREATE INDEX idx_dialogues_user_time ON dialogues(user_id, created_at DESC)
            """)
            await cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_summaries_user_time')
                CREATE INDEX idx_summaries_user_time ON summaries(user_id, created_at DESC)
            """)
            await cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_entities_user_type')
                CREATE INDEX idx_entities_user_type ON entities(user_id, entity_type)
            """)
            
            await conn.commit()

    async def save_dialogue(self, entry: DialogueEntry) -> int:
        """保存对话记录"""
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute("""
                INSERT INTO dialogues (user_id, query, response, intent, confidence, created_at, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                SELECT SCOPE_IDENTITY()
            """, (
                entry.user_id,
                entry.query,
                entry.response,
                entry.intent,
                entry.confidence,
                entry.created_at or datetime.now(),
                entry.session_id
            ))
            result = await cursor.fetchone()
            await conn.commit()
            return int(result[0]) if result else 0

    async def get_recent_dialogues(self, user_id: str, limit: int = 5) -> List[DialogueEntry]:
        """获取最近N轮对话"""
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute("""
                SELECT * FROM dialogues 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                OFFSET 0 ROWS FETCH NEXT ? ROWS ONLY
            """, (user_id, limit))

            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            dialogues = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                dialogues.append(DialogueEntry(
                    id=row_dict.get('id'),
                    user_id=row_dict.get('user_id'),
                    query=row_dict.get('query'),
                    response=row_dict.get('response'),
                    intent=row_dict.get('intent'),
                    confidence=row_dict.get('confidence'),
                    created_at=row_dict.get('created_at'),
                    session_id=row_dict.get('session_id')
                ))
            return list(reversed(dialogues))

    async def get_dialogue_count(self, user_id: str) -> int:
        """获取对话总数"""
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute("""
                SELECT COUNT(*) FROM dialogues WHERE user_id = ?
            """, (user_id,))
            result = await cursor.fetchone()
            return result[0] if result else 0

    async def save_summary(self, summary: ConversationSummary) -> int:
        """保存或更新摘要"""
        async with self._get_connection() as conn:
            cursor = await conn.cursor()

            # 检查是否已存在
            await cursor.execute("""
                SELECT TOP 1 id FROM summaries WHERE user_id = ? ORDER BY updated_at DESC
            """, (summary.user_id,))
            existing = await cursor.fetchone()

            if existing:
                await cursor.execute("""
                    UPDATE summaries 
                    SET summary_text = ?, dialogue_count = ?, end_time = ?, 
                        key_points = ?, updated_at = GETDATE()
                    WHERE id = ?
                """, (
                    summary.summary_text,
                    summary.dialogue_count,
                    summary.end_time,
                    summary.key_points,
                    existing[0]
                ))
                await conn.commit()
                return existing[0]
            else:
                await cursor.execute("""
                    INSERT INTO summaries (user_id, summary_text, dialogue_count, start_time, end_time, key_points, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, GETDATE(), GETDATE());
                    SELECT SCOPE_IDENTITY()
                """, (
                    summary.user_id,
                    summary.summary_text,
                    summary.dialogue_count,
                    summary.start_time,
                    summary.end_time,
                    summary.key_points
                ))
                result = await cursor.fetchone()
                await conn.commit()
                return int(result[0]) if result else 0

    async def get_latest_summary(self, user_id: str) -> Optional[ConversationSummary]:
        """获取最新摘要"""
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute("""
                SELECT TOP 1 * FROM summaries 
                WHERE user_id = ? 
                ORDER BY updated_at DESC
            """, (user_id,))

            row = await cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                row_dict = dict(zip(columns, row))
                return ConversationSummary(
                    id=row_dict.get('id'),
                    user_id=row_dict.get('user_id'),
                    summary_text=row_dict.get('summary_text'),
                    dialogue_count=row_dict.get('dialogue_count'),
                    start_time=row_dict.get('start_time'),
                    end_time=row_dict.get('end_time'),
                    key_points=row_dict.get('key_points'),
                    created_at=row_dict.get('created_at'),
                    updated_at=row_dict.get('updated_at')
                )
            return None

    async def get_all_summaries(self, user_id: str) -> List[ConversationSummary]:
        """获取所有历史摘要"""
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute("""
                SELECT * FROM summaries 
                WHERE user_id = ? 
                ORDER BY created_at ASC
            """, (user_id,))

            columns = [desc[0] for desc in cursor.description]
            summaries = []
            for row in await cursor.fetchall():
                row_dict = dict(zip(columns, row))
                summaries.append(ConversationSummary(
                    id=row_dict.get('id'),
                    user_id=row_dict.get('user_id'),
                    summary_text=row_dict.get('summary_text'),
                    dialogue_count=row_dict.get('dialogue_count'),
                    start_time=row_dict.get('start_time'),
                    end_time=row_dict.get('end_time'),
                    key_points=row_dict.get('key_points'),
                    created_at=row_dict.get('created_at'),
                    updated_at=row_dict.get('updated_at')
                ))
            return summaries

    async def save_entity(self, entity: EntityInfo) -> int:
        """保存或更新实体信息（MERGE逻辑）"""
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            now = datetime.now()
            
            # SQL Server使用MERGE语句实现UPSERT
            await cursor.execute("""
                MERGE entities AS target
                USING (VALUES (?, ?, ?)) AS source (user_id, entity_type, entity_key)
                ON target.user_id = source.user_id 
                   AND target.entity_type = source.entity_type 
                   AND target.entity_key = source.entity_key
                WHEN MATCHED THEN
                    UPDATE SET 
                        entity_value = ?,
                        confidence = ?,
                        updated_at = ?,
                        source_dialogue_id = ?
                WHEN NOT MATCHED THEN
                    INSERT (user_id, entity_type, entity_key, entity_value, confidence, source_dialogue_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                SELECT id FROM entities 
                WHERE user_id = ? AND entity_type = ? AND entity_key = ?;
            """, (
                # USING 参数
                entity.user_id, entity.entity_type, entity.entity_key,
                # UPDATE 参数
                entity.entity_value, entity.confidence, now, entity.source_dialogue_id,
                # INSERT 参数
                entity.user_id, entity.entity_type, entity.entity_key, 
                entity.entity_value, entity.confidence, entity.source_dialogue_id, 
                entity.created_at or now, now,
                # SELECT 参数
                entity.user_id, entity.entity_type, entity.entity_key
            ))
            
            result = await cursor.fetchone()
            await conn.commit()
            return result[0] if result else 0

    async def get_entities(
        self, 
        user_id: str, 
        entity_type: Optional[str] = None
    ) -> List[EntityInfo]:
        """获取用户实体信息"""
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            
            if entity_type:
                await cursor.execute("""
                    SELECT * FROM entities 
                    WHERE user_id = ? AND entity_type = ?
                    ORDER BY updated_at DESC
                """, (user_id, entity_type))
            else:
                await cursor.execute("""
                    SELECT * FROM entities 
                    WHERE user_id = ?
                    ORDER BY updated_at DESC
                """, (user_id,))

            columns = [desc[0] for desc in cursor.description]
            entities = []
            for row in await cursor.fetchall():
                row_dict = dict(zip(columns, row))
                entities.append(EntityInfo(
                    id=row_dict.get('id'),
                    user_id=row_dict.get('user_id'),
                    entity_type=row_dict.get('entity_type'),
                    entity_key=row_dict.get('entity_key'),
                    entity_value=row_dict.get('entity_value'),
                    confidence=row_dict.get('confidence'),
                    source_dialogue_id=row_dict.get('source_dialogue_id'),
                    created_at=row_dict.get('created_at'),
                    updated_at=row_dict.get('updated_at')
                ))
            return entities

    async def delete_old_dialogues(self, days: int) -> int:
        """删除指定天数前的对话记录"""
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # 先统计数量
            await cursor.execute("""
                SELECT COUNT(*) FROM dialogues 
                WHERE created_at < ?
            """, (cutoff_date,))
            result = await cursor.fetchone()
            count = result[0] if result else 0
            
            # 执行删除
            await cursor.execute("""
                DELETE FROM dialogues 
                WHERE created_at < ?
            """, (cutoff_date,))
            
            await conn.commit()
            
            if count > 0:
                logging.info(f"🗑️ 自动清理: 删除了 {count} 条 {days} 天前的对话记录")
            return count


class MemoryStoreFactory:
    """记忆存储工厂类"""

    @staticmethod
    async def create_store(config: MemoryConfig) -> BaseMemoryStore:
        """根据配置创建对应的存储实例（异步）"""
        if config.db_type.lower() == "sqlite":
            store = SQLiteMemoryStore(config.sqlite_path)
            await store.initialize()
            return store
        elif config.db_type.lower() == "sqlserver":
            store = SQLServerMemoryStore(
                host=config.sqlserver_host,
                port=config.sqlserver_port,
                user=config.sqlserver_user,
                password=config.sqlserver_password,
                database=config.sqlserver_database,
                driver=config.sqlserver_driver
            )
            await store.initialize()
            return store
        else:
            raise ValueError(f"不支持的数据库类型: {config.db_type}")
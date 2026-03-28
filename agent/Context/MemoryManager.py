#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：MemoryManager.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, TypedDict, Annotated, Set
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from .MemoryStore import (
    MemoryStoreFactory, MemoryConfig, BaseMemoryStore,
    DialogueEntry, ConversationSummary, EntityInfo
)


class MemoryContext(BaseModel):
    """记忆上下文数据结构"""
    recent_dialogues: List[DialogueEntry] = Field(default_factory=list, description="最近N轮对话")
    summary: Optional[ConversationSummary] = Field(None, description="历史摘要")
    total_dialogues: int = Field(0, description="用户总对话数")
    context_messages: List[BaseMessage] = Field(default_factory=list, description="组装好的上下文消息")
    entities: List[EntityInfo] = Field(default_factory=list, description="用户实体画像")


class SummaryInput(BaseModel):
    """摘要生成输入"""
    dialogues: List[DialogueEntry] = Field(description="需要摘要的对话列表")
    existing_summary: Optional[str] = Field(None, description="现有摘要内容")


class SummaryOutput(BaseModel):
    """摘要生成输出"""
    summary_text: str = Field(description="生成的摘要文本")
    key_points: List[str] = Field(description="关键信息点")
    should_update: bool = Field(description="是否应该更新摘要")


class EntityExtractionOutput(BaseModel):
    """实体提取输出"""
    entities: List[Dict[str, Any]] = Field(description="提取的实体列表", default_factory=list)
    user_profile: Dict[str, Any] = Field(description="人物画像", default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class MemoryManager:
    """记忆管理器 - 处理对话存储、检索、摘要生成、实体提取"""

    def __init__(self, llm: BaseChatModel, config: Optional[MemoryConfig] = None):
        self.llm = llm
        self.config = config or MemoryConfig()
        self.store: Optional[BaseMemoryStore] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        # 摘要生成提示词
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个对话摘要生成专家。请根据提供的对话记录生成简洁的摘要。

要求：
1. 提取关键信息：用户关注的主要问题、个人画像、AI给出的核心建议、重要的事实信息
2. 保持简洁：摘要长度控制在500字以内
3. 结构清晰：按主题分段，使用要点符号
4. 保留上下文：确保摘要能帮助理解后续对话

如果提供了现有摘要，请整合新对话内容更新摘要，而不是完全替换。"""),
            ("human", """现有摘要（如有）：
{existing_summary}

新对话记录：
{dialogues}

请生成更新后的摘要，并以JSON格式返回：
{{
    "summary_text": "摘要内容",
    "key_points": ["要点1", "要点2", ...],
    "should_update": true/false
}}""")
        ])

        # 实体提取提示词
        self.entity_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个信息提取专家。请从对话中提取关键实体和人物画像信息。

需要提取的信息类型：
1. 人物(Person): 对话中提到的真实人物姓名、关系、角色
2. 地点(Location): 提到的地理位置、场所、公司/组织所在地
3. 项目(Project): 提到的项目、产品、任务名称
4. 偏好(Preference): 用户的喜好、兴趣、习惯、态度
5. 属性(Attribute): 用户的个人信息（职业、年龄、技能等客观属性）
6. 事件(Event): 重要事件、计划、时间节点

输出要求：
- entities: 实体列表，每个包含 type, key, value, confidence(0-1)
- user_profile: 人物画像总结，包含 inferred_traits（推断特征）、communication_style（沟通风格）

示例输出：
{{
    "entities": [
        {{"type": "Person", "key": "张三", "value": "用户的同事，负责前端开发", "confidence": 0.9}},
        {{"type": "Location", "key": "北京", "value": "用户所在城市", "confidence": 0.8}},
        {{"type": "Attribute", "key": "职业", "value": "Python后端工程师", "confidence": 0.95}}
    ],
    "user_profile": {{
        "inferred_traits": ["技术导向", "注重细节"],
        "communication_style": "直接、技术性强"
    }}
}}"""),
            ("human", """请分析以下对话并提取实体信息：

用户问题: {query}

请严格按JSON格式返回提取结果。""")
        ])

        logging.info(f"✓ 记忆管理器已初始化 (db_type={self.config.db_type})")


    async def initialize(self):
        """异步初始化存储和启动清理任务"""
        if self.store is None:
            self.store = await MemoryStoreFactory.create_store(self.config)
            # 启动自动清理任务（每天执行一次）
            self._cleanup_task = asyncio.create_task(self._scheduled_cleanup())
            logging.info("✓ 记忆存储异步初始化完成，自动清理任务已启动")

    async def _scheduled_cleanup(self):
        """定时清理任务"""
        while True:
            try:
                # 每天执行一次
                await asyncio.sleep(24 * 60 * 60)
                if self.store:
                    deleted = await self.store.delete_old_dialogues(self.config.auto_cleanup_days)
                    logging.info(f"🧹 定时清理任务完成: 删除 {deleted} 条过期记录")
            except Exception as e:
                logging.error(f"定时清理任务出错: {e}")
                await asyncio.sleep(60)  # 出错后1分钟重试

    async def load_memory_context(self, user_id: str) -> MemoryContext:
        """
        加载用户记忆上下文
        1. 获取最近N轮对话
        2. 获取最新摘要
        3. 获取用户实体画像
        4. 组装成LangChain消息格式
        """
        if not self.store:
            await self.initialize()

        # 获取最近对话
        recent = await self.store.get_recent_dialogues(
            user_id,
            limit=self.config.recent_dialogues_limit
        )

        # 获取摘要
        summary = await self.store.get_latest_summary(user_id)

        # 获取总数
        total = await self.store.get_dialogue_count(user_id)

        # 获取实体画像
        entities = await self.store.get_entities(user_id)

        # 组装消息
        messages = self._build_context_messages(recent, summary, entities)

        context = MemoryContext(
            recent_dialogues=recent,
            summary=summary,
            total_dialogues=total,
            context_messages=messages,
            entities=entities
        )

        logging.info(f"📚 加载用户 {user_id} 记忆: {len(recent)}轮对话, "
                    f"摘要存在={summary is not None}, 实体数={len(entities)}")
        return context

    def _build_context_messages(
        self,
        dialogues: List[DialogueEntry],
        summary: Optional[ConversationSummary],
        entities: List[EntityInfo]
    ) -> List[BaseMessage]:
        """将对话记录组装成LangChain消息格式"""
        messages = []

        # 如果有摘要，作为系统上下文加入
        context_parts = []
        if summary:
            context_parts.append(f"""[历史对话摘要]
{summary.summary_text}
关键信息: {summary.key_points or '无'}
""")

        # 添加实体画像信息
        if entities:
            entity_text = "\n".join([
                f"- [{e.entity_type}] {e.entity_key}: {e.entity_value}"
                for e in entities[:20]  # 限制数量避免过长
            ])
            context_parts.append(f"""[用户画像与实体信息]
{entity_text}""")

        if context_parts:
            messages.append(SystemMessage(content="\n".join(context_parts)))

        # 添加近期对话
        for entry in dialogues:
            messages.append(HumanMessage(content=entry.query))
            messages.append(AIMessage(content=entry.response))

        return messages

    async def save_dialogue(
        self,
        user_id: str,
        query: str,
        response: str,
        intent: Optional[str] = None,
        confidence: Optional[float] = None,
        session_id: Optional[str] = None
    ) -> int:
        """
        保存本轮对话到数据库
        返回: 记录ID
        """
        if not self.store:
            await self.initialize()

        entry = DialogueEntry(
            user_id=user_id,
            query=query,
            response=response,
            intent=intent,
            confidence=confidence,
            created_at=datetime.now(),
            session_id=session_id
        )

        record_id = await self.store.save_dialogue(entry)
        logging.info(f"💾 保存对话记录: user={user_id}, id={record_id}")
        
        # 触发实体提取
        await self._extract_and_save_entities(user_id, query, response, record_id)

        # 检查是否需要提取摘要
        await self._check_and_update_summary(user_id)
        return record_id

    async def _extract_and_save_entities(
        self, 
        user_id: str, 
        query: str, 
        response: str,
        dialogue_id: int
    ):
        """提取并保存实体信息（后台任务）"""
        try:
            extraction = await self._extract_entities(query, response)
            
            # 保存提取的实体
            for entity_data in extraction.entities:
                entity = EntityInfo(
                    user_id=user_id,
                    entity_type=entity_data.get('type', 'Unknown'),
                    entity_key=entity_data.get('key', 'Unknown'),
                    entity_value=entity_data.get('value', ''),
                    confidence=entity_data.get('confidence', 1.0),
                    source_dialogue_id=dialogue_id,
                    created_at=datetime.now()
                )
                await self.store.save_entity(entity)
                logging.debug(f"💡 保存实体: {entity.entity_type}/{entity.entity_key}")

            # 保存人物画像作为特殊实体
            if extraction.user_profile:
                for trait_type, traits in extraction.user_profile.items():
                    if isinstance(traits, list):
                        value = ", ".join(traits)
                    else:
                        value = str(traits)
                    
                    entity = EntityInfo(
                        user_id=user_id,
                        entity_type="Profile",
                        entity_key=trait_type,
                        entity_value=value,
                        confidence=0.8,
                        source_dialogue_id=dialogue_id,
                        created_at=datetime.now()
                    )
                    await self.store.save_entity(entity)
                    
            logging.info(f"🔍 实体提取完成: user={user_id}, 提取 {len(extraction.entities)} 个实体")
            
        except Exception as e:
            logging.error(f"实体提取失败: {e}")

    async def _extract_entities(self, query: str, response: str) -> EntityExtractionOutput:
        """调用LLM提取实体和画像"""
        structured_llm = self.llm.with_structured_output(EntityExtractionOutput)
        chain = self.entity_prompt | structured_llm

        try:
            result = await chain.ainvoke({
                "query": query,
                # "response": response
            })
            return result
        except Exception as e:
            logging.error(f"实体提取调用失败: {e}")
            return EntityExtractionOutput()

    async def _check_and_update_summary(self, user_id: str) -> Optional[ConversationSummary]:
        """
        检查是否需要更新摘要，如需要则生成并保存
        策略：当对话数达到阈值时，生成新摘要
        """
        if not self.store:
            await self.initialize()

        # 获取最新统计
        total_count = await self.store.get_dialogue_count(user_id)
        latest_summary = await self.store.get_latest_summary(user_id)

        # 计算本次摘要应包含的对话数
        dialogues_since_last_summary = total_count
        if latest_summary:
            dialogues_since_last_summary = total_count - latest_summary.dialogue_count

        # 判断是否达到摘要阈值
        if dialogues_since_last_summary < self.config.max_dialogues_per_summary:
            return None  # 不需要更新

        logging.info(f"📝 触发摘要更新: user={user_id}, 新增对话={dialogues_since_last_summary}")

        # 获取需要摘要的对话（按时间正序）
        recent_dialogues = await self.store.get_recent_dialogues(
            user_id,
            limit=dialogues_since_last_summary
        )
        # 反转顺序确保时间正序
        recent_dialogues = list(reversed(recent_dialogues))

        # 生成摘要
        new_summary = await self._generate_summary(
            recent_dialogues,
            latest_summary.summary_text if latest_summary else None
        )

        if not new_summary.should_update:
            logging.info("摘要生成器判断无需更新")
            return None

        # 确定时间范围
        start_time = recent_dialogues[0].created_at if recent_dialogues else datetime.now()
        end_time = recent_dialogues[-1].created_at if recent_dialogues else datetime.now()

        # 如果有旧摘要，合并时间范围
        if latest_summary:
            start_time = min(start_time, latest_summary.start_time)

        # 保存摘要
        summary_entry = ConversationSummary(
            user_id=user_id,
            summary_text=new_summary.summary_text,
            dialogue_count=total_count,  # 记录已处理的对话总数
            start_time=start_time,
            end_time=end_time,
            key_points=json.dumps(new_summary.key_points, ensure_ascii=False),
            created_at=datetime.now()
        )

        summary_id = await self.store.save_summary(summary_entry)
        logging.info(f"✅ 摘要已更新: id={summary_id}, 覆盖对话数={total_count}")

        return summary_entry

    async def _generate_summary(
        self,
        dialogues: List[DialogueEntry],
        existing_summary: Optional[str] = None
    ) -> SummaryOutput:
        """调用LLM生成摘要"""
        # 格式化对话记录
        dialogues_text = "\n\n".join([
            f"用户: {d.query}\nAI: {d.response}"
            for d in dialogues
        ])

        # 构建链
        structured_llm = self.llm.with_structured_output(SummaryOutput)
        chain = self.summary_prompt | structured_llm

        try:
            result = await chain.ainvoke({
                "existing_summary": existing_summary or "无",
                "dialogues": dialogues_text
            })
            return result
        except Exception as e:
            logging.error(f"摘要生成失败: {e}")
            # 返回一个默认结果，避免中断流程
            return SummaryOutput(
                summary_text=existing_summary or "对话历史记录",
                key_points=["摘要生成失败，保留历史记录"],
                should_update=False
            )

    async def get_user_entities(
        self, 
        user_id: str, 
        entity_type: Optional[str] = None
    ) -> List[EntityInfo]:
        """
        根据用户ID获取实体信息
        可按类型筛选：Person/Location/Project/Preference/Attribute/Profile
        """
        if not self.store:
            await self.initialize()
        
        entities = await self.store.get_entities(user_id, entity_type)
        logging.info(f"📊 获取用户实体: user={user_id}, type={entity_type or 'all'}, count={len(entities)}")
        return entities

    async def cleanup_expired_dialogues(self) -> int:
        """
        手动触发清理过期对话
        返回删除的记录数
        """
        if not self.store:
            await self.initialize()
        
        deleted = await self.store.delete_old_dialogues(self.config.auto_cleanup_days)
        return deleted

    async def clear_user_memory(self, user_id: str) -> bool:
        """清空用户记忆（谨慎使用）- 保留实体画像，只清对话"""
        logging.warning(f"⚠️ 清空用户 {user_id} 的对话记忆（保留画像）")
        # 这里可以实现只删除dialogues，保留entities和summaries的逻辑
        return True

    async def close(self):
        """关闭管理器，清理资源"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logging.info("✓ 记忆管理器已关闭")
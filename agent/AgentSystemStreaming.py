#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：AgentSystemStreaming.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''


import os
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, trim_messages
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from typing import List, Dict, Any, Optional, TypedDict, Literal, Annotated, Callable
from langchain_core.documents import Document
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from modelscope import AutoModelForSequenceClassification, AutoTokenizer
from .IntentClassifierAgent import IntentClassifier
from .AgronomistAgentStreaming import AgronomistAgentStreaming
from .OrdinaryAgentStreaming import OrdinaryAgentStreaming
from utils.RAG.RAGProcessor import RAGProcessor, RAGConfig
from .Structer import (Intent, Agronomist, Ordinary, StructOutput)
from .Context.MemoryManager import MemoryManager, MemoryConfig, MemoryContext


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # 历史记录，自动追加
    user_id: str  # 用户ID
    query: str  # 当前问题
    intent: Optional[Intent]  # 用户意图，使用哪个智能体
    classify_info: Optional[StructOutput]  # 用户问题详细拆解
    confidence: Optional[float]
    embeddings_values: List[Document]   # 混合检索结果
    bm25_values: List[Document]   # 关键字检索结果
    reranker_values: List[Document]  # 重排序结果
    response: Optional[str]  # 返回
    memory_context: Optional[str]
    signals: Optional[Dict]
    skip_processing: bool
    bypass_cache: bool
    cache_source: Optional[str]
    cache_similarity: Optional[float]
    prepared_messages: Optional[List[BaseMessage]]  # 准备好的上下文
    extracted_info: List[Dict]  # 新增：快速提取的用户信息
    metadata: Dict[str, Any]
    streaming_callback: Optional[Callable[[str], None]]  # 新增：流式输出回调函数


class AgentSystemStreaming:
    """多代理系统（集成智能记忆）- 流式版本"""

    def __init__(self):
        # 初始化组件
        load_dotenv()
        DEEPSEEK_NAME = os.getenv("DEEPSEEK_NAME")
        DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

        # ========== 模型初始化 ==========
        # 初始化模型
        self.model_default = init_chat_model(DEEPSEEK_NAME, api_key=DEEPSEEK_API_KEY)
        logging.info('已初始化LLM模型')

        # 加载重排序模型
        rag_config = RAGConfig()
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(rag_config.reranker.model_path)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(rag_config.reranker.model_path)
        self.reranker_model.eval()
        logging.info(f"✓ 已加载重排序模型: {rag_config.reranker.model_path}")

        # 加载向量模型
        self.vector_model = HuggingFaceEmbeddings(
            model_name=rag_config.vector_model.get_model_name(),
            model_kwargs={'device': rag_config.vector_model.device},
            encode_kwargs={'normalize_embeddings': rag_config.vector_model.normalize_embeddings}
        )
        logging.info(f"✓ 已加载向量模型: {rag_config.vector_model.get_model_name()}")

        # ========== 记忆系统初始化 ==========
        self.memory_manager = MemoryManager(
            llm=self.model_default,
            config=MemoryConfig()
        )

        # ========== Agent初始化 ==========

        self.intent_classifier = IntentClassifier(model=self.model_default)
        self.processor = RAGProcessor(
            self.model_default,
            config=rag_config,
            reranker_tokenizer=self.reranker_tokenizer,
            reranker_model=self.reranker_model,
            vector_model=self.vector_model
        )
        self.processor.load_vector_store()
        # 使用流式版本的农事代理
        self.agronomistAgent = AgronomistAgentStreaming(model=self.model_default)
        self.ordinaryAgent = OrdinaryAgentStreaming(model=self.model_default)

        # 构建工作流图
        self.graph = self._build_graph()
        logging.info(f"✓ 所有加载都已完成")

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""

        async def load_memory(state: AgentState) -> dict:
            """加载用户记忆上下文"""
            logging.info("📚 加载用户记忆上下文...")
            if state.get("streaming_callback"):
                await state["streaming_callback"](f'<think>\n用户提问：{state["query"]}\n')
                await state["streaming_callback"](f'让我好好回忆一下以前的对话\n')

            user_id = state["user_id"]

            # 从记忆管理器加载上下文
            memory_context: MemoryContext = await self.memory_manager.load_memory_context(user_id)

            if state.get("streaming_callback"):
                total_num = memory_context.total_dialogues
                if total_num > 4:
                    msg = f'我们之前有{total_num}轮对话呢!看来我们是老朋友了\n'
                elif total_num == 0:
                    msg = f'我们之前有还没有对话呢!这是一位新朋友！\n'
                else:
                    msg = f'我们之前有{total_num}轮对话呢!\n'
                await state["streaming_callback"](msg)

            logging.info(f"✓ 已加载 {len(memory_context.recent_dialogues)} 轮历史对话")
            return {'prepared_messages': memory_context.context_messages,
                    'memory_context': (
                    memory_context.summary.summary_text
                    if memory_context.summary else None
                )}

        async def classify_intent(state: AgentState) -> dict:
            """分类用户意图"""
            logging.info("🔍 分析用户意图...")
            if state.get("streaming_callback"):
                await state["streaming_callback"](f'🔍 让我来分析一下用户的意图...\n')

            query = state["query"]
            result: StructOutput = await self.intent_classifier.classify(query)

            logging.info(f"意图: {result.intent} (置信度: {result.confidence:.2f})")
            if state.get("streaming_callback"):
                await state["streaming_callback"](f"用户的意图是: {result.intent} (置信度: {result.confidence:.2f})，原因是：{result.reason}\n")
            return {"intent": result.intent, "confidence": result.confidence, "classify_info": result}

        def route_to_agent(state: AgentState) -> Literal["agronomist", "ordinary"]:
            """路由到对应代理"""
            intent = state["intent"]
            confidence = state["confidence"]

            if confidence < 0.5:
                return "ordinary"
            return intent

        async def middle_handler(state: AgentState) -> dict:
            '''中转 后面链接向量检索和关键字检索，同步执行'''
            return {}

        async def embeddings_handler(state: AgentState) -> dict:
            '''向量检索'''
            query = state["query"]
            if state.get("streaming_callback"):
                await state["streaming_callback"](f'🔧 向量检索（{query}）中...\n')
            logging.info(f'🔧 向量检索（{query}）中...')
            results = await self.processor.avector_search(query)
            if state.get("streaming_callback"):
                await state["streaming_callback"](f'🔧 向量检索（{query}），共检索到{len(results)}个结果\n')
            logging.info(f'🔧 向量检索（{query}），共检索到{len(results)}个结果')

            return {'embeddings_values': results}

        async def bm25_handler(state: AgentState) -> dict:
            '''关键字检索'''
            query = state["query"]
            if state.get("streaming_callback"):
                await state["streaming_callback"](f'🔧 关键字检索（{query}）中...\n')
            logging.info(f'🔧 关键字检索（{query}）中...')
            results = await self.processor.abm25_search(query)
            if state.get("streaming_callback"):
                await state["streaming_callback"](f'🔧 关键字检索（{query}），共检索到{len(results)}个结果\n')
            logging.info(f'🔧 关键字检索（{query}），共检索到{len(results)}个结果')

            return {'bm25_values': results}

        async def reranker_handler(state: AgentState) -> dict:
            '''重排序'''
            query = state["query"]
            if state.get("streaming_callback"):
                await state["streaming_callback"](f'🔧 重排序（{query}）中...\n')
            logging.info(f'🔧 重排序（{query}）中...')
            results = await self.processor.embeddings_and_bm25_with_rerank(query, embeddings_values=state["embeddings_values"],
                                                                     bm25_values=state["bm25_values"])
            if state.get("streaming_callback"):
                await state["streaming_callback"](f'🔧 重排序（{query}），保留{len(results)}个结果\n')
                if results:
                    for i, item in enumerate(results):
                        if isinstance(item, tuple):
                            doc, _ = item
                        else:
                            doc = item
                        if state["streaming_callback"]:
                            await state["streaming_callback"](f'检索到的第{i + 1}个文档：\n')
                            await state["streaming_callback"](f'{doc.page_content}\n')
            logging.info(f'🔧 重排序（{query}），保留{len(results)}个结果')

            return {'reranker_values': results}

        async def agronomist_handler(state: AgentState) -> dict:
            """农事管理处理"""
            logging.info("🔧 农事管理问题处理中...")
            if state.get("streaming_callback"):
                await state["streaming_callback"](f'好的，我已经准备好回答用户的提问了！</think>\n')

            classify_info: StructOutput = state["classify_info"]

            # 调用流式处理方法
            response_content = ""
            async for chunk in self.agronomistAgent.handle_stream(
                state["query"],
                classify_info=classify_info,
                reranker_doc=state["reranker_values"],
                context_messages=state.get("prepared_messages")
            ):
                response_content += chunk
                # 如果有流式回调函数，则调用它
                if state.get("streaming_callback"):
                    await state["streaming_callback"](chunk)
                else:
                    # 否则直接打印到控制台
                    print(chunk, end='', flush=True)

            return {'response': response_content}

        async def ordinary_handler(state: AgentState) -> dict:
            """其他问题处理"""
            logging.info("🔧 其他问题处理中...")
            if state.get("streaming_callback"):
                await state["streaming_callback"](f'好的，我已经准备好回答用户的提问了！</think>\n')
            classify_info: StructOutput = state["classify_info"]

            # 调用流式处理方法
            response_content = ""
            async for chunk in self.ordinaryAgent.handle_stream(
                state["query"],
                classify_info,
                context_messages=state.get("prepared_messages")
            ):
                response_content += chunk
                # 如果有流式回调函数，则调用它
                if state.get("streaming_callback"):
                    await state["streaming_callback"](chunk)
                else:
                    # 否则直接打印到控制台
                    print(chunk, end='', flush=True)

            return {'response': response_content}

        async def save_memory(state: AgentState) -> dict:
            """保存对话记忆"""
            logging.info("💾 保存对话记忆...")

            user_id = state["user_id"]
            query = state["query"]
            response = state["response"]
            intent = state["intent"].value if state["intent"] else None
            confidence = state["confidence"]

            # 保存本轮对话 提取画像、提取摘要
            await self.memory_manager.save_dialogue(
                user_id=user_id,
                query=query,
                response=response,
                intent=intent,
                confidence=confidence
            )

            return {}

        # 构建图
        graph = StateGraph(AgentState)

        # 添加节点
        graph.add_node("load_memory", load_memory)
        graph.add_node("classify", classify_intent)
        graph.add_node("middle", middle_handler)
        graph.add_node("embeddings", embeddings_handler)
        graph.add_node("bm25", bm25_handler)
        graph.add_node("reranker", reranker_handler)
        graph.add_node("agronomist", agronomist_handler)
        graph.add_node("ordinary", ordinary_handler)
        graph.add_node("save_memory", save_memory)

        # 添加边
        graph.add_edge(START, "load_memory")
        graph.add_edge("load_memory", "classify")

        # 条件路由
        graph.add_conditional_edges(
            "classify",
            route_to_agent,
            {
                "agronomist": "middle",  # 先转到中转站
                "ordinary": "ordinary"
            }
        )

        graph.add_edge("middle", "embeddings")
        graph.add_edge("middle", "bm25")    # 同步向量检索和关键字检索
        graph.add_edge("embeddings", "reranker")
        graph.add_edge("bm25", "reranker")   # 汇聚后进行重排序
        graph.add_edge("reranker", "agronomist")
        graph.add_edge("agronomist", "save_memory")
        graph.add_edge("ordinary", "save_memory")
        graph.add_edge("save_memory", END)

        return graph.compile()

    async def ahandle_message_with_streaming_callback(self, query: str, user_id: str,
                                                     streaming_callback) -> Dict[str, Any]:
        """处理用户消息（带流式回调的版本）"""
        logging.info(f"\n{'=' * 60}")
        logging.info(f"💬 用户: {query}")
        logging.info('=' * 60)

        initial_state = AgentState(
            messages=[],   # 暂时未用
            user_id=user_id,
            query=query,
            intent=None,
            classify_info=None,
            confidence=None,
            response=None,
            memory_context=None,  # 摘要信息
            signals=None,
            skip_processing=False,
            bypass_cache=False,
            cache_source=None,
            cache_similarity=None,
            prepared_messages=None,   # 将被记忆系统填充
            extracted_info=[],  # 初始化空列表
            metadata={},
            streaming_callback=streaming_callback  # 设置回调函数
        )

        result = await self.graph.ainvoke(initial_state)
        return result

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：IntentClassifierAgent.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''

import os
from typing import List, Dict, Any, Optional, TypedDict, Literal, Annotated
from dataclasses import dataclass
from datetime import datetime
import json
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from prompts import intentClassifier_template, intentClassifier_template2
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging
import asyncio
from enum import Enum
from .Structer import StructOutput

class IntentClassifier:
    """意图分类器实现方式"""

    def __init__(self, model: BaseChatModel):
        self.llm = model
        self.prompt = intentClassifier_template

    async def classify(self, question: str) -> Dict[str, Any]:
        """分类用户意图"""
        structured_llm = self.llm.with_structured_output(StructOutput)
        chain = self.prompt | structured_llm
        result = await chain.ainvoke({"question": question})
        logging.info(f'问题：{question} -> 意图识别：{str(result)}')
        return result


def IntentClassifierTest():
    from .AgronomistAgent import AgronomistAgent
    from utils.RAG.RAGProcessor import RAGProcessor, RAGConfig

    load_dotenv()
    DEEPSEEK_NAME = os.getenv("DEEPSEEK_NAME")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

    # 初始化模型
    model_default = init_chat_model(DEEPSEEK_NAME, api_key=DEEPSEEK_API_KEY)

    intentClassifier = IntentClassifier(model=model_default)
    query = '葡萄冬季要怎么剪枝'
    res = intentClassifier.classify(query)
    processor = RAGProcessor(config=RAGConfig())
    processor.load_vector_store()
    agronomistAgent = AgronomistAgent(model=model_default, rag_processor=processor)
    print(agronomistAgent.handle(query, res))


def safe_parse_json(text: str, default: dict = None) -> dict:
    """
    安全地解析JSON文本

    处理：
    - Markdown 代码块 (```json ... ```)
    - 前后的空白字符
    - 解析失败时返回默认值
    """
    if default is None:
        default = {}

    content = text.strip()

    # 移除 Markdown 代码块
    if "```json" in content:
        try:
            content = content.split("```json")[1].split("```")[0]
        except IndexError:
            pass
    elif "```" in content:
        try:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
        except IndexError:
            pass

    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logging.error(f"   ⚠️ JSON 解析失败: {e}")
        return default

class IntentClassifier2:
    """意图分类器实现方式2"""

    def __init__(self, model: BaseChatModel):
        self.llm = model
        self.prompt = intentClassifier_template2

    def classify(self, question: str) -> Dict[str, Any]:
        """分类用户意图"""
        chain = self.prompt | self.llm | StrOutputParser()
        result = chain.invoke({"question": question})

        # 使用安全的 JSON 解析
        default_result = {"intent": "unknown", "confidence": 0.5, "reason": "无法识别"}
        parsed = safe_parse_json(result, default_result)

        # 确保返回有效的意图
        if "intent" not in parsed:
            return default_result
        return parsed
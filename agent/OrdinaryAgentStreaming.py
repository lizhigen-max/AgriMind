#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：OrdinaryAgentStreaming.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''

import os
import logging
from typing import List, Dict, Any, Optional, TypedDict, Literal, Annotated, AsyncGenerator
from prompts import ordinary_prompt
from utils.RAG.RAGProcessor import RAGProcessor
from langchain_core.language_models import BaseChatModel
from .Structer import StructOutput
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, trim_messages


class OrdinaryAgentStreaming:
    """一般对话 - 流式输出版本"""

    def __init__(self, model: BaseChatModel):
        self.llm = model
        self.prompt = ordinary_prompt

    async def handle_stream(self, question: str, classify_info: StructOutput = None,
                           context_messages: Optional[List[BaseMessage]] = None) -> AsyncGenerator[str, None]:
        """流式处理方法 - 真正实现流式输出"""
        # 简单对话
        if context_messages:
            messages = context_messages.copy()
            messages.append(HumanMessage(content=question))
        else:
            messages = [HumanMessage(content=question)]

        # 将system_prompt注入
        messages.insert(0, SystemMessage(content=self.prompt))

        # 直接使用模型的流式功能，而不是agent
        try:
            # 使用模型的 astream 方法实现真正的流式输出
            full_response = ""
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    # 只输出新增的部分
                    new_content = chunk.content
                    if new_content:
                        yield new_content
                        full_response += new_content

        except Exception as e:
            logging.error(f"流式处理出错: {e}")
            # 回退到普通调用
            try:
                result = self.llm.invoke(messages)
                if hasattr(result, 'content'):
                    yield result.content
                else:
                    yield "抱歉，处理过程中出现错误。"
            except Exception as e2:
                logging.error(f"回退处理也失败: {e2}")
                yield "抱歉，处理过程中出现错误。"
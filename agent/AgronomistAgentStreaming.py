#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：AgronomistAgentStreaming.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''

import os
import logging
from typing import List, Dict, Any, Optional, TypedDict, Literal, Annotated, AsyncGenerator
from prompts import agronomist_prompt
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from .Structer import StructOutput
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, trim_messages


class AgronomistAgentStreaming:
    """农事管理 - 流式输出版本"""

    def __init__(self, model: BaseChatModel):
        self.llm = model
        self.prompt = agronomist_prompt

    def handle(self, question: str, reranker_doc: List[Document] = None, classify_info: StructOutput = None,
               context_messages: Optional[List[BaseMessage]] = None) -> str:
        """非流式处理方法（保持兼容性）"""
        contexts = []
        if reranker_doc:
            for i, item in enumerate(reranker_doc):
                if isinstance(item, tuple):
                    doc, _ = item
                else:
                    doc = item
                contexts.append(doc.page_content)

        if contexts:
            context = f"""基于以下农业知识回答问题:
<农业知识>
{';'.join(contexts)}
</农业知识>
用户问题：{question}
"""
        else:
            context = f"""未检索到相关知识库内容。用户问题：{question}，请基于一般农业知识回答，或建议升级人工专员。"""
        if context_messages:
            messages = context_messages.copy()
            for m in messages:
                if isinstance(m, HumanMessage) and '用户问题：' in m.content:
                    m.content = m.content.split('用户问题：')[-1]  # 去除历史消息中知识库内容

            # 确保最后一条是当前查询
            if not isinstance(messages[-1], HumanMessage):
                messages.append(HumanMessage(content=context))
            else:
                if messages[-1].content == question:
                    messages[-1].content = context
                else:
                    messages.append(HumanMessage(content=context))
        else:
            messages = [HumanMessage(content=context)]

        # 将system_prompt注入
        messages.insert(0, SystemMessage(content=self.prompt))

        # 直接使用模型的 invoke 方法
        result = self.llm.invoke(messages)

        if hasattr(result, 'content'):
            return result.content
        return "抱歉，暂未收录相关信息，建议升级成人工专员"

    async def handle_stream(self, question: str, reranker_doc: List[Document] = None, classify_info: StructOutput = None,
                           context_messages: Optional[List[BaseMessage]] = None) -> AsyncGenerator[str, None]:
        """流式处理方法 - 真正实现流式输出"""
        # 调用知识库检索
        query_strengthen = ''
        if classify_info and classify_info.agronomist:
            query_strengthen = classify_info.query_strengthen

        contexts = []
        if reranker_doc:
            for i, item in enumerate(reranker_doc):
                if isinstance(item, tuple):
                    doc, _ = item
                else:
                    doc = item
                contexts.append(doc.page_content)

        if contexts:
            context = f"""基于以下农业知识回答问题:
<农业知识>
{';'.join(contexts)}
</农业知识>
用户问题：{question}
"""
        else:
            context = f"""未检索到相关知识库内容。用户问题：{question}。"""
        if context_messages:
            messages = context_messages.copy()
            messages.append(HumanMessage(content=context))
        else:
            messages = [HumanMessage(content=context)]

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

    def handle_stream_sync(self, question: str, classify_info: StructOutput = None,
                          context_messages: Optional[List[BaseMessage]] = None) -> str:
        """同步流式处理方法（用于测试）"""
        import asyncio

        # 创建事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # 定义异步生成器函数
        async def async_generator():
            async for chunk in self.handle_stream(question, classify_info, context_messages):
                yield chunk

        # 收集所有输出
        result = ""
        try:
            gen = async_generator()
            while True:
                try:
                    chunk = loop.run_until_complete(gen.__anext__())
                    result += chunk
                    print(chunk, end='', flush=True)
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

        return result


# 保持兼容性 - 创建别名
AgronomistAgent = AgronomistAgentStreaming  # 使用流式版本作为默认版本
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：StreamlitUI.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''

# Streamlit run .\StreamlitUI.py

import streamlit as st
import asyncio
import logging
import queue
import threading
import time
import sys
import os
import traceback
from typing import List, Dict, Any
from datetime import datetime
import uuid
from config import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="智能农业助手", page_icon="🌾", layout="wide")
st.title("🌾 智能农业助手")


# 初始化 Session State
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_lgz"
    if "agent_system" not in st.session_state:
        st.session_state.agent_system = None


init_session_state()

# 导入并初始化
try:
    from agent.AgentSystemStreaming import AgentSystemStreaming
except Exception as e:
    st.error(f"导入失败: {e}")
    st.stop()


@st.cache_resource
def get_agent_system():
    try:
        return AgentSystemStreaming()
    except Exception as e:
        st.error(f"初始化失败: {e}")
        return None


if st.session_state.agent_system is None:
    agent = get_agent_system()
    if agent is not None:
        st.session_state.agent_system = agent
        st.success("✅ 系统初始化完成")
    else:
        st.error("系统初始化失败")
        st.stop()

# 侧边栏
with st.sidebar:
    st.title("⚙️ 设置")
    user_id = st.text_input("👤 用户 ID", value=st.session_state.user_id)
    st.session_state.user_id = user_id
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

# 显示历史
for msg in st.session_state.messages:
    with st.chat_message(msg.get("role", "user")):
        st.markdown(msg.get("content", ""))


# ========== 核心：异步回调包装器 ==========
class AsyncCallbackRunner:
    """把同步回调包装成异步形式，解决 AgentSystem 里的 await 问题"""

    def __init__(self, text_queue: queue.Queue):
        self.text_queue = text_queue

    async def __call__(self, chunk: str):
        """异步调用，实际只是往队列放数据"""
        self.text_queue.put(chunk)
        # 让出控制权，避免阻塞
        await asyncio.sleep(0)


class StreamRunner:
    def __init__(self, agent_system, query: str, user_id: str, chat_history: List[Dict]):
        self.agent_system = agent_system
        self.query = query
        self.user_id = user_id
        self.chat_history = chat_history
        self.text_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.error_info = None

    def _run_async(self):
        """在独立线程中运行"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def process():
                try:
                    # 创建异步回调包装器
                    async_callback = AsyncCallbackRunner(self.text_queue)

                    # 调用 Agent，传入异步回调
                    result = await self.agent_system.ahandle_message_with_streaming_callback(
                        query=self.query,
                        user_id=self.user_id,
                        streaming_callback=async_callback,  # 传入异步包装器
                        # chat_history=self.chat_history
                    )

                    if result is None:
                        self.error_info = "Agent 返回 None"
                        self.result_queue.put(("error", self.error_info))
                    else:
                        self.result_queue.put(("success", result))

                except Exception as e:
                    self.error_info = f"{type(e).__name__}: {str(e)}"
                    logger.error(f"处理异常: {self.error_info}")
                    logger.error(traceback.format_exc())
                    self.result_queue.put(("error", self.error_info))

            loop.run_until_complete(process())
            loop.close()

        except Exception as e:
            self.error_info = f"事件循环错误: {str(e)}"
            self.result_queue.put(("error", self.error_info))

    def start(self):
        self.thread = threading.Thread(target=self._run_async, daemon=True)
        self.thread.start()

    def get_updates(self):
        updates = []
        while True:
            try:
                updates.append(self.text_queue.get_nowait())
            except queue.Empty:
                break
        return updates

    def get_result(self, timeout=0.1):
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def is_alive(self):
        return self.thread.is_alive()


# ========== 用户输入 ==========
if prompt := st.chat_input("请输入您的问题"):
    if st.session_state.agent_system is None:
        st.error("系统未初始化")
        st.stop()

    # 添加用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # 准备历史
    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-11:-1]
    ]

    # 创建运行器
    runner = StreamRunner(
        st.session_state.agent_system,
        prompt,
        st.session_state.user_id,
        chat_history
    )
    runner.start()

    # 主线程轮询
    with st.chat_message("assistant"):
        full_response = ""
        org_response = ""
        start_time = time.time()
        max_wait = 300

        try:
            # 思考过程
            with st.expander("🧠 查看思考过程", expanded=True):
                while True:
                    if time.time() - start_time > max_wait:
                        raise TimeoutError("响应超时")

                    # 获取文本更新
                    updates = runner.get_updates()
                    if updates:
                        update_str = "".join(updates)
                        org_response += update_str
                        update_str = update_str.replace('\n', '<br>')
                        update_str = update_str.replace('<think>', '')
                        update_str = update_str.replace('</think>', '')
                        st.markdown(update_str, unsafe_allow_html=True)
                        if '<think>' in org_response and '</think>' in org_response:
                            logger.info(org_response)

                            # 检查是否有答案被截取
                            org_response_split = org_response.split('</think>')
                            if len(org_response_split) > 1 and len(org_response_split[-1]) > 0:
                                full_response = org_response_split[-1]
                            break

                    time.sleep(0.05)

            placeholder = st.empty()
            with st.spinner("回答中..."):
                while True:
                    if time.time() - start_time > max_wait:
                        raise TimeoutError("响应超时")

                    # 获取文本更新
                    updates = runner.get_updates()
                    if updates:
                        full_response += "".join(updates)
                        placeholder.markdown(full_response + "▌")

                    # 检查结果
                    status, result = runner.get_result(timeout=0.05)
                    if status is not None:
                        if status == "success":
                            placeholder.markdown(full_response)

                            if isinstance(result, dict):
                                intent = result.get("intent", "unknown")
                                conf = result.get("confidence", 0.0)
                                st.caption(f"意图: {intent} | 置信度: {conf:.1%}")

                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": full_response,
                                    "timestamp": datetime.now().isoformat(),
                                    "metadata": result
                                })
                            break
                        else:
                            raise Exception(result)

                    time.sleep(0.05)

        except Exception as e:
            placeholder = st.empty()
            placeholder.error(f"错误: {str(e)}")
            if runner.error_info:
                st.code(f"详细错误: {runner.error_info}")

st.divider()
st.caption("智能农业助手运行中")
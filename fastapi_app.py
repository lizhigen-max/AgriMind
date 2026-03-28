#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：fastapi_api.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''
"""
FastAPI流式输出接口
提供HTTP接口实现葡萄种植问答的流式输出
"""

import io
import sys
import asyncio
import uuid
from datetime import datetime
from typing import AsyncGenerator, Dict, Any

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
from config import *

# 导入环境变量
from dotenv import load_dotenv
load_dotenv()

# 添加项目根目录到Python路径
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.AgentSystemStreaming import AgentSystemStreaming


# 请求模型
class ChatRequest(BaseModel):
    question: str
    user_id: str = "default_user"
    chat_history: list = []


# 响应模型
class ChatResponse(BaseModel):
    message: str
    intent: str = None
    confidence: float = None


# 创建FastAPI应用
app = FastAPI(
    title="葡萄种植问答系统",
    description="基于多智能体的葡萄种植知识问答系统，支持流式输出",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局agent系统实例
agent_system = None


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化agent系统"""
    global agent_system
    try:
        print("🚀 初始化Agent系统...")
        agent_system = AgentSystemStreaming()
        print("✅ Agent系统初始化成功")
    except Exception as e:
        print(f"❌ Agent系统初始化失败: {e}")
        raise


# @app.on_event("shutdown")
# async def shutdown_event():
#     """应用关闭时清理资源"""
#     global agent_system
#     if agent_system:
#         print("🧹 清理资源...")
#         # 可以在这里执行一些清理操作
#         print("✅ 清理完成")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "葡萄种植问答系统API",
        "version": "1.0.0",
        "endpoints": [
            "/chat - 流式问答接口",
            "/health - 健康检查",
            "/docs - API文档"
        ]
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent_system": "initialized" if agent_system else "not_initialized"
    }


@app.post("/chat")
async def chat_streaming(request: ChatRequest):
    """
    流式问答接口 - 通过回调函数实现流式输出
    """
    if not agent_system:
        raise HTTPException(status_code=503, detail="Agent系统未初始化")

    # 创建异步队列用于桥接回调和生成器
    queue = asyncio.Queue()

    # 标记处理是否完成
    done_event = asyncio.Event()

    async def generate_stream():
        """消费队列，生成 SSE 流"""
        try:
            while not done_event.is_set() or not queue.empty():
                try:
                    # 非阻塞获取，超时则继续循环检查
                    chunk = await asyncio.wait_for(queue.get(), timeout=0.1)

                    if chunk is None:  # None 作为结束信号
                        break

                    data = {
                        "type": "chunk",
                        "content": chunk,
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                except asyncio.TimeoutError:
                    continue

        except Exception as e:
            error_data = {
                "type": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

        finally:
            yield "data: [DONE]\n\n"

    # 定义回调函数 - 将数据放入队列
    async def streaming_callback(chunk: str):
        """流式输出回调函数"""
        # 使用 asyncio.run_coroutine_threadsafe 在事件循环中放入队列
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(queue.put(chunk))
            else:
                loop.run_until_complete(queue.put(chunk))
        except RuntimeError:
            # 如果没有事件循环，创建新任务
            asyncio.run(queue.put(chunk))

    # 在后台运行 agent 处理
    async def run_agent():
        try:
            # 调用带回调的流式处理
            await agent_system.ahandle_message_with_streaming_callback(
                request.question,
                request.user_id,
                streaming_callback
            )
        except Exception as e:
            # 将错误信息放入队列
            await queue.put(f"[ERROR]: {str(e)}")
        finally:
            # 发送结束信号
            await queue.put(None)
            done_event.set()

    # 启动后台任务
    asyncio.create_task(run_agent())

    # 立即返回流式响应
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )



if __name__ == "__main__":
    # 运行服务器
    print("🚀 启动FastAPI服务器...")
    print("📋 API文档: http://localhost:8000/docs")
    print("💬 流式问答: POST http://localhost:8000/chat")

    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
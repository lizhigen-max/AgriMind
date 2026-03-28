#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：fastapi_app_test.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''
"""
FastAPI流式输出接口测试程序
测试流式问答接口的功能
"""

import io
import sys
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import aiohttp
import requests
from aiohttp import ClientTimeout, ClientSession


# 简单的同步测试函数
def simple_test():
    """简单的同步测试"""
    print("🚀 简单流式接口测试")
    print("=" * 60)

    base_url = "http://localhost:8000"
    question = "帮我创建一份海南5天旅游计划？"

    try:
        # 测试健康检查
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ 服务正常运行")
        else:
            print(f"❌ 服务异常: {response.status_code}")
            return

        print(f"\n❓ 问题: {question}")
        print("🤖 AI回答: ", end="", flush=True)

        # 发送流式请求
        response = requests.post(
            f"{base_url}/chat",
            json={"question": question, "user_id": "simple_test"},
            stream=True,
            timeout=600
        )

        if response.status_code == 200:
            # 逐块读取
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if data.get('type') == 'chunk':
                                print(data.get('content', ''), end='', flush=True)
                        except:
                            pass

            print("\n\n✅ 简单测试完成!")
        else:
            print(f"\n❌ 请求失败: {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请先启动FastAPI服务:")
        print("   python fastapi_app.py")
    except Exception as e:
        print(f"❌ 测试异常: {e}")




if __name__ == "__main__":
    # 检查命令行参数
    import sys
    simple_test()

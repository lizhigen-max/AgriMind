#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：_Streaming_test.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''

import time
import asyncio
import os
import sys
from config import *
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.AgentSystemStreaming import AgentSystemStreaming


async def simple_streaming_demo():
    print("🍇 葡萄农事问答系统 - 流式输出演示\n")

    # 创建流式代理系统
    agent_system = AgentSystemStreaming()

    # 用户ID
    user_id = "demo_user"

    # 测试问题
    test_questions = [
        "我叫李志根，从长沙去北京的推荐落脚旅游点",
        # "我之前问过哪几个城市旅游攻略来着？帮我总结一下"
        # "帮我做一份去长沙旅游的攻略",
        # "帮我做一份去广东旅游的攻略"
        # "葡萄怎么剪枝？",
        # "葡萄冬季修剪应该注意什么？",
        # "葡萄叶子发黄是什么原因？"
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ 问题: {question}")
        print(f"{'='*60}")
        print("🤖 AI回答 (流式输出):")
        print("-" * 40)

        # 使用带回调的流式处理来实现真正的流式输出
        async def streaming_callback(chunk: str):
            """流式输出回调函数 - 实时打印每个块"""
            print(chunk, end='', flush=True)

        # 调用带回调的流式处理
        result = await agent_system.ahandle_message_with_streaming_callback(
            question,
            user_id,
            streaming_callback
        )

        print("\n" + "-" * 40)
        print(f"✅ 回答完成!")
        print(f"📊 意图识别: {result.get('intent', 'unknown')}")
        print(f"🎯 置信度: {result.get('confidence', 0):.2f}")
        print(f"📄 总长度: {len(result.get('response', ''))} 字符")

        if result.get('skip_processing'):
            print(f"💾 命中缓存: {result.get('cache_source')}")

    # 结束会话
    print(f"\n{'='*60}")
    time.sleep(600)
    print("结束会话...")
    print("\n✨ 演示完成!")


# 主函数
async def main():
    """主函数 - 运行所有演示"""
    try:
        await simple_streaming_demo()
    except KeyboardInterrupt:
        print("\n\n演示被中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
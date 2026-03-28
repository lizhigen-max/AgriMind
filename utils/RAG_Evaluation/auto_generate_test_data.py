#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：auto+generate_test+data.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''
"""
从文档自动生成测试数据（问题 + 标准答案）
混合策略生成测试数据：
1. LLM自动生成（80%）
2. 模板问题（10%）
3. 人工标注关键问题（10%）
"""
import os
from typing import List, Dict
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
import json

from dotenv import load_dotenv
load_dotenv()

# 初始化 LLM
llm = init_chat_model(
    "deepseek:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)


def generate_qa_from_document(doc_path: str, num_questions: int = 10) -> List[Dict]:
    """
    从Word文档自动生成问答对

    Args:
        doc_path: Word文档路径
        num_questions: 生成问题数量

    Returns:
        [{"question": "...", "ground_truth": "...", "context": "..."}]
    """
    # 1. 加载Word文档
    print(f"📄 加载文档: {doc_path}")
    loader = UnstructuredWordDocumentLoader(doc_path)
    documents = loader.load()

    # 2. 分块（保持语义完整）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    print(f"✂️  文档分块: {len(chunks)} 个片段")

    # 3. 从每个chunk生成问答对
    qa_pairs = []

    for i, chunk in enumerate(chunks[:num_questions]):
        print(f"🤖 生成问答对 {i + 1}/{min(num_questions, len(chunks))}...")

        # 构建提示词
        prompt = f"""基于以下文档内容，生成1个高质量的问题和答案。

文档内容：
{chunk.page_content}

要求：
1. 问题要具体、明确，可以直接从文档中找到答案
2. 答案要完整、准确，包含文档中的关键信息
3. 答案长度控制在100-200字

请按以下JSON格式输出（不要包含其他内容）：
{{
    "question": "问题",
    "ground_truth": "标准答案",
    "category": "分类（如：果树栽培/病虫害防治）"
}}
"""

        # 调用LLM生成
        response = llm.invoke(prompt)

        try:
            # 解析JSON
            import json
            qa = json.loads(response.content)
            qa["context"] = chunk.page_content  # 保留原始上下文
            qa_pairs.append(qa)

        except Exception as e:
            print(f"⚠️  解析失败，跳过: {e}")
            continue

    return qa_pairs


def save_test_data(qa_pairs: List[Dict], output_path: str):
    """保存为测试数据格式"""
    test_data = {
        "metadata": {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "total_samples": len(qa_pairs),
            "source": "auto_generated"
        },
        "test_cases": qa_pairs
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 测试数据已保存: {output_path}")


# 使用示例
if __name__ == "__main__":
    from datetime import datetime

    # 从Word文档生成
    doc_path = "./温宿葡萄栽培手册.docx"  # 你的Word文档路径
    qa_pairs = generate_qa_from_document(doc_path, num_questions=20)

    # 保存
    save_test_data(qa_pairs, "auto_test_data.json")
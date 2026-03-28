#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AgriMind
@File    ：RAGAS_Evaluator.py
@IDE     ：PyCharm
@Author  ：lizhigen
@Email   ：lizhigen1996@aliyun.com
@Date    ：2026/3/28 21:08
'''
"""
基于官方 ragas 库的 RAG 全自动化评估框架

安装依赖:
    pip install ragas langchain langchain-community langchain-openai datasets

官方文档: https://docs.ragas.io/
"""
import os
import json
import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from ..RAG.RAGProcessor import RAGProcessor, RAGConfig as RAG_Proc_Config

import pandas as pd
from datasets import Dataset

# ragas 官方导入
from ragas import evaluate
from ragas.metrics import (
    # 检索指标
    _ContextPrecision,
    _ContextRecall,
    _ContextEntityRecall,
    _NoiseSensitivity,
    # 生成指标
    _Faithfulness,
    _ResponseRelevancy,
    _AnswerCorrectness,
    _AnswerSimilarity,
    # 综合
    _AspectCritic,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()


@dataclass
class RAGASConfig:
    """RAGAS 评估配置"""
    # 项目名称
    project_name: str = "RAG_Evaluation"

    # 输出目录
    output_dir: str = "./data/ragas_results"

    # LLM 配置（用于评估的法官模型）
    eval_llm_provider: str = "deepseek"  # openai, deepseek, azure
    eval_llm_model: str = "deepseek-chat"
    eval_llm_api_key: Optional[str] = None
    eval_llm_api_base: Optional[str] = None

    # Embedding 配置
    embedding_model: str = "./models/BAAI/bge-large-zh-v1.5"
    embedding_device: str = "cpu"

    # 评估指标
    metrics: List[str] = None

    # 并发配置
    max_workers: int = 4
    timeout: int = 120

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "context_precision",
                "context_recall",
                "context_entity_recall",
                "noise_sensitivity",
                "faithfulness",
                "response_relevancy",
                "answer_correctness",
                "answer_similarity"
            ]

        # 从环境变量读取 API key
        if not self.eval_llm_api_key:
            if self.eval_llm_provider == "deepseek":
                self.eval_llm_api_key = os.getenv("DEEPSEEK_API_KEY")
            else:
                self.eval_llm_api_key = os.getenv("OPENAI_API_KEY")


class RAGAS_Evaluator:
    """
    基于官方 ragas 库的 RAG 评估器

    特性:
    - 使用 ragas 官方指标 (Faithfulness, ContextRecall, AnswerCorrectness 等)
    - 支持多种 LLM 作为评估法官 (OpenAI, DeepSeek, Azure)
    - 自动化测试数据生成和评估
    - 生成专业可视化报告
    """

    def __init__(self, config: Optional[RAGASConfig] = None):
        self.config = config or RAGASConfig()

        # 创建输出目录
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # 初始化评估用的 LLM 和 Embedding
        self._init_llm()
        self._init_embedding()
        self._init_metrics()

    def _init_llm(self):
        """初始化评估用的 LLM（法官模型）"""
        try:
            if self.config.eval_llm_provider == "deepseek":
                self.eval_llm = init_chat_model(
                    "deepseek:deepseek-chat",
                    api_key=self.config.eval_llm_api_key
                )
            else:
                self.eval_llm = init_chat_model(
                    self.config.eval_llm_model,
                    api_key=self.config.eval_llm_api_key
                )

            # 包装为 ragas 格式
            self.ragas_llm = LangchainLLMWrapper(self.eval_llm)
            logging.info(f"✅ 评估 LLM 初始化完成: {self.config.eval_llm_model}")

        except Exception as e:
            logging.error(f"❌ LLM 初始化失败: {e}")
            raise

    def _init_embedding(self):
        """初始化评估用的 Embedding 模型"""
        try:
            self.embedding = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": self.config.embedding_device},
                encode_kwargs={"normalize_embeddings": True}
            )

            # 包装为 ragas 格式
            self.ragas_embedding = LangchainEmbeddingsWrapper(self.embedding)
            logging.info(f"✅ Embedding 模型初始化完成: {self.config.embedding_model}")

        except Exception as e:
            logging.error(f"❌ Embedding 初始化失败: {e}")
            raise

    def _init_metrics(self):
        """初始化评估指标"""
        self.metric_map = {
            # 检索指标
            "context_precision": _ContextPrecision(),
            "context_recall": _ContextRecall(),
            "context_entity_recall": _ContextEntityRecall(),
            "noise_sensitivity": _NoiseSensitivity(),
            # 生成指标
            "faithfulness": _Faithfulness(),
            "response_relevancy": _ResponseRelevancy(),
            "answer_correctness": _AnswerCorrectness(),
            "answer_similarity": _AnswerSimilarity(),
        }

        # 选择启用的指标
        self.selected_metrics = []
        for metric_name in self.config.metrics:
            if metric_name in self.metric_map:
                self.selected_metrics.append(self.metric_map[metric_name])

        logging.info(f"✅ 评估指标: {[m.name for m in self.selected_metrics]}")

    def prepare_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dataset:
        """
        准备评估数据集

        Args:
            questions: 问题列表
            answers: RAG 生成的回答列表
            contexts_list: 每个问题的检索上下文列表 (List[List[str]])
            ground_truths: 标准答案列表（可选，用于计算 correctness）

        Returns:
            datasets.Dataset 格式，兼容 ragas
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        }

        if ground_truths:
            data["ground_truth"] = ground_truths

        return Dataset.from_dict(data)

    def evaluate(
        self,
        dataset: Dataset,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        执行 ragas 评估

        Args:
            dataset: 评估数据集
            show_progress: 是否显示进度

        Returns:
            包含评估分数的 DataFrame
        """
        logging.info(f"🚀 开始评估 {len(dataset)} 个样本...")

        # 配置运行参数
        run_config = RunConfig(
            max_workers=self.config.max_workers,
            timeout=self.config.timeout
        )

        # 执行 ragas 评估
        result = evaluate(
            dataset=dataset,
            metrics=self.selected_metrics,
            llm=self.ragas_llm,
            embeddings=self.ragas_embedding,
            run_config=run_config,
            raise_exceptions=False,  # 遇到错误继续评估其他样本
            show_progress=show_progress
        )

        # 转换为 DataFrame
        result_df = result.to_pandas()

        logging.info("✅ 评估完成!")
        return result_df

    def evaluate_rag_pipeline(
        self,
        rag_pipeline: Callable[[str], Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        端到端评估 RAG 管道

        Args:
            rag_pipeline: RAG 管道函数，输入 question，返回 {"answer": str, "contexts": List[str]}
            test_data: 测试数据列表，每个元素包含 {"question": str, "ground_truth": str}

        Returns:
            评估结果 DataFrame
        """
        questions = []
        answers = []
        contexts_list = []
        ground_truths = []

        logging.info("🔄 执行 RAG 管道获取回答...")

        for i, item in enumerate(test_data):
            question = item["question"]
            ground_truth = item.get("ground_truth", "")

            logging.info(f"  处理 {i+1}/{len(test_data)}: {question[:50]}...")

            # 执行 RAG 管道
            try:
                result = rag_pipeline(question)
                answer = result.get("answer", "")
                contexts = result.get("contexts", [])

                questions.append(question)
                answers.append(answer)
                contexts_list.append(contexts)
                ground_truths.append(ground_truth)

            except Exception as e:
                logging.error(f"❌ 处理问题失败: {question}, 错误: {e}")
                # 填充空值保持对齐
                questions.append(question)
                answers.append("")
                contexts_list.append([])
                ground_truths.append(ground_truth)

        # 准备数据集
        dataset = self.prepare_dataset(questions, answers, contexts_list, ground_truths)

        # 执行评估
        return self.evaluate(dataset, show_progress)

    async def aevaluate_rag_pipeline(
        self,
        rag_pipeline: Callable[[str], Any],
        test_data: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """异步评估"""
        questions = []
        answers = []
        contexts_list = []
        ground_truths = []

        logging.info("🔄 异步执行 RAG 管道...")

        tasks = []
        for item in test_data:
            question = item["question"]
            ground_truth = item.get("ground_truth", "")
            ground_truths.append(ground_truth)
            questions.append(question)

            # 创建异步任务
            if asyncio.iscoroutinefunction(rag_pipeline):
                task = rag_pipeline(question)
            else:
                loop = asyncio.get_event_loop()
                task = loop.run_in_executor(None, rag_pipeline, question)
            tasks.append(task)

        # 批量执行
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                answers.append("")
                contexts_list.append([])
            else:
                answers.append(result.get("answer", ""))
                contexts_list.append(result.get("contexts", []))

        dataset = self.prepare_dataset(questions, answers, contexts_list, ground_truths)
        return self.evaluate(dataset)

    def generate_report(
        self,
        result_df: pd.DataFrame,
        test_data: Optional[List[Dict]] = None
    ) -> Dict[str, str]:
        """
        生成评估报告

        Returns:
            生成的文件路径字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_paths = {}

        # 1. CSV 详细结果
        csv_path = os.path.join(
            self.config.output_dir,
            f"{self.config.project_name}_{timestamp}_results.csv"
        )
        result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        report_paths["csv"] = csv_path

        # 2. JSON 详细结果
        json_path = os.path.join(
            self.config.output_dir,
            f"{self.config.project_name}_{timestamp}_results.json"
        )
        result_df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        report_paths["json"] = json_path

        # 3. 统计摘要
        summary = self._calculate_summary(result_df)
        summary_path = os.path.join(
            self.config.output_dir,
            f"{self.config.project_name}_{timestamp}_summary.json"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        report_paths["summary"] = summary_path

        # 4. HTML 可视化报告
        html_path = self._generate_html_report(result_df, summary, timestamp)
        report_paths["html"] = html_path

        # 打印摘要
        self._print_summary(summary)

        return report_paths

    def _calculate_summary(self, result_df: pd.DataFrame) -> Dict:
        """计算统计摘要"""
        # 获取所有指标列（排除非指标列）
        exclude_cols = ["question", "answer", "contexts", "ground_truth"]
        metric_cols = [c for c in result_df.columns if c not in exclude_cols]

        summary = {
            "project_name": self.config.project_name,
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(result_df),
            "metrics": {}
        }

        for col in metric_cols:
            if result_df[col].dtype in ["float64", "int64"]:
                summary["metrics"][col] = {
                    "mean": float(result_df[col].mean()),
                    "median": float(result_df[col].median()),
                    "std": float(result_df[col].std()),
                    "min": float(result_df[col].min()),
                    "max": float(result_df[col].max()),
                    "p25": float(result_df[col].quantile(0.25)),
                    "p75": float(result_df[col].quantile(0.75)),
                }

        return summary

    def _generate_html_report(
            self,
            result_df: pd.DataFrame,
            summary: Dict,
            timestamp: str
    ) -> str:
        """生成 HTML 可视化报告"""

        def get_score_class(score):
            if score >= 0.8:
                return "excellent"
            elif score >= 0.6:
                return "good"
            elif score >= 0.4:
                return "fair"
            else:
                return "poor"

        # 获取指标列（只选择数值列）
        exclude_cols = ["user_input", "response", "retrieved_contexts"]
        metric_cols = []
        for col in result_df.columns:
            if col not in exclude_cols:
                # 检查列是否为数值类型
                if result_df[col].dtype in ["float64", "int64"]:
                    metric_cols.append(col)

        # 评估维度说明数据
        metric_descriptions = {
            "context_recall": {
                "zh_name": "上下文召回率",
                "description": "评估检索到的上下文是否包含回答问题所需的全部关键信息。",
                "calculation": "将标准答案分解为关键事实，检查每个事实是否能在检索上下文中找到对应，计算覆盖比例。"
            },
            "context_precision": {
                "zh_name": "上下文精确率",
                "description": "评估检索到的上下文中与问题相关的信息占比，衡量检索质量。",
                "calculation": "检查上下文中的每个片段对回答问题的有用性，计算相关片段占总片段的比例。"
            },
            "context_entity_recall": {
                "zh_name": "上下文实体召回率",
                "description": "评估检索到的上下文中关键实体的覆盖程度，衡量是否包含回答问题所需的所有核心实体（如人名、术语、事件名等）",
                "calculation": "检索到的关键实体数量 / 真实答案中所有必要关键实体数量"
            },
            "noise_sensitivity(mode=relevant)": {
                "zh_name": "噪声敏感度",
                "description": "衡量系统检索到的上下文中包含无关信息（噪声）的比例，评估系统过滤无关内容的能力。分数越低表示系统越能有效过滤噪声",
                "calculation": "检索到的无关段落数量 / 检索到的总段落数量"
            },
            "faithfulness": {
                "zh_name": "忠实度",
                "description": "评估生成的回答是否忠实于检索到的上下文信息，检查回答中的事实是否能在上下文中找到依据。",
                "calculation": "通过提取回答中的关键陈述，验证每个陈述是否能被上下文支持，计算被支持的陈述比例。"
            },
            "response_relevancy": {
                "zh_name": "回答相关性",
                "description": "评估回答与问题的相关程度，检测回答中是否包含无关或冗余信息。",
                "calculation": "基于问题生成伪问题，计算这些伪问题与原始问题的相似度，衡量回答的针对性。"
            },
            "answer_relevancy": {
                "zh_name": "回答相关性",
                "description": "评估回答与问题的相关程度，检测回答中是否包含无关或冗余信息。",
                "calculation": "基于问题生成伪问题，计算这些伪问题与原始问题的相似度，衡量回答的针对性。"
            },
            "answer_similarity": {
                "zh_name": "回答相似度",
                "description": "评估生成的回答与标准答案在语义上的相似程度。",
                "calculation": "使用嵌入模型将回答和标准答案转换为向量，计算两者之间的余弦相似度。"
            },
            "answer_correctness": {
                "zh_name": "回答正确性",
                "description": "综合评估回答的准确性和语义相似性，衡量回答与标准答案的一致程度。",
                "calculation": "结合语义相似度和事实正确性，通过对比回答与标准答案的关键事实进行加权评分。"
            }
        }

        # 生成指标卡片HTML
        metric_cards = ""
        for metric_name in metric_cols:
            if metric_name in summary.get("metrics", {}):
                score = summary["metrics"][metric_name].get("mean", 0)
                score_class = get_score_class(score)
                metric_cards += f"""
                <div class="metric-card {score_class}">
                    <h3>{metric_name.replace('_', ' ').title()}</h3>
                    <h4>{metric_descriptions.get(metric_name, {}).get('zh_name', '')}</h4>
                    <div class="score">{score:.3f}</div>
                    <div class="score-bar">
                        <div class="score-fill" style="width: {score * 100}%"></div>
                    </div>
                </div>
                """

        # 生成样本表格行
        table_rows = ""
        for i, (_, row) in enumerate(result_df.head(n=100).iterrows(), 1):
            row_html = "<tr>"
            row_html += f"<td>{i}</td>"

            # 问题列 - 使用 title 属性显示完整内容
            question_full = str(row.get('user_input', ''))
            question_display = question_full[:50] + '...' if len(question_full) > 50 else question_full
            row_html += f"<td class='truncate' title='{question_full.replace(chr(39), chr(39) + chr(39))}'>{question_display}</td>"

            # 回答列 - 使用 title 属性显示完整内容
            answer_full = str(row.get('response', ''))
            answer_display = answer_full[:50] + '...' if len(answer_full) > 50 else answer_full
            row_html += f"<td class='truncate' title='{answer_full.replace(chr(39), chr(39) + chr(39))}'>{answer_display}</td>"

            # 指标列（安全格式化）
            for col in metric_cols:
                val = row.get(col, 0)
                try:
                    # 尝试转换为浮点数并格式化
                    val_float = float(val)
                    row_html += f"<td>{val_float:.3f}</td>"
                except (ValueError, TypeError):
                    # 转换失败则直接显示字符串
                    row_html += f"<td>{val}</td>"

            row_html += "</tr>"
            table_rows += row_html

        # 生成评估维度说明HTML
        metric_desc_html = ""
        for metric_key, desc in metric_descriptions.items():
            # 只显示当前报告中存在的指标
            if metric_key in metric_cols or any(metric_key in m for m in metric_cols):
                metric_name = metric_key.replace('_', ' ').title()
                zh_name = desc["zh_name"]
                description = desc["description"]
                calculation = desc["calculation"]
                metric_desc_html += f"""
                <div class="metric-desc-item">
                    <h4>{metric_name} <span class="zh-name">{zh_name}</span></h4>
                    <p class="desc-text"><strong>含义：</strong>{description}</p>
                    <p class="calc-text"><strong>计算方式：</strong>{calculation}</p>
                </div>
                """

        # HTML 模板（简化版）
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>RAGAS 评估报告 - {self.config.project_name}</title>
    <style>
        :root {{ --primary: #2563eb; --success: #10b981; --warning: #f59e0b; --danger: #ef4444; --text-secondary: #64748b; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; background: #f8fafc; line-height: 1.6; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        header {{ background: linear-gradient(135deg, var(--primary), #3b82f6); color: white; padding: 2rem; border-radius: 12px; margin-bottom: 2rem; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
        .metric-card {{ background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #e2e8f0; }}
        .metric-card.excellent {{ border-left-color: var(--success); }}
        .metric-card.good {{ border-left-color: #3b82f6; }}
        .metric-card.fair {{ border-left-color: var(--warning); }}
        .metric-card.poor {{ border-left-color: var(--danger); }}
        .metric-card .score {{ font-size: 2rem; font-weight: bold; margin: 0.5rem 0; }}
        .score-bar {{ height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden; }}
        .score-fill {{ height: 100%; background: var(--primary); border-radius: 4px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f8fafc; font-weight: 600; }}
        .truncate {{ max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: help; }}
        .truncate:hover {{ background-color: #f1f5f9; }}

        /* 评估维度说明样式 */
        .metrics-description {{ background: white; border-radius: 12px; padding: 2rem; margin-top: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .metrics-description h2 {{ margin-top: 0; color: #1e293b; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; margin-bottom: 1.5rem; }}
        .metric-desc-item {{ padding: 1.25rem; border-left: 4px solid var(--primary); background: #f8fafc; border-radius: 8px; margin-bottom: 1rem; }}
        .metric-desc-item:last-child {{ margin-bottom: 0; }}
        .metric-desc-item h4 {{ margin: 0 0 0.5rem 0; color: #1e293b; font-size: 1.1rem; }}
        .metric-desc-item .zh-name {{ color: var(--primary); font-weight: 600; margin-left: 0.5rem; }}
        .metric-desc-item .desc-text {{ margin: 0.25rem 0; color: #475569; }}
        .metric-desc-item .calc-text {{ margin: 0.25rem 0; color: var(--text-secondary); font-size: 0.9rem; }}
        .metric-desc-item strong {{ color: #334155; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 RAGAS 评估报告</h1>
            <p>项目: {self.config.project_name} | 样本数: {summary.get('total_samples', 0)} | 时间: {timestamp}</p>
        </header>

        <div class="metrics-grid">
            {metric_cards}
        </div>

        <div style="background: white; border-radius: 12px; padding: 2rem; margin-bottom: 2rem;">
            <h2>📋 详细评估样本</h2>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>问题</th>
                            <th>回答</th>
                            {''.join([f'''<th>{metric_descriptions.get(c, {}).get('zh_name', c.replace("_", " ").title())}</th>''' for c in metric_cols])}
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="metrics-description">
            <h2>📖 评估维度说明</h2>
            {metric_desc_html}
        </div>
    </div>
</body>
</html>"""

        html_path = os.path.join(
            self.config.output_dir,
            f"{self.config.project_name}_{timestamp}_report.html"
        )
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_path


    def _print_summary(self, summary: Dict):
        """打印评估摘要"""
        logging.info("\\n" + "="*70)
        logging.info(f"📊 RAGAS 评估报告: {summary['project_name']}")
        logging.info("="*70)
        logging.info(f"总样本数: {summary['total_samples']}")
        logging.info(f"评估时间: {summary['timestamp']}")
        logging.info("\\n📈 指标分数:")
        logging.info("-" * 50)

        for metric_name, stats in summary["metrics"].items():
            score = stats["mean"]
            if np.isnan(score):
                bar = ''
            else:
                bar = "█" * int(score * 20)
            logging.info(f"{metric_name:25s}: {score:.3f} {bar}")

        logging.info("="*70)


# ==================== 使用示例 ====================

def example_usage():

    # 1. 配置 RAGAS 评估器
    config = RAGASConfig(
        project_name="温宿_RAG评估",
        output_dir="./data/ragas_results",
        eval_llm_provider="deepseek",
        eval_llm_model="deepseek-chat",
        metrics=[
            "context_precision",
            "context_recall",
            "context_entity_recall",
            "noise_sensitivity",
            "faithfulness",
            "response_relevancy",
            "answer_correctness",
            "answer_similarity"
        ]
    )

    evaluator = RAGAS_Evaluator(config)

    # 2. 初始化你的 RAG 系统
    rag_config = RAG_Proc_Config()
    rag_processor = RAGProcessor(rag_config)
    rag_processor.load_vector_store()

    DEEPSEEK_API_KEY = rag_config.deepseek_api_key
    model = init_chat_model("deepseek:deepseek-chat", api_key=DEEPSEEK_API_KEY)

    # 3. 定义 RAG 管道
    def rag_pipeline(question: str):
        # 检索
        docs = rag_processor.ensemble_search_with_rerank(question, k=8)
        contexts = [doc.page_content for doc in docs]

        prompt = f"你是一个专业的知识库文案整理员，请你根据用户的提问：{question}，结合本地知识库检索到的内容{'；'.join(contexts)}，" \
                 f"生成1个高质量的答案" \
                 f"要求："  \
                 f"1. 答案要完整、准确，包含知识库检索内容中的关键信息"   \
                 f"2. 答案长度控制在100-200字"  \
                 f"3. 仅需要输出最终的答案，不要包含其他内容"
        response = model.invoke(prompt)
        answer = response.content
        logging.info(f"预置问题：{question}")
        logging.info(f"AI 回复：{response.content}")

        return {
            "answer": answer,
            "contexts": contexts
        }

    # 4. 加载测试数据
    data_path = './utils/RAG_Evaluation/test_data.json'
    with open(data_path, 'r', encoding='utf-8') as fp:
        data_json = json.loads(fp.read())
    test_data = []
    test_cases = data_json['test_cases']
    for d in test_cases:
        test_data.append({'question': d['question'], 'ground_truth': d['ground_truth']})
    # test_data = test_data[:2]

    # 5. 执行评估
    result_df = evaluator.evaluate_rag_pipeline(rag_pipeline, test_data)

    # 6. 生成报告
    report_paths = evaluator.generate_report(result_df, test_data)
    logging.info(f"\\n✅ 报告已生成: {report_paths}")


if __name__ == "__main__":
    example_usage()


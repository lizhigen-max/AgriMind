# RAGAS 官方指标详解

## 📊 一、检索指标 (Retriever Metrics)

### 1. ContextPrecision (上下文精确率)
**含义**：检索到的上下文中有多少是真正有用的

**计算方式**：
- 对每个检索到的上下文块，LLM 判断其是否对回答问题有用（0 或 1）
- 使用 Average Precision 计算：
  ```
  AP = Σ(P@i × v_i) / Σv_i
  其中 P@i 是前 i 个结果的精确率，v_i 是第 i 个结果是否有用
  ```

**直观理解**：
- 分数 0.8 = 80% 的检索结果是有用的
- 分数低 = 检索器返回了很多无关内容（噪声）

**示例**：
```
检索结果：[有用, 有用, 无用, 有用, 无用]
ContextPrecision = 0.6  (3/5 有用)
```

---

### 2. ContextRecall (上下文召回率)
**含义**：回答问题所需的所有信息，有多少被成功检索到了

**计算方式**：
1. 将标准答案(ground_truth)分解为多个事实陈述
2. 对每个陈述，判断是否能从检索到的上下文中推断出来
3. 召回率 = 被支持的陈述数 / 总陈述数

**直观理解**：
- 分数 0.8 = 80% 的所需信息被检索到了
- 分数低 = 检索器漏掉了关键信息

**示例**：
```
标准答案包含 5 个事实：A, B, C, D, E
检索上下文支持了：A, B, C
ContextRecall = 3/5 = 0.6
```

---

### 3. ContextEntityRecall (上下文实体召回率)
**含义**：标准答案中的关键实体（人名、地名、专有名词）有多少出现在检索结果中

**计算方式**：
```
实体召回率 = 检索结果中出现的实体数 / 标准答案中的总实体数
```

**适用场景**：
- 需要精确识别特定实体的问题
- 如"李志根有什么核心优势？"需要召回"李志根"这个实体

---

### 4. NoiseSensitivity (噪声敏感度)
**含义**：系统有多容易被无关信息误导而产生错误回答

**计算方式**：
```
NoiseSensitivity = 被无关信息误导的错误数 / 总错误数
```

**直观理解**：
- 分数 0.0 = 完美，从不被噪声误导
- 分数 0.5 = 一半的错误是因为看了无关内容
- 分数 1.0 = 所有错误都是被噪声误导的

**为什么重要**：
- 评估 RAG 系统的"抗干扰能力"
- 高噪声敏感度 = 需要改进重排序或过滤策略

---

## 📝 二、生成指标 (Generation Metrics)

### 5. Faithfulness (忠实度 / 事实一致性)
**含义**：生成的回答是否忠实于检索到的上下文，有没有"幻觉"

**计算方式**（两步 LLM 判断）：
1. **陈述提取**：LLM 将回答分解为原子陈述（简单事实句）
   ```
   回答："葡萄冬季修剪应在12月进行，主要方法有短截和回缩。"
   陈述1："葡萄冬季修剪应在12月进行"
   陈述2："葡萄冬季修剪方法有短截"
   陈述3："葡萄冬季修剪方法有回缩"
   ```

2. **验证**：对每个陈述，LLM 判断是否能从上下文中推断
   ```
   Faithfulness = 被支持的陈述数 / 总陈述数
   ```

**直观理解**：
- 分数 1.0 = 所有陈述都有上下文支持（无幻觉）
- 分数 0.7 = 30% 的陈述是编造的或无法验证

**与 AnswerCorrectness 的区别**：
- Faithfulness：回答 vs 检索上下文（防幻觉）
- AnswerCorrectness：回答 vs 标准答案（是否正确）

---

### 6. ResponseRelevancy (回答相关性)
**含义**：回答是否直接针对问题，有没有答非所问

**计算方式**：
1. 从回答中生成多个"可能的问题"
2. 计算这些生成的问题与原始问题的嵌入相似度
3. 取平均相似度作为分数

**直观理解**：
- 分数 0.9 = 回答非常切题
- 分数 0.3 = 回答离题万里

**示例**：
```
问题："葡萄冬季怎么修剪？"
回答："葡萄是一种美味的水果，营养丰富..."（不相关）
Score = 0.1

回答："冬季修剪应在落叶后2-3周进行..."（相关）
Score = 0.95
```

---

### 7. AnswerCorrectness (回答正确性)
**含义**：回答与标准答案相比，事实上的正确程度

**计算方式**（三步）：
1. **分解**：将回答和标准答案都分解为事实陈述
2. **匹配**：用 LLM 判断哪些陈述相互支持/矛盾
3. **计算**：
   ```
   Precision = 回答中正确的陈述数 / 回答总陈述数
   Recall = 标准答案中被回答覆盖的陈述数 / 标准答案总陈述数
   F1 = 2 × Precision × Recall / (Precision + Recall)
   ```

**直观理解**：
- 综合了精确率和召回率
- 分数 0.8 = 回答基本正确，可能有少量遗漏或错误

**与 Faithfulness 的区别**：
```
Faithfulness: 回答 ←→ 检索上下文（有没有胡说）
AnswerCorrectness: 回答 ←→ 标准答案（对不对）

场景1：上下文错误 → Faithfulness高，AnswerCorrectness低
场景2：上下文正确但回答不全 → Faithfulness高，AnswerCorrectness中
场景3：回答幻觉 → Faithfulness低，AnswerCorrectness低
```

---

### 8. AnswerSimilarity (回答相似度)
**含义**：回答与标准答案在语义上的相似程度

**计算方式**：
```
CosineSimilarity(嵌入(回答), 嵌入(标准答案))
```

**与 AnswerCorrectness 的区别**：
- AnswerSimilarity：纯语义相似，不管事实对错
- AnswerCorrectness：基于事实分解和验证

**适用场景**：
- 需要评估回答"像不像"标准答案，而不严格验证事实

---

## 🎯 三、综合指标

### 9. AspectCritic (方面批评)
**含义**：让 LLM 从特定维度评价回答质量

**使用方式**：
```python
from ragas.metrics import AspectCritic

# 定义评价维度
correctness = AspectCritic(
    name="correctness",
    definition="回答是否事实正确？",
    strictness=3  # 严格程度 1-5
)

helpfulness = AspectCritic(
    name="helpfulness", 
    definition="回答对用户是否有帮助？"
)
```

**输出**：0-1 分数 + 文字评价

---

## 📈 四、指标选择建议

### 最小可用组合（快速验证）
```python
metrics = [
    Faithfulness(),      # 防幻觉（最重要）
    AnswerCorrectness(), # 正确性
]
```

### 标准组合（推荐）
```python
metrics = [
    ContextPrecision(),    # 检索质量
    ContextRecall(),       # 检索覆盖
    Faithfulness(),        # 生成质量
    AnswerCorrectness(),   # 正确性
]
```

### 完整组合（深度评估）
```python
metrics = [
    ContextPrecision(),
    ContextRecall(),
    ContextEntityRecall(),
    NoiseSensitivity(),
    Faithfulness(),
    ResponseRelevancy(),
    AnswerCorrectness(),
]
```

### 不同场景的重点指标

| 场景 | 重点指标 | 为什么 |
|------|---------|--------|
| **客服问答** | Faithfulness, AnswerCorrectness | 必须准确，不能胡说 |
| **知识库搜索** | ContextRecall, ContextPrecision | 信息要全面且精确 |
| **创意写作** | ResponseRelevancy | 回答要切题 |
| **医学/法律** | Faithfulness, NoiseSensitivity | 绝对不能被噪声误导 |

---

## 🔧 五、分数解读指南

### 分数范围
- **0.0 - 0.4**：差，需要重大改进
- **0.4 - 0.6**：一般，有明显问题
- **0.6 - 0.8**：良好，可以优化
- **0.8 - 1.0**：优秀，接近完美

### 常见分数组合及含义

| Faithfulness | AnswerCorrectness | 含义 |
|-------------|-------------------|------|
| 高 | 高 | 完美：无幻觉且正确 |
| 高 | 低 | 检索上下文有问题，回答忠实于错误信息 |
| 低 | 高 | 回答正确但靠猜，不是基于上下文 |
| 低 | 低 | 严重问题：既幻觉又错误 |

| ContextPrecision | ContextRecall | 含义 |
|------------------|---------------|------|
| 高 | 高 | 检索完美：精确且全面 |
| 高 | 低 | 检索精确但遗漏信息 |
| 低 | 高 | 检索全面但有很多噪声 |
| 低 | 低 | 检索系统需要大修 |

---

## 💡 六、实际使用示例

```python
from ragas import evaluate
from ragas.metrics import (
    ContextPrecision, ContextRecall,
    Faithfulness, AnswerCorrectness
)
from datasets import Dataset

# 准备数据
dataset = Dataset.from_dict({
    "question": ["葡萄冬季怎么修剪？"],
    "answer": ["应在落叶后2-3周进行..."],
    "contexts": [["上下文1...", "上下文2..."]],
    "ground_truth": ["标准答案..."]
})

# 执行评估
result = evaluate(
    dataset=dataset,
    metrics=[
        ContextPrecision(),
        ContextRecall(),
        Faithfulness(),
        AnswerCorrectness()
    ]
)

# 结果解读
print(result)
# {
#   'context_precision': 0.85,  # 85%检索结果有用
#   'context_recall': 0.70,     # 只找到70%所需信息
#   'faithfulness': 0.90,       # 回答基本无幻觉
#   'answer_correctness': 0.75  # 回答基本正确但有遗漏
# }

# 诊断：检索召回不足，需要优化检索策略
```

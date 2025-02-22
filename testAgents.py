import pandas as pd
import re
from typing import Optional, Dict, List, Tuple
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 简化后的状态模型，只保留评估必需的字段
class TranslationState(BaseModel):
    original_text: str = Field(..., description="原始文本")
    LLM_translation: str = Field(default="", description="大模型翻译")
    google_translation: str = Field(default="", description="谷歌翻译")
    evaluation_score: Optional[Dict[str, float]] = Field(default=None, description="评估分数")
    evaluation_feedback: Optional[str] = Field(default=None, description="评估反馈")
    LLM_score: Optional[float] = Field(default=None, description="大模型翻译分数")
    google_score: Optional[float] = Field(default=None, description="谷歌翻译分数")
    better_translation: Optional[str] = Field(default=None, description="更好的翻译版本")

    class Config:
        arbitrary_types_allowed = True

# 初始化 LLM
llm = ChatOpenAI(
    temperature=0.2, 
    model_name="gpt-4o-mini",
    base_url="https://api.tu-zi.com/v1",
    api_key="sk-UdXOiiUE11esG1frq6OOTWMghQEON0DTgRs9oMPvyDzTf5Vj"
)

# 修改评估提示词
evaluate_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一位专业的中英互译评估专家。请基于以下多个维度对比评估两个翻译版本：

评估维度：
1. 准确性（40分）：翻译是否准确传达原文含义
2. 流畅性（30分）：译文是否通顺自然
3. 原意保留（30分）：是否保留了原文的语气和细节

评估要求：
1. 请独立评估每个翻译，不要互相影响
2. 重点关注翻译的准确性和专业性
3. 注意发现每个翻译的优点
4. 保持客观中立的评价态度

输出格式：
大模型翻译：[总分]分（准确性[分数]，流畅性[分数]，原意保留[分数]）
谷歌翻译：[总分]分（准确性[分数]，流畅性[分数]，原意保留[分数]）
更好的翻译：[选择]
原因：[2-3句话说明各自优劣，需要具体指出用词或句式的例子]"""),
])
evaluator = evaluate_prompt | llm

def parse_evaluation(evaluation_text: str) -> Tuple[float, float, str]:
    """解析评估文本，提取分数和结论"""
    # 提取分数（查找形如"大模型翻译：85分"和"谷歌翻译：90分"的模式）
    llm_score_match = re.search(r'大模型翻译：(\d+)分', evaluation_text)
    google_score_match = re.search(r'谷歌翻译：(\d+)分', evaluation_text)
    
    llm_score = float(llm_score_match.group(1)) if llm_score_match else 0
    google_score = float(google_score_match.group(1)) if google_score_match else 0
    
    # 判断哪个更好（查找明确的结论陈述）
    if "大模型翻译" in evaluation_text.split("更好的翻译：")[1].split("\n")[0]:
        better = "大模型翻译"
    else:
        better = "谷歌翻译"
    
    return llm_score, google_score, better

def evaluate_translations(state: TranslationState) -> TranslationState:
    """评估翻译质量并返回更新后的状态"""
    # 构建评估上下文
    evaluation_context = {
        "original_text": state.original_text,
        "LLM_translation": state.LLM_translation,
        "google_translation": state.google_translation
    }
    
    result = evaluator.invoke(evaluation_context)
    evaluation_text = result.content
    
    # 解析评估结果
    llm_score, google_score, better = parse_evaluation(evaluation_text)
    
    # 更新状态
    state.evaluation_feedback = evaluation_text
    state.LLM_score = llm_score
    state.google_score = google_score
    state.better_translation = state.LLM_translation if better == "大模型翻译" else state.google_translation
    
    return state

def process_excel_batch(evaluated_prompt) -> list:
    """批量处理Excel文件中的翻译评估"""
    df = pd.read_excel('./out/test.xlsx')
    results = []
    
    # 获取指令文本
    prompt_text = """
                    请将以下中文文本直译成英文。
                    要求：
                    1. 严格按照原文的字面意思进行翻译
                    2. 保持原文的语序结构
                    3. 不要添加任何解释或意译
                    4. 确保每个词语都得到准确对应的翻译
                  """

    for _, row in df.iterrows():
        state = TranslationState(
            original_text=row['Chinese'],
            LLM_translation=row['English'],
            google_translation=row['Google English']
        )
        
        evaluated_state = evaluate_translations(state)
        
        results.append({
            'Prompt': prompt_text,  # 只存储指令文本
            'Original': state.original_text,
            'LLM Translation': state.LLM_translation,
            'Google Translation': state.google_translation,
            'Evaluation': state.evaluation_feedback,
        })

    results_df = pd.DataFrame(results)
    results_df.to_excel('./out/test_evaluation_results.xlsx', index=False)
    return results_df

if __name__ == "__main__":
    # 批量评估
    results_df = process_excel_batch(evaluate_prompt)
    
    # 打印统计摘要
    print("\n=== 评估统计摘要 ===")
    print(f"总评估数量: {len(results_df)}")
    print("\n评估完成，结果已保存到 ./out/test_evaluation_results.xlsx")
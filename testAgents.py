import pandas as pd
import re
from typing import Optional, Dict, List, Tuple
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 简化后的状态模型，只保留评估必需的字段
class TranslationState(BaseModel):
    original_text: str = Field(..., description="原始文本")
    human_translation: str = Field(default="", description="人工翻译")
    google_translation: str = Field(default="", description="谷歌翻译")
    evaluation_score: Optional[Dict[str, float]] = Field(default=None, description="评估分数")
    evaluation_feedback: Optional[str] = Field(default=None, description="评估反馈")
    human_score: Optional[float] = Field(default=None, description="人工翻译分数")
    google_score: Optional[float] = Field(default=None, description="谷歌翻译分数")
    better_translation: Optional[str] = Field(default=None, description="更好的翻译版本")

    class Config:
        arbitrary_types_allowed = True

# 初始化 LLM
llm = ChatOpenAI(
    temperature=0.2, 
    model_name="gpt-3.5-turbo-1106",
    base_url="https://api.chatanywhere.tech/v1",
    api_key="sk-AtjQFIBglEgUp1UQoKl4qIXp18UKRqXCQRMp815RUlviQH4X"
)

# 评估智能体
evaluate_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一位翻译评估专家。请对比以下两个英语翻译并给出简洁的评估。
    
    评估格式要求：
    1. 首先给出两个翻译的分数（0-100分）
    2. 说明哪个翻译更好
    3. 用3-4句话解释原因，并指出具体哪个单词或词句更好
    
    输出格式示例：
    LLM翻译85分，谷歌翻译90分，谷歌翻译更好。原因：谷歌翻译语言更自然流畅，句子结构更符合英语表达习惯。例如，谷歌翻译中的“example phrase”比LLM翻译中的“sample phrase”更准确。"""),
])
evaluator = evaluate_prompt | llm

def parse_evaluation(evaluation_text: str) -> Tuple[float, float, str]:
    """解析评估文本，提取分数和结论"""
    # 提取分数
    scores = re.findall(r'(\d+)分', evaluation_text)
    human_score = float(scores[0]) if len(scores) > 0 else 0
    google_score = float(scores[1]) if len(scores) > 1 else 0
    
    # 判断哪个更好
    better = "大模型翻译" if "大模型翻译更好" in evaluation_text else "谷歌翻译"
    
    return human_score, google_score, better

def evaluate_translations(state: TranslationState) -> TranslationState:
    """评估翻译质量并返回更新后的状态"""
    result = evaluator.invoke(state.model_dump())
    evaluation_text = result.content
    
    # 解析评估结果
    human_score, google_score, better = parse_evaluation(evaluation_text)
    
    # 更新状态
    state.evaluation_feedback = evaluation_text
    state.human_score = human_score
    state.google_score = google_score
    state.better_translation = better
    
    return state

def process_excel_batch(evaluated_prompt) -> list:
    """批量处理Excel文件中的翻译评估"""
    df = pd.read_excel('./out/translations_test_100_google.xlsx')
    results = []
    
    # 获取指令文本
    prompt_text = """
                    你是一位专业的中英互译专家。请将以下中文文本直译成英文。
                    要求：
                    1. 严格按照原文的字面意思进行翻译
                    2. 保持原文的语序结构
                    3. 不要添加任何解释或意译
                    4. 确保每个词语都得到准确对应的翻译
                  """

    for _, row in df.iterrows():
        state = TranslationState(
            original_text=row['Chinese'],
            human_translation=row['English'],
            google_translation=row['Google English']
        )
        
        evaluated_state = evaluate_translations(state)
        
        results.append({
            'Prompt': prompt_text,  # 只存储指令文本
            'Original': state.original_text,
            'LLM Translation': state.human_translation,
            'Google Translation': state.google_translation,
            'Evaluation': state.evaluation_feedback,
        })

    results_df = pd.DataFrame(results)
    results_df.to_excel('./out/evaluation_results_100.xlsx', index=False)
    return results_df

if __name__ == "__main__":
    # 批量评估
    results_df = process_excel_batch(evaluate_prompt)
    
    # 打印统计摘要
    print("\n=== 评估统计摘要 ===")
    print(f"总评估数量: {len(results_df)}")
    print("\n评估完成，结果已保存到 evaluation_results.xlsx")
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

原文：{original_text}

大模型翻译：{LLM_translation}
谷歌翻译：{google_translation}

评估维度：
1. 准确性：翻译是否准确传达原文含义
2. 流畅性：译文是否通顺自然
3. 原意保留：是否保留了原文的语气和细节

评估要求:     
1. 请独立评估每个翻译，不要互相影响
2. 重点关注翻译的准确性和专业性
3. 注意发现每个翻译的优点
4. 保持客观中立的评价态度
5. 两个翻译优秀程度尽量可以做到6：4的比例

输出格式：
更好的翻译：[选择]
大模型翻译：[总分]分
谷歌翻译：[总分]分
原因：[2-3句话说明各自优劣，需要具体指出用词或句式的例子]""")
])

# 添加evaluator定义
evaluator = evaluate_prompt | llm

def parse_evaluation(evaluation_text: str) -> Tuple[float, float, str]:
    """解析评估文本，提取分数和结论"""
    try:
        # 提取分数（查找形如"大模型翻译：85分"和"谷歌翻译：90分"的模式）
        llm_score_match = re.search(r'大模型翻译：(\d+)分', evaluation_text)
        google_score_match = re.search(r'谷歌翻译：(\d+)分', evaluation_text)
        
        llm_score = float(llm_score_match.group(1)) if llm_score_match else 0
        google_score = float(google_score_match.group(1)) if google_score_match else 0
        
        # 更安全地判断哪个更好
        better_parts = evaluation_text.split("更好的翻译：")
        if len(better_parts) > 1:
            conclusion = better_parts[1].split("\n")[0].strip()
            if "大模型" in conclusion:
                better = "大模型翻译"
            else:
                better = "谷歌翻译"
        else:
            # 如果没有明确结论，根据分数决定
            better = "大模型翻译" if llm_score >= google_score else "谷歌翻译"
        
        return llm_score, google_score, better
        
    except Exception as e:
        print(f"解析评估结果时出错: {str(e)}")
        print(f"评估文本: {evaluation_text}")
        # 返回默认值
        return 0, 0, "谷歌翻译"

def evaluate_translations(state: TranslationState) -> TranslationState:
    """评估翻译质量并返回更新后的状态"""
    evaluation_context = {
        "original_text": state.original_text,
        "LLM_translation": state.LLM_translation,
        "google_translation": state.google_translation
    }
    
    # 确保所有必需的字段都有值
    if not all(evaluation_context.values()):
        print("警告：存在空的翻译字段")
        print(evaluation_context)
        return state
        
    try:
        result = evaluator.invoke(evaluation_context)
        evaluation_text = result.content
        
        # 解析评估结果
        llm_score, google_score, better = parse_evaluation(evaluation_text)
        
        # 更新状态
        state.evaluation_feedback = evaluation_text
        state.LLM_score = llm_score
        state.google_score = google_score
        state.better_translation = state.LLM_translation if better == "大模型翻译" else state.google_translation
        
    except Exception as e:
        print(f"评估过程发生错误: {str(e)}")
        state.evaluation_feedback = f"评估失败: {str(e)}"
    
    return state

def process_excel_batch(evaluated_prompt) -> list:
    """批量处理Excel文件中的翻译评估"""
    try:
        df = pd.read_excel('./out/translations_literature_1500_google_2_test.xlsx')
        if df.empty:
            raise ValueError("Excel文件为空")
            
        print(f"成功读取Excel文件，共{len(df)}行数据")
        print("列名:", df.columns.tolist())
        
        results = []
        for idx, row in df.iterrows():
            print(f"\n处理第{idx+1}条数据:")
            print(f"中文原文: {row['Chinese']}")
            print(f"LLM翻译: {row['English']}")
            print(f"谷歌翻译: {row['Google English']}")
            
            state = TranslationState(
                original_text=row['Chinese'],
                LLM_translation=row['English'],
                google_translation=row['Google English']
            )
            
            evaluated_state = evaluate_translations(state)
            results.append({
                'Original': state.original_text,
                'LLM Translation': state.LLM_translation,
                'Google Translation': state.google_translation,
                'Evaluation': evaluated_state.evaluation_feedback,
                'LLM Score': evaluated_state.LLM_score,
                'Google Score': evaluated_state.google_score,
                'Better Translation': evaluated_state.better_translation
            })
            
        results_df = pd.DataFrame(results)
        results_df.to_excel('./out/literature_evaluation_results.xlsx', index=False)
        return results_df
        
    except Exception as e:
        print(f"处理过程发生错误: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    # 批量评估
    results_df = process_excel_batch(evaluate_prompt)
    
    # 打印统计摘要
    print("\n=== 评估统计摘要 ===")
    print(f"总评估数量: {len(results_df)}")
    print("\n评估完成，结果已保存到 ./out/literature_evaluation_results.xlsx")
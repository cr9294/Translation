import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
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
    model_name="gpt-3.5-turbo-0125",
    base_url="https://free.v36.cm/v1",
    api_key="sk-JYEqiFja0N5GaUWYD1Ea425f85Bd4e459b195f616957919b"
)

# 评估智能体
evaluate_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一位翻译评估专家。请对比以下两个英语翻译并给出简洁的评估。
    
    评估格式要求：
    1. 首先给出两个翻译的分数（0-100分）
    2. 说明哪个翻译更好
    3. 用3-4句话解释原因
    
    输出格式示例：
    LLM翻译85分，谷歌翻译90分，谷歌翻译更好。原因：谷歌翻译语言更自然流畅，句子结构更符合英语表达习惯。因为："""),
])
evaluator = evaluate_prompt | llm

def parse_evaluation(evaluation_text: str) -> Tuple[float, float, str]:
    """解析评估文本，提取分数和结论"""
    # 提取分数
    scores = re.findall(r'(\d+)分', evaluation_text)
    human_score = float(scores[0]) if len(scores) > 0 else 0
    google_score = float(scores[1]) if len(scores) > 1 else 0
    
    # 判断哪个更好
    better = "人工翻译" if "人工翻译更好" in evaluation_text else "谷歌翻译"
    
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

def process_excel_batch() -> list:
    """批量处理Excel文件中的翻译评估"""
    df = pd.read_excel('translations_result_google_test.xlsx')
    results = []

    for _, row in df.iterrows():
        state = TranslationState(
            original_text=row['Chinese'],
            human_translation=row['English'],
            google_translation=row['Google English']
        )
        
        evaluated_state = evaluate_translations(state)
        
        results.append({
            'Original': state.original_text,
            'Human Translation': state.human_translation,
            'Google Translation': state.google_translation,
            'Evaluation': state.evaluation_feedback,
        })

    results_df = pd.DataFrame(results)
    results_df.to_excel('evaluation_results.xlsx', index=False)
    return results_df

def visualize_results(df: pd.DataFrame):
    """生成评估结果的可视化图表"""
    plt.style.use('seaborn')
    
    # 创建图形
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 分数对比箱型图
    plt.subplot(2, 2, 1)
    scores_data = pd.DataFrame({
        'Human': df['Human Score'],
        'Google': df['Google Score']
    })
    sns.boxplot(data=scores_data)
    plt.title('翻译分数分布对比')
    plt.ylabel('分数')
    
    # 2. 更好翻译来源统计
    plt.subplot(2, 2, 2)
    better_counts = df['Better Translation'].value_counts()
    plt.pie(better_counts, labels=better_counts.index, autopct='%1.1f%%')
    plt.title('更好翻译来源占比')
    
    # 3. 分数差异直方图
    plt.subplot(2, 2, 3)
    score_diff = df['Human Score'] - df['Google Score']
    sns.histplot(score_diff, bins=20)
    plt.title('人工译文与谷歌译文分数差异分布')
    plt.xlabel('分数差异 (人工 - 谷歌)')
    
    # 4. 分数相关性散点图
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='Human Score', y='Google Score')
    plt.title('人工译文与谷歌译文分数相关性')
    
    plt.tight_layout()
    plt.savefig('translation_evaluation_analysis.png')
    plt.close()

if __name__ == "__main__":
    # 批量评估并生成可视化
    results_df = process_excel_batch()
    visualize_results(results_df)
    
    # 打印统计摘要
    print("\n=== 评估统计摘要 ===")
    print(f"总评估数量: {len(results_df)}")
    print("\n平均分数:")
    print(f"人工翻译: {results_df['Human Score'].mean():.2f}")
    print(f"谷歌翻译: {results_df['Google Score'].mean():.2f}")
    print(f"\n更好的翻译来源统计:\n{results_df['Better Translation'].value_counts()}")
    print("\n评估完成，结果已保存到 evaluation_results.xlsx")
    print("可视化结果已保存到 translation_evaluation_analysis.png")
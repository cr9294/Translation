import pandas as pd
from typing import Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

# 定义状态模型
class TranslationState(BaseModel):
    text: str = Field(description="要翻译的文本")
    translation: Optional[str] = Field(default=None, description="翻译结果")
    
# 定义智能体
llm = ChatOpenAI(
    temperature=0.2, 
    model_name="gpt-3.5-turbo-0125",
    base_url="https://free.v36.cm/v1",
    api_key="sk-JYEqiFja0N5GaUWYD1Ea425f85Bd4e459b195f616957919b"
)

# 翻译指令模板
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一位专业的中英互译专家。请将以下中文文本直译成英文。
    要求：
    1. 严格按照原文的字面意思进行翻译
    2. 保持原文的语序结构
    3. 不要添加任何解释或意译
    4. 确保每个词语都得到准确对应的翻译
    """),
    ("human", "{text}"),
])

# 意译模板（暂时注释）
"""
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", '''你是一位专业的中英互译专家。请将以下中文文本意译成英文。
    要求：
    1. 使用地道的英语表达方式重组句子
    2. 可以调整语序，确保表达自然流畅
    3. 可以适当调整词语，只要保持原意
    4. 注重表达效果，确保英文读者容易理解
    '''),
    ("human", "{text}"),
])
"""

# 专业词汇翻译模板（暂时注释）
"""
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", '''你是一位专业的中英互译专家。请将以下中文文本翻译成英文，特别注意专业术语的处理。
    要求：
    1. 准确识别并保留专业术语和技术词汇
    2. 对专业术语使用行业通用的标准英文翻译
    3. 在保持专业性的同时确保句子通顺
    4. 如遇到新兴技术词汇，优先使用业内认可的英文表达
    5. 可以在翻译时加入常用缩写(如 AI, ML, DL 等)
    '''),
    ("human", "{text}"),
])
"""

# 定义翻译函数
def translate_text(state: TranslationState) -> TranslationState:
    translation_response = translator.invoke({"text": state.text})
    state.translation = translation_response.content
    return state

# 创建工作流
workflow = StateGraph(TranslationState)
workflow.add_node("translate", translate_text)
workflow.set_entry_point("translate")
chain = workflow.compile()

# 读取Excel文件并处理
def process_translations(file_path: str):
    df = pd.read_excel(file_path)
    results = []
    
    for text in df['Chinese']:
        initial_state = TranslationState(text=text)
        # chain.invoke 返回的是字典形式，需要用字典方式访问
        result = chain.invoke(initial_state)
        results.append(result["translation"])  # 使用字典索引而不是属性访问
    
    df['English'] = results
    filename = "./out/translations_test_10.xlsx"
    df.to_excel(filename, index=False)
    print(f"翻译完成，结果已保存到 {filename}")

translator = translate_prompt | llm

if __name__ == "__main__":
    process_translations("./out/test_10.xlsx")  # 替换为实际的Excel文件路径
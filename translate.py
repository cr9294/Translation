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
    model_name="gpt-4o-mini",
    base_url="https://api.tu-zi.com/v1",
    api_key="sk-PEvjpQELoLDQqNX75f8c457d2cC244E4846a4c13C497Fd73"
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
    df.to_excel('translations_result.xlsx', index=False)
    print("翻译完成，结果已保存到 translations_result.xlsx")

translator = translate_prompt | llm

if __name__ == "__main__":
    process_translations("filtered_contents.xlsx")  # 替换为实际的Excel文件路径
import pandas as pd
from typing import Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

# 1. 使用 Pydantic BaseModel 定义状态
class TranslationState(BaseModel):
    original_text: str = Field(..., description="原始文本")
    human_translation: str = Field(default="", description="人工翻译")  # 添加默认值
    google_translation: str = Field(default="", description="谷歌翻译")  # 添加默认值
    source_language: str = Field(default="Chinese", description="源语言")
    target_language: str = Field(default="English", description="目标语言")
    evaluation_score: Optional[Dict[str, float]] = Field(default=None, description="评估分数")
    evaluation_feedback: Optional[str] = Field(default=None, description="评估反馈")
    refined_translation: Optional[str] = Field(default=None, description="优化后的翻译")
    final_translation: Optional[str] = Field(default=None, description="最终翻译结果")

    class Config:
        arbitrary_types_allowed = True

# 2. 定义智能体
llm = ChatOpenAI(
    temperature=0.2, 
    model_name="gpt-4o-mini",
    base_url="https://api.tu-zi.com/v1",
    api_key="sk-PEvjpQELoLDQqNX75f8c457d2cC244E4846a4c13C497Fd73"
)

# 翻译智能体
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional translator specializing in translating {source_language} to {target_language}. Please provide a high-quality translation of the following text."),
    ("human", "{original_text}"),
])
translator = translate_prompt | llm

# 修改评估智能体
evaluate_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一位翻译评估专家。请对比以下两个英语翻译并给出简洁的评估。
    
    评估格式要求：
    1. 首先给出两个翻译的分数（0-100分）
    2. 说明哪个翻译更好
    3. 用1-2句话解释原因
    4. 总字数限制在100字以内
    
    输出格式示例：
    LLM翻译85分，谷歌翻译90分，谷歌翻译更好。原因：谷歌翻译语言更自然流畅，句子结构更符合英语表达习惯。"""),
    ("human", """原文：{original_text}
人工翻译：{human_translation}
谷歌翻译：{google_translation}"""),
])
evaluator = evaluate_prompt | llm

# 修改润色智能体的提示模板
refine_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional translator tasked with refining an English translation based on the provided evaluation."),
    ("human", """Please improve this translation:
Original Text: {original_text}
Current Translation: {translation}
Evaluation Feedback: {evaluation}

Please provide an improved translation that addresses the feedback.""")
])
refiner = refine_prompt | llm

# 3. 更新 LangGraph 节点函数
# 修改 initialize_state 函数
def initialize_state(state: dict) -> dict:
    """初始化状态，确保所有必要字段都存在"""
    if isinstance(state, dict):
        return {
            "original_text": state.get("original_text", ""),
            "human_translation": state.get("human_translation", ""),
            "google_translation": state.get("google_translation", ""),
            "source_language": state.get("source_language", "Chinese"),
            "target_language": state.get("target_language", "English")
        }
    else:
        # 如果输入是 TranslationState 对象，直接返回其字典表示
        return state.model_dump()

# 修改 translate 函数
def translate(state: dict) -> dict:
    try:
        current_state = TranslationState.model_validate(state)
        translated_result = translator.invoke({
            "source_language": current_state.source_language,
            "target_language": current_state.target_language,
            "original_text": current_state.original_text
        })
        return {
            "human_translation": translated_result.content,
            "google_translation": translated_result.content
        }
    except Exception as e:
        print(f"Translation error: {e}")
        return state

def evaluate(state: dict) -> dict:
    current_state = TranslationState.model_validate(state)
    result = evaluator.invoke(current_state.model_dump())
    return {"evaluation_feedback": result.content}

# 修改润色函数
def refine(state: dict) -> dict:
    current_state = TranslationState.model_validate(state)
    
    # 准备润色所需的输入参数
    refinement_input = {
        "original_text": current_state.original_text,
        "translation": current_state.human_translation,  # 使用人工翻译作为基础
        "evaluation": current_state.evaluation_feedback  # 使用评估反馈
    }
    
    try:
        result = refiner.invoke(refinement_input)
        return {"refined_translation": result.content}
    except Exception as e:
        print(f"Refinement error: {e}")
        return {"refined_translation": current_state.human_translation}

def should_refine(state: dict) -> str:
    current_state = TranslationState.model_validate(state)
    if current_state.evaluation_feedback and any(
        keyword in current_state.evaluation_feedback.lower() 
        for keyword in ["improve", "建议"]
    ):
        return "refine"
    return "no_refine"

def final_review(state: dict) -> dict:
    current_state = TranslationState.model_validate(state)
    final_trans = current_state.refined_translation or current_state.human_translation
    return {"final_translation": final_trans}

def output_result(state: dict) -> dict:
    current_state = TranslationState.model_validate(state)
    output = current_state.final_translation or current_state.human_translation
    print("最终翻译结果:", output)
    return {"output": output}

# 4. 构建 LangGraph
builder = StateGraph(TranslationState)

# 修改 start 节点函数
builder.add_node("start", initialize_state)  # 使用新的初始化函数
builder.add_node("translator", translate)
builder.add_node("evaluator", evaluate)
builder.add_node("refiner", refine)
builder.add_node("final_reviewer", final_review)
builder.add_node("output", output_result)

# 添加边
builder.add_edge("start", "translator")
builder.add_edge("translator", "evaluator")
builder.add_edge("evaluator", "final_reviewer") # 默认不润色直接到最终审查

# 添加条件边 (根据评估结果决定是否润色)
builder.add_conditional_edges(
    "evaluator",
    should_refine,
    {
        "refine": "refiner",
        "no_refine": "final_reviewer",
    }
)
builder.add_edge("refiner", "final_reviewer")

builder.add_edge("final_reviewer", "output")

# 设置入口节点
builder.set_entry_point("start")

# 构建可运行的图
graph = builder.compile()

# 5. 运行 LangGraph
inputs = {
    "original_text": '''在经过一周多时间的灰度测试后，微信终于在近日宣布：语音消息倍速播放功能正式上线了。
现在只需将微信更新至8.0.55以上版本，iOS和安卓用户都可以使用该功能。
当聊天中的语音消息时长超过5秒时，对语音消息进行重听或转文字后，语音条旁边就会自动出现“倍速播放”的按钮。''',
    "source_language": "Chinese",
    "target_language": "English",
    "human_translation": "",  # 添加空字符串作为初始值
    "google_translation": ""  # 添加空字符串作为初始值
}

result = graph.invoke(inputs)
print(result)

# 4. Excel处理函数更新
def process_excel_batch() -> list:
    df = pd.read_excel('google_text.xlsx')
    results = []

    for _, row in df.iterrows():
        state = TranslationState(
            original_text=row['Chinese'],
            human_translation=row['English'],
            google_translation=row['Google English']
        )
        
        evaluation = evaluator.invoke(state.model_dump())
        
        results.append({
            'Original': state.original_text,
            'Human Translation': state.human_translation,
            'Google Translation': state.google_translation,
            'Evaluation': evaluation.content
        })

    results_df = pd.DataFrame(results)
    results_df.to_excel('evaluation_results.xlsx', index=False)
    return results

if __name__ == "__main__":
    # 测试单个翻译
    test_input = TranslationState(
        original_text='''在经过一周多时间的灰度测试后，微信终于在近日宣布：语音消息倍速播放功能正式上线了。
        现在只需将微信更新至8.0.55以上版本，iOS和安卓用户都可以使用该功能。
        当聊天中的语音消息时长超过5秒时，对语音消息进行重听或转文字后，语音条旁边就会自动出现"倍速播放"的按钮。''',
        human_translation="",
        google_translation="",
        source_language="Chinese",
        target_language="English"
    )
    
    result = graph.invoke(test_input.model_dump())
    print(result)
    
    # 批量处理Excel文件
    results = process_excel_batch()
    print("评估完成，结果已保存到 evaluation_results.xlsx")
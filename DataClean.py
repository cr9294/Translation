import json
import pandas as pd
import re

def is_valid_text(text):
    """
    检查文本是否有效（不含乱码且格式正确）
    """
    # 检查是否包含过多特殊字符
    special_char_ratio = len(re.findall(r'[^\u4e00-\u9fff\w。，、；：""（）《》？！]', text)) / len(text)
    if special_char_ratio > 0.1:  # 特殊字符比例超过10%视为异常
        return False
    
    # 检查中文字符比例
    chinese_char_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    if chinese_char_count / len(text) < 0.5:  # 中文字符少于50%视为异常
        return False
    
    return True

def truncate_content(content, max_length=1000):
    """
    将超过最大长度的文本截断到最近的句号
    """
    if len(content) <= max_length:
        return content
    
    # 在最大长度位置向前查找最近的句号
    last_period = content[:max_length].rstrip().rfind('。')
    if last_period == -1:  # 如果没找到句号，则在最大长度处直接截断
        return content[:max_length]
    return content[:last_period + 1]  # 包含句号

def process_json_data():
    filtered_contents = []
    
    with open('news2016zh_valid.json', 'r', encoding='utf-8') as f:
        count = 0
        invalid_count = 0
        for line in f:
            if count >= 3000:
                break
            try:
                data = json.loads(line.strip())
                content = data.get('content', '')
                
                # 内容长度检查
                if len(content) < 300:
                    continue
                
                # 检查文本质量
                if not is_valid_text(content):
                    invalid_count += 1
                    continue
                
                # 长度截断
                if len(content) > 1000:
                    content = truncate_content(content)
                
                filtered_contents.append({
                    'content': content
                })
                count += 1
            except (json.JSONDecodeError, UnicodeError):
                invalid_count += 1
                continue

    # 创建DataFrame并保存
    df = pd.DataFrame(filtered_contents)
    df.columns = ['Chinese']
    
    df.to_excel('filtered_contents.xlsx', index=False)
    print(f"处理完成，共保存 {len(filtered_contents)} 条有效数据")
    print(f"已过滤 {invalid_count} 条无效或含乱码数据")

if __name__ == "__main__":
    process_json_data()
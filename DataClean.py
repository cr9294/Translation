import json
import pandas as pd

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
        for line in f:
            if count >= 500:
                break
            try:
                data = json.loads(line.strip())
                content = data.get('content', '')
                
                # 如果内容长度小于300，跳过
                if len(content) < 300:
                    continue
                
                # 如果内容超过1000，进行智能截断
                if len(content) > 1000:
                    content = truncate_content(content)
                
                filtered_contents.append({
                    'content': content
                })
                count += 1
            except json.JSONDecodeError:
                continue

    # 创建DataFrame，设置中文列名
    df = pd.DataFrame(filtered_contents)
    df.columns = ['Chinese']  # 将'content'改为'文本内容'
    
    # 保存为Excel
    df.to_excel('filtered_contents.xlsx', index=False)
    print(f"处理完成，共保存 {len(filtered_contents)} 条数据")

if __name__ == "__main__":
    process_json_data()
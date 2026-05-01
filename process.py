import os
import google.generativeai as genai
import pandas as pd
from datetime import datetime

# 1. 配置 API (从环境变量读取，保证安全)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

BATCH_SIZE = 20 # 每次处理的行数

def process_sentences():
    # 读取进度
    if os.path.exists("last_index.txt"):
        with open("last_index.txt", "r") as f:
            start_idx = int(f.read().strip())
    else:
        start_idx = 0

    # 读取输入文件
    df = pd.read_csv("input.csv")
    
    # 截取本次要处理的行
    end_idx = start_idx + BATCH_SIZE
    batch_df = df.iloc[start_idx:end_idx]
    
    if batch_df.empty:
        print("所有内容已处理完毕。")
        return

    # 拼接文本给 AI
    input_text = "\n".join(batch_df['内容'].astype(str).tolist())
    
    # 获取 Prompt 并调用 API
    # 这里引用你之前定义的 Prompt 逻辑
    prompt = f"请解析以下文本：\n{input_text}" 
    
    response = model.generate_content(prompt)
    html_content = response.text.replace("```html", "").replace("```", "")

    # 更新 RSS (简化逻辑：直接更新 feed.xml 里的项)
    update_rss(html_content)

    # 记录进度
    with open("last_index.txt", "w") as f:
        f.write(str(end_idx))

def update_rss(new_content):
    # 这里可以用简单的字符串操作或 xml 库来维护 feed.xml
    # 每次运行生成一个新的 <item> 放入 RSS 头部
    pass

if __name__ == "__main__":
    process_sentences()

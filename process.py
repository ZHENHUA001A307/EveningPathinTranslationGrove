import os
import json
import pandas as pd
from google import genai
from datetime import datetime

# --- 配置区 ---
INPUT_FILE = 'input.csv'
PROGRESS_FILE = 'last_index.txt'
PROMPT_FILE = 'prompt.json'
RSS_FILE = 'feed.xml'
BATCH_SIZE = 20

print(f"--- 任务启动: {datetime.now()} ---")

# --- 1. 初始化 AI ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("错误: 找不到环境变量 GEMINI_API_KEY")
    exit(1)

# 初始化客户端
client = genai.Client(api_key=api_key)

def load_system_instruction():
    if not os.path.exists(PROMPT_FILE):
        print(f"错误: 找不到 {PROMPT_FILE}")
        exit(1)
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    inst = f"角色: {data['assistant_profile']['role']}\n任务: {data['assistant_profile']['task']}\n"
    inst += "要求:\n" + "\n".join(data['instructions'])
    return inst

# --- 2. 进度管理 ---
def get_last_index():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return int(content) if content else 0
    return 0

# --- 3. RSS 生成 ---
def update_rss(html_content, source_info):
    print("正在尝试写入 feed.xml...")
    now = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
    # 彻底移除 Markdown 代码块标记，防止 RSS 解析失败
    html_clean = html_content.replace("```html", "").replace("```", "").strip()
    
    item_xml = f"""
    <item>
        <title>外语学习 - {source_info} - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
        <description><![CDATA[{html_clean}]]></description>
        <pubDate>{now}</pubDate>
        <guid>{datetime.now().timestamp()}</guid>
    </item>"""
    
    if not os.path.exists(RSS_FILE):
        content = f'<?xml version="1.0" encoding="UTF-8" ?>\n<rss version="2.0">\n<channel>\n<title>AI学习推送</title>\n{item_xml}\n</channel>\n</rss>'
    else:
        with open(RSS_FILE, 'r', encoding='utf-8') as f:
            old = f.read()
        # 插入到 channel 标签后，保证最新条目在最前
        content = old.replace("<channel>", f"<channel>\n{item_xml}")
    
    with open(RSS_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    print("feed.xml 写入成功。")

# --- 4. 主流程 ---
def main():
    try:
        # 基础文件检查
        if not os.path.exists(INPUT_FILE):
            print(f"错误: 找不到 {INPUT_FILE}")
            return

        df = pd.read_csv(INPUT_FILE)
        total_rows = len(df)
        start_idx = get_last_index()

        if start_idx >= total_rows:
            print("所有内容已处理完毕。")
            return

        # 截取本批次
        batch = df.iloc[start_idx : start_idx + BATCH_SIZE]
        input_text = "\n".join([str(row['内容']) for _, row in batch.iterrows()])

        # --- 5. 核心测试点：Gemini 1.5 Flash 调用 ---
        print(f"正在调用 Gemini API 处理第 {start_idx} 到 {start_idx + len(batch) - 1} 行...")
        
        # 注意：这里模型名称仅写 'gemini-1.5-flash'
        # system_instruction 放在 config 字典内
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=input_text,
            config={
                'system_instruction': load_system_instruction()
            }
        )
        
        if not response.text:
            print("API 返回空，可能是触发了安全过滤。")
            return
            
        print(f"API 调用成功！")

        # 更新 RSS 和进度
        source_name = batch.iloc[0]['来源'] if '来源' in batch.columns else "Daily"
        update_rss(response.text, source_name)

        new_idx = start_idx + len(batch)
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            f.write(str(new_idx))
        print(f"进度已更新至: {new_idx}")

    except Exception as e:
        print(f"!!! 程序崩溃 !!!")
        # 打印更多错误细节用于调试
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main()

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
client = genai.Client(
    api_key=api_key,
    http_options={'api_version': 'v1'} # 关键：强制使用 v1
)
if not api_key:
    print("错误: 找不到环境变量 GEMINI_API_KEY")
    exit(1)

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
    item_xml = f"""
    <item>
        <title>外语学习 - {source_info} - {datetime.now().strftime('%Y%m%d')}</title>
        <description><![CDATA[{html_content}]]></description>
        <pubDate>{now}</pubDate>
        <guid>{datetime.now().timestamp()}</guid>
    </item>"""
    
    if not os.path.exists(RSS_FILE):
        content = f'<?xml version="1.0" encoding="UTF-8" ?>\n<rss version="2.0">\n<channel>\n<title>AI学习推送</title>\n{item_xml}\n</channel>\n</rss>'
    else:
        with open(RSS_FILE, 'r', encoding='utf-8') as f:
            old = f.read()
        content = old.replace("<channel>", f"<channel>\n{item_xml}")
    
    with open(RSS_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    print("feed.xml 写入成功。")

# --- 4. 主流程 ---
def main():
    try:
        # (读取 CSV 和进度代码保持不变...)
        
        print("正在调用 Gemini API (v1)...")
        response = client.models.generate_content(
            model='gemini-1.5-flash', 
            contents=input_text,
            config={
                'system_instruction': load_system_instruction(),
                'response_mime_type': 'text/plain' 
            }
        )
        
        # (后续处理和 RSS 代码保持不变...)
        
        # 成功后更新进度
        new_idx = start_idx + len(batch)
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            f.write(str(new_idx))
        print(f"进度已更新至: {new_idx}")

    except Exception as e:
        print(f"!!! 程序崩溃 !!! 错误类型: {type(e).__name__}")
        print(f"错误详情: {str(e)}")
        raise e

if __name__ == "__main__":
    main()

import os
import json
import pandas as pd
from datetime import datetime

# 尝试更稳健的导入
try:
    from google import genai
except ImportError:
    # 如果还是报错，尝试直接从子模块导入（针对某些环境问题）
    import google.genai as genai

# --- 配置区 ---
INPUT_FILE = 'input.csv'
PROGRESS_FILE = 'last_index.txt'
PROMPT_FILE = 'prompt.json'
RSS_FILE = 'feed.xml'
BATCH_SIZE = 20

# --- 1. 初始化最新 AI 客户端 ---
# 新版 SDK 会自动识别环境变量 GEMINI_API_KEY，也可以显式传入
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def load_system_instruction():
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    inst = f"角色: {data['assistant_profile']['role']}\n"
    inst += f"任务: {data['assistant_profile']['task']}\n"
    inst += "具体要求:\n" + "\n".join(data['instructions'])
    inst += f"\n输出模板起始部分: {data['html_template']}"
    return inst

# --- 2. 进度管理与 RSS 逻辑 (保持不变) ---
def get_last_index():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return int(f.read().strip())
    return 0

def save_last_index(idx):
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(idx))

def update_rss(html_content, source_info):
    now = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
    title = f"外语学习推送 - {datetime.now().strftime('%Y-%m-%d')}"
    item_xml = f"""
    <item>
        <title>{title} ({source_info})</title>
        <link>https://github.com/your-username</link>
        <description><![CDATA[{html_content}]]></description>
        <pubDate>{now}</pubDate>
        <guid>{datetime.now().timestamp()}</guid>
    </item>
    """
    if not os.path.exists(RSS_FILE):
        rss_base = f'<?xml version="1.0" encoding="UTF-8" ?>\n<rss version="2.0">\n<channel>\n<title>AI 外语精读</title>\n{item_xml}\n</channel>\n</rss>'
        with open(RSS_FILE, 'w', encoding='utf-8') as f:
            f.write(rss_base)
    else:
        with open(RSS_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content = content.replace("<channel>", f"<channel>\n{item_xml}")
        with open(RSS_FILE, 'w', encoding='utf-8') as f:
            f.write(new_content)

# --- 3. 主流程 ---
def main():
    start_idx = get_last_index()
    if not os.path.exists(INPUT_FILE):
        print("Error: input.csv not found")
        return

    df = pd.read_csv(INPUT_FILE)
    if start_idx >= len(df):
        print("All caught up!")
        return

    batch = df.iloc[start_idx : start_idx + BATCH_SIZE]
    source_name = batch['来源'].iloc[0] if '来源' in batch.columns else "Daily"
    input_text = "\n".join(batch['内容'].astype(str).tolist())

    print(f"Processing lines {start_idx} to {start_idx + len(batch)}...")

    try:
        # 新版 SDK 的调用方式
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=input_text,
            config={
                'system_instruction': load_system_instruction(),
                'response_mime_type': 'text/plain' # 确保输出纯文本
            }
        )
        
        result_html = response.text.strip()
        # 移除 Markdown 包裹
        if "```" in result_html:
            result_html = result_html.split("```")[1]
            if result_html.startswith("html"):
                result_html = result_html[4:]

        update_rss(result_html, source_name)
        save_last_index(start_idx + len(batch))
        print("Done! RSS updated.")

    except Exception as e:
        print(f"API Error: {e}")

if __name__ == "__main__":
    main()

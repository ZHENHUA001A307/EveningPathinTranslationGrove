import os
import json
import pandas as pd
import google.generativeai as genai
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom

# --- 配置区 ---
INPUT_FILE = 'input.csv'
PROGRESS_FILE = 'last_index.txt'
PROMPT_FILE = 'prompt.json'
RSS_FILE = 'feed.xml'
BATCH_SIZE = 20  # 每次处理的行数

# --- 1. 初始化 AI ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def load_system_instruction():
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 将 JSON 拼接成一段给 AI 的系统指令
    inst = f"角色: {data['assistant_profile']['role']}\n"
    inst += f"任务: {data['assistant_profile']['task']}\n"
    inst += "具体要求:\n" + "\n".join(data['instructions'])
    inst += f"\n输出模板起始部分: {data['html_template']}"
    return inst

model = genai.GenerativeModel(
    model_name='gemini-1.5-flash', # flash 速度快且便宜，适合批量翻译
    system_instruction=load_system_instruction()
)

# --- 2. 进度管理 ---
def get_last_index():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return int(f.read().strip())
    return 0

def save_last_index(idx):
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(idx))

# --- 3. RSS 生成逻辑 ---
def update_rss(html_content, source_info):
    now = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
    title = f"外语学习推送 - {datetime.now().strftime('%Y-%m-%d')}"
    
    # 简单的 RSS 模板生成
    item_xml = f"""
    <item>
        <title>{title} ({source_info})</title>
        <link>[https://github.com/your-username/your-repo](https://github.com/your-username/your-repo)</link>
        <description><![CDATA[{html_content}]]></description>
        <pubDate>{now}</pubDate>
        <guid>{datetime.now().timestamp()}</guid>
    </item>
    """
    
    if not os.path.exists(RSS_FILE):
        rss_base = f"""<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
<channel>
    <title>AI 外语精读推送</title>
    <description>基于 Gemini API 的自动逐句讲解</description>
    {item_xml}
</channel>
</rss>"""
        with open(RSS_FILE, 'w', encoding='utf-8') as f:
            f.write(rss_base)
    else:
        # 如果文件存在，把新的 item 插入到开头
        with open(RSS_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content = content.replace("<channel>", f"<channel>\n{item_xml}")
        with open(RSS_FILE, 'w', encoding='utf-8') as f:
            f.write(new_content)

# --- 4. 主流程 ---
def main():
    start_idx = get_last_index()
    
    if not os.path.exists(INPUT_FILE):
        print(f"找不到 {INPUT_FILE} 文件")
        return

    df = pd.read_csv(INPUT_FILE)
    
    if start_idx >= len(df):
        print("所有内容已处理。")
        return

    # 截取本次要处理的数据
    batch = df.iloc[start_idx : start_idx + BATCH_SIZE]
    source_name = batch['来源'].iloc[0] if '来源' in batch.columns else "未知来源"
    
    # 将 DataFrame 转换为纯文本供 AI 处理
    # 格式：1. 句子内容... \n 2. 句子内容...
    lines = []
    for i, row in batch.iterrows():
        lines.append(f"{row['内容']}")
    input_text = "\n".join(lines)

    print(f"正在处理第 {start_idx} 到 {start_idx + len(batch)} 行...")
    
    try:
        response = model.generate_content(input_text)
        result_html = response.text.strip()
        
        # 清理可能存在的 Markdown 代码块标记
        result_html = result_html.replace("```html", "").replace("```", "")

        # 更新 RSS
        update_rss(result_html, source_name)
        
        # 保存进度
        save_last_index(start_idx + len(batch))
        print("RSS 更新成功！")
        
    except Exception as e:
        print(f"处理失败: {e}")

if __name__ == "__main__":
    main()

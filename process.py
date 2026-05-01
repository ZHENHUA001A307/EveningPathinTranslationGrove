import os
import json
import pandas as pd
from google import genai
from datetime import datetime
import traceback

# ======================= 配置区 =======================
# 语言特定配置
LANG_CONFIG = {
    'en': {
        'input_file': 'input_en.csv',
        'progress_file': 'last_index_en.txt',
        'batch_size': 20,           # 每次推送的句子数量
        'reading_speed': 350,       # 英语阅读速度（字符/分钟）
    },
    'jp': {
        'input_file': 'input_jp.csv',
        'progress_file': 'last_index_jp.txt',
        'batch_size': 15,
        'reading_speed': 180,       # 日语阅读速度（字符/分钟）
    }
}

# 信号量文件（记录当前应推送的语言：奇数=英语，偶数=日语）
TOGGLE_FILE = 'toggle.txt'

# 共用文件
PROMPT_FILE = 'prompt.json'
RSS_FILE = 'feed.xml'
MODEL_NAME = 'gemini-3-flash-preview'  # 使用稳定可用的模型

# ======================= 辅助函数 =======================
def get_signal_language():
    """返回当前应该推送的语言 ('en' 或 'jp')，并递增信号量"""
    if not os.path.exists(TOGGLE_FILE):
        # 默认从英语开始（信号量=1）
        with open(TOGGLE_FILE, 'w') as f:
            f.write('1')
        return 'en', 1

    with open(TOGGLE_FILE, 'r') as f:
        val = int(f.read().strip() or 0)

    lang = 'en' if val % 2 == 1 else 'jp'
    # 递增并写回（先读后写，确保下次运行变化）
    new_val = val + 1
    with open(TOGGLE_FILE, 'w') as f:
        f.write(str(new_val))
    return lang, new_val - 1   # 返回语言和本次使用的信号量原值

def load_system_instruction():
    if not os.path.exists(PROMPT_FILE):
        print(f"错误: 找不到 {PROMPT_FILE}")
        exit(1)
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 将 JSON 配置转成指令字符串
    inst = f"角色: {data['assistant_profile']['role']}\n任务: {data['assistant_profile']['task']}\n"
    inst += "要求:\n" + "\n".join(data['instructions'])
    return inst

def get_last_index(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return int(content) if content else 0
    return 0

def update_progress(progress_file, new_index):
    with open(progress_file, 'w', encoding='utf-8') as f:
        f.write(str(new_index))

def calculate_reading_time(text, lang):
    """计算预计阅读时间，返回 (分钟, 提示字符串)"""
    total_chars = len(text)               # 字符总数（含字母、符号、空格）
    speed = LANG_CONFIG[lang]['reading_speed']
    minutes = total_chars / speed * 3
    # 向上取整，最少显示1分钟
    minutes_ceil = max(1, int(minutes) + (1 if minutes > int(minutes) else 0))
    marker = f"⏱ 预计阅读时间：{minutes_ceil} 分钟 ({total_chars} 字符)"
    return minutes_ceil, marker

def update_rss(html_content, source_info, lang):
    """将 AI 生成的表格和阅读标记合并，写入 RSS"""
    # 在表格最前面加上阅读时间标记（单独一行，不加表格内）
    # 注意：html_content 是 AI 生成的完整表格，我们直接在其前面拼一个段落
    marked_html = f"<p>{source_info.get('time_marker', '')}</p>\n{html_content}"
    
    now = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
    # 清理可能的 Markdown 包裹符
    html_clean = marked_html.replace("```html", "").replace("```", "").strip()
    
    title_prefix = "外语学习"
    if lang == 'en':
        title_prefix = "🇬🇧 英语学习"
    else:
        title_prefix = "🇯🇵 日语学习"
    
    item_xml = f"""
    <item>
        <title>{title_prefix} - {source_info['name']} - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
        <description><![CDATA[{html_clean}]]></description>
        <pubDate>{now}</pubDate>
        <guid>{datetime.now().timestamp()}</guid>
    </item>"""
    
    if not os.path.exists(RSS_FILE):
        content = f'<?xml version="1.0" encoding="UTF-8" ?>\n<rss version="2.0">\n<channel>\n<title>AI学习推送</title>\n{item_xml}\n</channel>\n</rss>'
    else:
        with open(RSS_FILE, 'r', encoding='utf-8') as f:
            old = f.read()
        if "<item>" in old:
            content = old.replace("<item>", f"{item_xml}\n    <item>", 1)
        else:
            content = old.replace("<channel>", f"<channel>\n{item_xml}")
    
    with open(RSS_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    print("feed.xml 写入成功。")

# ======================= 主流程 =======================
def main():
    try:
        # 1. 根据信号量决定当前语言
        lang, signal_value = get_signal_language()
        print(f"当前信号量 = {signal_value} → 推送语言：{'英语' if lang == 'en' else '日语'}")
        
        cfg = LANG_CONFIG[lang]
        input_file = cfg['input_file']
        progress_file = cfg['progress_file']
        batch_size = cfg['batch_size']
        
        # 2. 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"错误: 找不到 {input_file}，跳过此次推送。")
            return
        
        # 3. 读取 CSV
        df = pd.read_csv(input_file)
        if '内容' not in df.columns:
            print(f"错误: {input_file} 缺少 '内容' 列")
            return
        
        total_rows = len(df)
        start_idx = get_last_index(progress_file)
        
        if start_idx >= total_rows:
            print(f"{input_file} 所有内容已处理完毕 (start_idx={start_idx}, total={total_rows})，无新数据。")
            return
        
        # 4. 取一批数据
        end_idx = min(start_idx + batch_size, total_rows)
        batch = df.iloc[start_idx:end_idx]
        input_text = "\n".join([str(row['内容']) for _, row in batch.iterrows()])
        
        # 5. 计算阅读时间标记（基于原始文本）
        minutes, marker = calculate_reading_time(input_text, lang)
        print(f"本次推送 {len(batch)} 行，总字符数 {len(input_text)}，预计阅读 {minutes} 分钟")
        
        # 6. 调用 Gemini API
        print(f"正在调用 Gemini 处理第 {start_idx} 到 {end_idx-1} 行...")
        client = genai.Client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=input_text,
            config={
                'system_instruction': load_system_instruction(),
                'temperature': 0.7
            }
        )
        
        if not response.text:
            print("警告: API 未能生成内容")
            return
        
        print("API 调用成功！")
        
        # 7. 组织来源信息（用于 RSS 标题和阅读标记）
        source_name = batch.iloc[0]['来源'] if '来源' in df.columns else (f"en_batch_{start_idx}" if lang == 'en' else f"jp_batch_{start_idx}")
        source_info = {
            'name': source_name,
            'time_marker': marker
        }
        
        # 8. 写入 RSS（AI 生成的表格 + 阅读标记）
        update_rss(response.text, source_info, lang)
        
        # 9. 更新对应语言的进度
        new_idx = end_idx
        update_progress(progress_file, new_idx)
        print(f"进度已更新至 {new_idx} (总行数 {total_rows})")
        
    except Exception as e:
        print("!!! 程序崩溃 !!!")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    print(f"--- 任务启动: {datetime.now()} ---")
    main()

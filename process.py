import os
import json
import logging
import sys
import traceback
from datetime import datetime
import pandas as pd
from google import genai

# ======================= 日志配置 =======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ======================= 动态环境变量配置区 =======================
LANG_CN = os.getenv('LANG_CN', '外语')
INPUT_FILE = os.getenv('INPUT_FILE', 'corpus/input_en.csv')
PROGRESS_FILE = os.getenv('PROGRESS_FILE', 'progress/last_index_en.txt')
PROMPT_FILE = os.getenv('PROMPT_FILE', 'prompts/en_prompt.json')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '40'))
READING_SPEED = int(os.getenv('READING_SPEED', '150'))
RSS_FILE = os.getenv('RSS_FILE', 'feed.xml')
MODEL_NAME = os.getenv('MODEL_NAME', 'gemini-2.5-flash') 

# ======================= 辅助函数 =======================
def load_system_instruction():
    if not os.path.exists(PROMPT_FILE):
        logger.error(f"❌ 错误：提示词配置文件不存在: {PROMPT_FILE}")
        sys.exit(1)
    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"❌ 错误：读取 {PROMPT_FILE} 失败: {e}")
        sys.exit(1)

    inst = f"角色: {data['assistant_profile']['role']}\n任务: {data['assistant_profile']['task']}\n"
    inst += "要求:\n" + "\n".join(data['instructions'])
    return inst

def get_last_index():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            idx = int(content) if content else 0
            logger.info(f"📍 当前进度读取 [{PROGRESS_FILE}]: {idx}")
            return idx
    logger.info(f"💡 进度文件 [{PROGRESS_FILE}] 不存在，将从第 0 行开始")
    return 0

def update_progress(new_index):
    os.makedirs(os.path.dirname(PROGRESS_FILE) or '.', exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        f.write(str(new_index))
    logger.info(f"✅ 进度更新成功 [{PROGRESS_FILE}] -> {new_index}")

def calculate_reading_time(text):
    total_chars = len(text)
    minutes = total_chars / READING_SPEED * 3
    minutes_ceil = max(1, int(minutes) + (1 if minutes > int(minutes) else 0))
    return f"⏱ 预计阅读时间：{minutes_ceil} 分钟 ({total_chars} 字符)"

def update_rss(html_content, source_name, time_marker):
    marked_html = f"<p>{time_marker}</p>\n{html_content}"
    now = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
    html_clean = marked_html.replace("```html", "").replace("```", "").strip()

    title = f"[{LANG_CN}] {source_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    item_xml = f"""
    <item>
        <title>{title}</title>
        <description><![CDATA[{html_clean}]]></description>
        <pubDate>{now}</pubDate>
        <guid>{datetime.now().timestamp()}</guid>
    </item>"""

    try:
        if not os.path.exists(RSS_FILE):
            content = f'<?xml version="1.0" encoding="UTF-8" ?>\n<rss version="2.0">\n<channel>\n<title>AI 语言学习推送源</title>\n{item_xml}\n</channel>\n</rss>'
        else:
            with open(RSS_FILE, 'r', encoding='utf-8') as f:
                old = f.read()
            if "<item>" in old:
                content = old.replace("<item>", f"{item_xml}\n    <item>", 1)
            else:
                content = old.replace("<channel>", f"<channel>\n{item_xml}")
        
        with open(RSS_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"🚀 RSS 新条目已成功写入/追加: {title}")
    except Exception as e:
        logger.error(f"❌ 错误：写入 RSS 文件失败: {e}")
        sys.exit(1)

# ======================= 主流程 =======================
def main():
    logger.info(f"--- [{LANG_CN}] 推送任务正式启动 ---")
    try:
        if not os.path.exists(INPUT_FILE):
            logger.error(f"❌ 错误：语料源文件不存在，请检查路径: {INPUT_FILE}")
            sys.exit(1) # 改为令 Action 报错的退出

        df = pd.read_csv(INPUT_FILE)
        if df.empty:
            logger.error(f"❌ 错误：语料文件 {INPUT_FILE} 内容为空，无法生成推送")
            sys.exit(1)

        df['内容'] = df['内容'].fillna('').astype(str)
        df['来源'] = df['来源'].fillna('未知来源').astype(str)

        total_rows = len(df)
        start_idx = get_last_index()

        if start_idx >= total_rows:
            # 语料学完属于正常业务现象，不报错，但打印醒目提示
            logger.warning(f"⚠️ 提示：所有语料已学完 (当前索引 {start_idx} >= 总行数 {total_rows})。本次无新内容写入。")
            return

        # 动态批次切分
        first_source = df.at[start_idx, '来源']
        end_idx = start_idx
        while end_idx < total_rows:
            if df.at[end_idx, '来源'] != first_source:
                logger.info(f"[批次截断] 检测到来源由 '{first_source}' 变为 '{df.at[end_idx, '来源']}'")
                break
            if end_idx - start_idx >= BATCH_SIZE:
                break
            end_idx += 1

        batch = df.iloc[start_idx:end_idx]
        logger.info(f"📊 本批次待处理: 行 {start_idx} 至 {end_idx - 1} (共 {len(batch)} 行) | 来源: {first_source}")

        input_text = "\n".join(batch['内容'].tolist())
        time_marker = calculate_reading_time(input_text)

        # Gemini API 调用
        system_instruction = load_system_instruction()
        logger.info("📡 正在向 Gemini 接口发起请求...")
        
        try:
            client = genai.Client()
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=input_text,
                config={
                    'system_instruction': system_instruction,
                    'temperature': 0.3
                }
            )
            if not response.text:
                logger.error("❌ 错误：Gemini API 返回了空文本 (Empty Response)")
                sys.exit(1)
        except Exception as e:
            logger.error(f"❌ 错误：Gemini API 调用失败: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1) # 接口报错时令 Action 变红，方便捕获密钥或额度错误

        # 执行写入
        update_rss(response.text, first_source, time_marker)
        update_progress(end_idx)
        logger.info(f"--- 🎉 [{LANG_CN}] 推送任务全部顺利完成 ---")

    except Exception as e:
        logger.error("❌ 顶级未知异常崩溃:")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

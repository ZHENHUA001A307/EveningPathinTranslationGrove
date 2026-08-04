import os
import json
import logging
import sys
import traceback
from datetime import datetime
import pandas as pd
import requests

# ======================= 日志配置 =======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ======================= 动态环境配置区 =======================
LANG_CN = os.getenv('LANG_CN', '外语')
INPUT_FILE = os.getenv('INPUT_FILE', 'corpus/input_en.csv')
PROGRESS_FILE = os.getenv('PROGRESS_FILE', 'progress/last_index_en.txt')
PROMPT_FILE = os.getenv('PROMPT_FILE', 'prompts/en_prompt.json')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '20'))
READING_SPEED = int(os.getenv('READING_SPEED', '50'))
RSS_FILE = os.getenv('RSS_FILE', 'feed.xml')

# 🆕 切片模式：by_topic（按话题） 或 by_count（纯按条数）
SLICE_MODE = os.getenv('SLICE_MODE', 'by_count')

# DeepSeek / OpenAI 兼容配置
API_KEY = os.getenv('DEEPSEEK_API_KEY')
API_URL = os.getenv('API_URL', 'https://api.deepseek.com/chat/completions')
MODEL_NAME = os.getenv('MODEL_NAME', 'deepseek-v4-pro')

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
    import re

    marked_html = f"<p>{time_marker}</p>\n{html_content}"
    now = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
    
    html_clean = marked_html.replace("```html", "").replace("```", "").strip()
    title = f"[{LANG_CN}] {source_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    new_item = f"""    <item>
        <title>{title}</title>
        <description><![CDATA[{html_clean}]]></description>
        <pubDate>{now}</pubDate>
        <guid>{datetime.now().timestamp()}</guid>
    </item>"""

    try:
        existing_items = []
        
        if os.path.exists(RSS_FILE) and os.path.getsize(RSS_FILE) > 0:
            with open(RSS_FILE, 'r', encoding='utf-8') as f:
                old_content = f.read()
            existing_items = re.findall(r'<item>.*?</item>', old_content, re.DOTALL)

        all_items = [new_item] + existing_items
        items_to_keep = all_items[:20]
        logger.info(f"📦 RSS 容器控容中：当前池内总计 {len(all_items)} 条，已截取保留最新 {len(items_to_keep)} 条")

        items_joined = "\n".join(items_to_keep)
        final_xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
<channel>
    <title>AI 语言学习推送源</title>
    <link>https://github.com</link>
    <description>由 DeepSeek 驱动的自动化语言学习 RSS 推送</description>
{items_joined}
</channel>
</rss>"""
        
        with open(RSS_FILE, 'w', encoding='utf-8') as f:
            f.write(final_xml.strip())
            
        logger.info(f"🚀 RSS 新条目已成功顶入并瘦身: {title}")
        
    except Exception as e:
        logger.error(f"❌ 错误：写入或裁剪 RSS 文件失败: {e}")
        sys.exit(1)

# ======================= 🆕 核心切片逻辑（支持两种模式） =======================
def get_batch(df, start_idx):
    """
    根据 SLICE_MODE 决定如何切分批次：
    - by_topic: 按来源（话题）切片，每个单元独立推送
    - by_count: 无视来源，纯按 BATCH_SIZE 条数切片
    """
    total_rows = len(df)
    end_idx = start_idx

    if SLICE_MODE == 'by_count':
        # 🆕 模式二：纯按条数切片，无视来源
        end_idx = min(start_idx + BATCH_SIZE, total_rows)
        source_name = df.at[start_idx, '来源']
        # 如果跨越多个来源，标题做标记
        if end_idx < total_rows and df.at[end_idx, '来源'] != source_name:
            source_name = f"{source_name} 等多主题"
        logger.info(f"🔧 切片模式: by_count（按条数），本批次 {end_idx - start_idx} 条")
        return end_idx, source_name

    else:
        # 🆕 模式一（默认）：按话题（来源）切片，原逻辑
        first_source = df.at[start_idx, '来源']
        while end_idx < total_rows:
            if df.at[end_idx, '来源'] != first_source:
                break
            if end_idx - start_idx >= BATCH_SIZE:
                break
            end_idx += 1
        logger.info(f"🔧 切片模式: by_topic（按话题），本批次 {end_idx - start_idx} 条")
        return end_idx, first_source

# ======================= 主流程 =======================
def main():
    logger.info(f"--- [{LANG_CN}] 推送任务正式启动 (使用模型: {MODEL_NAME}) ---")
    logger.info(f"🔧 切片模式: {SLICE_MODE} | 批次大小: {BATCH_SIZE}")
    
    if not API_KEY:
        logger.error("❌ 错误：未配置环境变量 DEEPSEEK_API_KEY，请检查 GitHub Secrets")
        sys.exit(1)

    try:
        if not os.path.exists(INPUT_FILE):
            logger.error(f"❌ 错误：语料源文件不存在: {INPUT_FILE}")
            sys.exit(1)

        df = pd.read_csv(INPUT_FILE)
        if df.empty:
            logger.error(f"❌ 错误：语料文件 {INPUT_FILE} 内容为空")
            sys.exit(1)

        df['内容'] = df['内容'].fillna('').astype(str)
        df['来源'] = df['来源'].fillna('未知来源').astype(str)

        total_rows = len(df)
        start_idx = get_last_index()

        if start_idx >= total_rows:
            logger.warning(f"⚠️ 提示：所有语料已学完 (索引 {start_idx} >= 总行数 {total_rows})。无新内容写入。")
            return

        # 🆕 调用新的切片函数
        end_idx, source_name = get_batch(df, start_idx)
        batch = df.iloc[start_idx:end_idx]
        logger.info(f"📊 本批次待处理: 行 {start_idx} 至 {end_idx - 1} (共 {len(batch)} 行) | 来源标识: {source_name}")

        input_text = "\n".join(batch['内容'].tolist())
        time_marker = calculate_reading_time(input_text)
        system_instruction = load_system_instruction()

        # ==================== OpenAI / DeepSeek 标准请求 ====================
        logger.info(f"📡 正在向 API 接口发起 POST 请求: {API_URL}")
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": input_text}
            ],
            "stream": False
        }
        
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                logger.error(f"❌ 错误：API 响应失败，状态码: {response.status_code}")
                logger.error(f"错误详情: {response.text}")
                sys.exit(1)
                
            res_data = response.json()
            ai_content = res_data['choices'][0]['message']['content']
            
            if not ai_content:
                logger.error("❌ 错误：API 返回了空文本内容")
                sys.exit(1)
                
            logger.info("✅ API 成功返回解析数据。")
        except Exception as e:
            logger.error(f"❌ 错误：网络请求或解析 JSON 失败: {e}")
            sys.exit(1)
        # ====================================================================

        # 写入 RSS 并更新进度
        update_rss(ai_content, source_name, time_marker)
        update_progress(end_idx)
        logger.info(f"--- 🎉 [{LANG_CN}] 推送任务全部顺利完成 ---")

    except Exception as e:
        logger.error("❌ 顶级未知异常崩溃:")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

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
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '40'))
READING_SPEED = int(os.getenv('READING_SPEED', '150'))
RSS_FILE = os.getenv('RSS_FILE', 'feed.xml')

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
    """将 AI 生成的 HTML 表格追加写入 RSS 文件，并严格限制只保留最新 20 条"""
    import re  # 引入正则工具箱，用来精准抓取旧条目

    marked_html = f"<p>{time_marker}</p>\n{html_content}"
    now = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
    
    # 彻底清洗可能夹杂的 Markdown 围栏标记
    html_clean = marked_html.replace("```html", "").replace("```", "").strip()
    title = f"[{LANG_CN}] {source_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    # 1. 构建本次全新的 item 节点
    new_item = f"""    <item>
        <title>{title}</title>
        <description><![CDATA[{html_clean}]]></description>
        <pubDate>{now}</pubDate>
        <guid>{datetime.now().timestamp()}</guid>
    </item>"""

    try:
        existing_items = []
        
        # 2. 如果文件存在且不为空，把里面所有的旧 <item>...</item> 全都捞出来
        if os.path.exists(RSS_FILE) and os.path.getsize(RSS_FILE) > 0:
            with open(RSS_FILE, 'r', encoding='utf-8') as f:
                old_content = f.read()
            # 使用正则抓取所有历史 item 块
            existing_items = re.findall(r'<item>.*?</item>', old_content, re.DOTALL)

        # 3. 把新条目放在最前面（置顶），并和旧条目合并
        all_items = [new_item] + existing_items

        # 4. ⚡ 核心控容：强行截取前 20 条（最新写入的 20 次内容）
        items_to_keep = all_items[:20]
        logger.info(f"📦 RSS 容器控容中：当前池内总计 {len(all_items)} 条，已截取保留最新 {len(items_to_keep)} 条")

        # 5. 重新格式化成一个标准的、干净的 RSS 文件结构
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
        
        # 6. 覆盖写入
        with open(RSS_FILE, 'w', encoding='utf-8') as f:
            f.write(final_xml.strip())
            
        logger.info(f"🚀 RSS 新条目已成功顶入并瘦身: {title}")
        
    except Exception as e:
        logger.error(f"❌ 错误：写入或裁剪 RSS 文件失败: {e}")
        sys.exit(1)

# ======================= 主流程 =======================
def main():
    logger.info(f"--- [{LANG_CN}] 推送任务正式启动 (使用模型: {MODEL_NAME}) ---")
    
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

        # 动态批次切分
        first_source = df.at[start_idx, '来源']
        end_idx = start_idx
        while end_idx < total_rows:
            if df.at[end_idx, '来源'] != first_source:
                logger.info(f"[批次截断] 来源由 '{first_source}' 变为 '{df.at[end_idx, '来源']}'")
                break
            if end_idx - start_idx >= BATCH_SIZE:
                break
            end_idx += 1

        batch = df.iloc[start_idx:end_idx]
        logger.info(f"📊 本批次待处理: 行 {start_idx} 至 {end_idx - 1} (共 {len(batch)} 行) | 来源: {first_source}")

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
        update_rss(ai_content, first_source, time_marker)
        update_progress(end_idx)
        logger.info(f"--- 🎉 [{LANG_CN}] 推送任务全部顺利完成 ---")

    except Exception as e:
        logger.error("❌ 顶级未知异常崩溃:")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

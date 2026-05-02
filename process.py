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

# ======================= 配置区 =======================
# 语言特定配置
LANG_CONFIG = {
    'en': {
        'input_file': 'input_en.csv',
        'progress_file': 'last_index_en.txt',
        'batch_size': 20,           # 每次推送的句子数量上限
        'reading_speed': 150,       # 英语阅读速度（字符/分钟）
    },
    'jp': {
        'input_file': 'input_jp.csv',
        'progress_file': 'last_index_jp.txt',
        'batch_size': 15,
        'reading_speed': 150,       # 日语阅读速度（字符/分钟）
    }
}

# 信号量文件（记录当前应推送的语言：奇数=英语，偶数=日语）
TOGGLE_FILE = 'toggle.txt'

# 共用文件
PROMPT_FILE = 'prompt.json'
RSS_FILE = 'feed.xml'
MODEL_NAME = 'gemini-3-flash-preview'  # 可替换为其他稳定模型

# ======================= 辅助函数 =======================
def get_signal_language():
    """返回当前应该推送的语言 ('en' 或 'jp')，并递增信号量"""
    if not os.path.exists(TOGGLE_FILE):
        with open(TOGGLE_FILE, 'w') as f:
            f.write('1')
        logger.info("信号量文件不存在，初始化为1（英语）")
        return 'en', 1

    with open(TOGGLE_FILE, 'r') as f:
        raw = f.read().strip()
    try:
        val = int(raw) if raw else 0
    except ValueError:
        logger.warning(f"信号量文件内容非法 ('{raw}')，重置为1")
        val = 0

    lang = 'en' if val % 2 == 1 else 'jp'
    new_val = val + 1
    with open(TOGGLE_FILE, 'w') as f:
        f.write(str(new_val))
    logger.debug(f"信号量: {val} -> {new_val}，语言: {lang}")
    return lang, val   # 返回语言和本次使用的信号量原值

def load_system_instruction():
    """加载系统指令 prompt"""
    if not os.path.exists(PROMPT_FILE):
        logger.error(f"系统指令文件不存在: {PROMPT_FILE}")
        sys.exit(1)
    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"读取 {PROMPT_FILE} 失败: {e}")
        sys.exit(1)

    inst = f"角色: {data['assistant_profile']['role']}\n任务: {data['assistant_profile']['task']}\n"
    inst += "要求:\n" + "\n".join(data['instructions'])
    return inst

def get_last_index(progress_file):
    """读取进度文件中的最后处理索引，不存在则返回0"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            idx = int(content) if content else 0
            logger.debug(f"读取进度 {progress_file}: {idx}")
            return idx
    logger.debug(f"进度文件 {progress_file} 不存在，从0开始")
    return 0

def update_progress(progress_file, new_index):
    """更新进度文件"""
    with open(progress_file, 'w', encoding='utf-8') as f:
        f.write(str(new_index))
    logger.debug(f"更新进度 {progress_file}: {new_index}")

def calculate_reading_time(text, lang):
    """计算预计阅读时间，返回 (分钟, 提示字符串)"""
    total_chars = len(text)
    speed = LANG_CONFIG[lang]['reading_speed']
    minutes = total_chars / speed * 3
    minutes_ceil = max(1, int(minutes) + (1 if minutes > int(minutes) else 0))
    marker = f"⏱ 预计阅读时间：{minutes_ceil} 分钟 ({total_chars} 字符)"
    return minutes_ceil, marker

def update_rss(html_content, source_info, lang):
    """将生成的表格和阅读标记写入 RSS 文件"""
    marked_html = f"<p>{source_info.get('time_marker', '')}</p>\n{html_content}"
    now = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
    html_clean = marked_html.replace("```html", "").replace("```", "").strip()

    title_prefix = "英语学习" if lang == 'en' else "日语学习"
    title = f"{title_prefix} - {source_info['name']} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    item_xml = f"""
    <item>
        <title>{title}</title>
        <description><![CDATA[{html_clean}]]></description>
        <pubDate>{now}</pubDate>
        <guid>{datetime.now().timestamp()}</guid>
    </item>"""

    try:
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
        logger.info(f"RSS 条目已写入: {title}")
    except Exception as e:
        logger.error(f"写入 RSS 文件失败: {e}")
        raise

# ======================= 主流程 =======================
def main():
    logger.info(f"--- 任务启动: {datetime.now()} ---")
    try:
        # 1. 决定本次语言
        lang, signal_value = get_signal_language()
        logger.info(f"信号量={signal_value} -> 推送语言: {'英语' if lang == 'en' else '日语'}")

        cfg = LANG_CONFIG[lang]
        input_file = cfg['input_file']
        progress_file = cfg['progress_file']
        batch_size = cfg['batch_size']

        # 2. 检查输入文件
        if not os.path.exists(input_file):
            logger.error(f"输入文件缺失: {input_file}")
            return

        # 3. 读取 CSV
        try:
            df = pd.read_csv(input_file)
            logger.info(f"成功读取 {input_file}，数据形状 {df.shape}")
        except Exception as e:
            logger.error(f"读取 CSV 失败: {e}")
            return

        if df.empty:
            logger.warning(f"{input_file} 是空文件，无数据可推送")
            return

        # 检查必要列
        required_cols = ['来源', '内容']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"{input_file} 缺少必需列: {col}")
                return

        # 清洗数据：填充缺失值并转为字符串
        df['内容'] = df['内容'].fillna('').astype(str)
        df['来源'] = df['来源'].fillna('未知来源').astype(str)

        total_rows = len(df)
        start_idx = get_last_index(progress_file)

        if start_idx >= total_rows:
            logger.info(f"{input_file} 已全部处理完毕 (索引 {start_idx} >= 总行数 {total_rows})，本次无推送")
            return

        # 4. 基于来源的动态批次切分
        first_source = df.at[start_idx, '来源']
        logger.info(f"批次起始行 {start_idx}，来源: '{first_source}'")

        end_idx = start_idx
        while end_idx < total_rows:
            # 如果来源变化，立即停止（不包含该变化行）
            if df.at[end_idx, '来源'] != first_source:
                logger.info(f"第 {end_idx} 行来源变化 ({first_source} -> {df.at[end_idx, '来源']})，批次截断")
                break
            # 如果已达到批大小上限，停止
            if end_idx - start_idx >= batch_size:
                break
            end_idx += 1

        batch = df.iloc[start_idx:end_idx]
        if batch.empty:
            logger.error("抽取的批次为空，检查逻辑")
            return

        logger.info(f"本批次: 行{start_idx}~{end_idx-1} (共{len(batch)}行)，来源: '{first_source}'")

        # 5. 拼接内容
        input_text = "\n".join(batch['内容'].tolist())

        # 6. 阅读时间估算
        minutes, marker = calculate_reading_time(input_text, lang)
        logger.info(f"预估阅读时间: {minutes} 分钟，文本长度 {len(input_text)} 字符")

        # 7. 调用 Gemini API（带详细错误捕获）
        system_instruction = load_system_instruction()
        try:
            client = genai.Client()
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=input_text,
                config={
                    'system_instruction': system_instruction,
                    'temperature': 0.7
                }
            )
            if not response.text:
                logger.error("Gemini API 返回空文本，本次推送放弃")
                return
            logger.info("Gemini API 调用成功，返回内容长度 %d", len(response.text))
        except Exception as e:
            logger.error(f"Gemini API 调用异常: {e}")
            logger.debug(traceback.format_exc())
            return  # 不更新进度，下次重试同一批

        # 8. 写入 RSS
        source_info = {
            'name': first_source,
            'time_marker': marker
        }
        update_rss(response.text, source_info, lang)

        # 9. 更新进度（仅在全部成功后）
        update_progress(progress_file, end_idx)
        logger.info(f"进度已更新至 {end_idx}，任务成功完成\n")

    except Exception as e:
        logger.error("未捕获的顶级异常导致程序崩溃")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

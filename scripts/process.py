import os
import re
import json
from pathlib import Path
from google import genai

# 初始化最新版 Gemini 客户端
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_ID = "gemini-1.5-flash"

def smart_split(text, chunk_size=10):
    # 移除可能存在的字幕时间戳
    text = re.sub(r'\d+\n\d{2}:\d{2}.*', '', text) 
    sentences = re.split(r'(?<=[.!?。！？])\s*', text.strip())
    sentences = [s for s in sentences if s.strip()]
    return [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

def get_prompt(lang_code):
    path = Path(f'prompts/{lang_code}.json')
    if not path.exists():
        # 如果找不到对应语言，回退到默认英语配置
        return {
            "system_prompt": "You are a language learning assistant.",
            "user_prompt_template": "Analyze the following {language} text:\n{text}",
            "language_name": "English"
        }
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    input_dir = Path("input")
    html_dir = Path("html_files")
    html_dir.mkdir(exist_ok=True)

    # 处理逻辑
    for file_path in input_dir.glob("*.txt"):
        print(f">>> 正在处理文件: {file_path.name}")
        
        lang_code = "ja" if "ja" in file_path.name.lower() else "en"
        config = get_prompt(lang_code)
        
        # 核心修复：检查 JSON 键值是否存在
        sys_p = config.get('system_prompt', 'Analyze text for language learning.')
        user_p_temp = config.get('user_prompt_template', 'Text: {text}')
        lang_name = config.get('language_name', 'Foreign Language')

        with open(file_path, "r", encoding="utf-8") as f:
            chunks = smart_split(f.read())

        # 读取进度
        progress_file = Path('progress.txt')
        current_idx = int(progress_file.read_text()) if progress_file.exists() else 1

        for chunk in chunks:
            # 构造最终提示词
            prompt_text = user_p_temp.format(language=lang_name, text=chunk)
            
            try:
                # 使用最新 SDK 的生成方法
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=prompt_text,
                    config={
                        "system_instruction": sys_p,
                        "response_mime_type": "text/plain" # 确保返回纯文本便于正则提取
                    }
                )
                
                # 提取 HTML 表格
                html_match = re.search(r"<table>.*?</table>", response.text, re.DOTALL)
                if html_match:
                    output_file = html_dir / f"{current_idx:0>3}.html"
                    output_file.write_text(html_match.group(0), encoding="utf-8")
                    print(f"Successfully generated: {output_file.name}")
                    current_idx += 1
            except Exception as e:
                print(f"!!! AI Processing Error: {e}")

        # 更新进度并清理已处理文件
        progress_file.write_text(str(current_idx))
        file_path.unlink() 

if __name__ == "__main__":
    main()

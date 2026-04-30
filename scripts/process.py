import os
import re
import json
import google.generativeai as genai
from pathlib import Path

# --- 配置区 ---
genai.configure(api_key=os.environ["AIzaSyBOogDSMLy-vBNfembDrmrGfUfOJUJNtCI"])
model = genai.GenerativeModel('gemini-1.5-flash')

def smart_split(text, chunk_size=10):
    """
    针对不同来源的智能切分逻辑
    """
    # 1. 预处理：如果是字幕文件，简单清理时间戳（可选）
    text = re.sub(r'\d+\n\d{2}:\d{2}.*', '', text) 
    
    # 2. 分句正则：支持中英日标点
    # 匹配 . ! ? (英文) 和 。 ！ ？ (日文)
    sentences = re.split(r'(?<=[.!?。！？])\s*', text.strip())
    sentences = [s for s in sentences if s.strip()]
    
    return [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

def get_prompt(lang_code):
    with open(f'prompts/{lang_code}.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    input_dir = Path("input")
    if not input_dir.exists(): return

    # 遍历 input 文件夹下的所有 txt 文件
    for file_path in input_dir.glob("*.txt"):
        print(f"正在处理: {file_path.name}")
        
        # 简单判断语言：根据文件名包含 'ja' 或 'en'，默认 en
        lang_code = "ja" if "ja" in file_path.name.lower() else "en"
        config = get_prompt(lang_code)
        
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        chunks = smart_split(full_text)
        
        # 读取当前序号
        if os.path.exists('progress.txt'):
            with open('progress.txt', 'r') as f:
                current_idx = int(f.read().strip())
        else:
            current_idx = 1

        for chunk in chunks:
            final_prompt = f"{config['system_prompt']}\n\n{config['user_prompt_template'].format(language=config['language_name'], text=chunk)}"
            
            try:
                response = model.generate_content(final_prompt)
                # 提取 HTML 表格代码块
                html_match = re.search(r"<table>.*?</table>", response.text, re.DOTALL)
                if html_match:
                    output_file = f"html_files/{current_idx:0>3}.html"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(html_match.group(0))
                    print(f"已生成: {output_file}")
                    current_idx += 1
            except Exception as e:
                print(f"调用 AI 出错: {e}")

        # 更新进度
        with open('progress.txt', 'w') as f:
            f.write(str(current_idx))
            
        # 处理完后移动或删除原文件，防止下次重复处理
        file_path.unlink() 

if __name__ == "__main__":
    main()

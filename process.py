import os
import google.generativeai as genai
import pandas as pd
from datetime import datetime
import google.generativeai as genai

# 定义你刚才那个复杂的 Prompt
# 我们把 JSON 逻辑转义为一段清晰的描述性文字
SYSTEM_PROMPT = """
你是一个精通英语和日语的外语教学专家。
任务：批量解析用户输入的文本，输出一个符合 Anki 导入格式的 HTML 表格。

要求：
1. 识别语言：自动判断输入是英文还是日文，并据此调整解析深度。
2. 结构化输出：严格按照以下 HTML 结构输出，包含：序号、阅读解析、词汇、中文翻译、背景注解。
3. 阅读解析：使用斜杠/进行分段划分词组短语，使结构透明。
4. 词汇：仅列出重点词汇及中文含义，使用 <br> 换行。
5. 禁止废话：只输出代码块内的 <table> 结构，严禁输出任何 Markdown 解释或其他文字。

HTML 模板参考：
<table border='1' style='border-collapse: collapse; width: 100%;'>
  <thead>
    <tr style='background-color: #f2f2f2;'>
      <th>序号</th><th>阅读解析</th><th>词汇</th><th>中文翻译</th><th>背景注解</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>[序号]</td>
      <td>[阅读解析内容]</td>
      <td>单词1：解释 <br> 单词2：解释</td>
      <td>[中文翻译]</td>
      <td>[背景注解]</td>
    </tr>
  </tbody>
</table>
"""

# 初始化模型时直接注入 Prompt
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash', # 或者 'gemini-1.5-pro'
    system_instruction=SYSTEM_PROMPT 
)

def get_ai_response(text_batch):
    # 调用时只需要传入纯文本
    response = model.generate_content(text_batch)
    return response.text
# 1. 配置 API (从环境变量读取，保证安全)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

BATCH_SIZE = 20 # 每次处理的行数

def process_sentences():
    # 读取进度
    if os.path.exists("last_index.txt"):
        with open("last_index.txt", "r") as f:
            start_idx = int(f.read().strip())
    else:
        start_idx = 0

    # 读取输入文件
    df = pd.read_csv("input.csv")
    
    # 截取本次要处理的行
    end_idx = start_idx + BATCH_SIZE
    batch_df = df.iloc[start_idx:end_idx]
    
    if batch_df.empty:
        print("所有内容已处理完毕。")
        return

    # 拼接文本给 AI
    input_text = "\n".join(batch_df['内容'].astype(str).tolist())
    
    # 获取 Prompt 并调用 API
    # 这里引用你之前定义的 Prompt 逻辑
    prompt = f"请解析以下文本：\n{input_text}" 
    
    response = model.generate_content(prompt)
    html_content = response.text.replace("```html", "").replace("```", "")

    # 更新 RSS (简化逻辑：直接更新 feed.xml 里的项)
    update_rss(html_content)

    # 记录进度
    with open("last_index.txt", "w") as f:
        f.write(str(end_idx))

def update_rss(new_content):
    # 这里可以用简单的字符串操作或 xml 库来维护 feed.xml
    # 每次运行生成一个新的 <item> 放入 RSS 头部
    pass

if __name__ == "__main__":
    process_sentences()

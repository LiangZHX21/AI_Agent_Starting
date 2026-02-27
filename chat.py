import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 对话历史，这就是 AI 的记忆
history = []

def chat(user_input: str) -> str:
    # 把用户输入加入历史
    history.append({
        "role": "user",
        "content": user_input
    })
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个helpful assistant"}
        ] + history  # 每次把完整历史都带上
    )
    
    reply = response.choices[0].message.content
    
    # 把 AI 回复也加入历史
    history.append({
        "role": "assistant",
        "content": reply
    })
    
    return reply

# 模拟多轮对话
print(chat("我叫Zephyr，我是一名.NET开发工程师"))
print("---")
print(chat("我叫什么名字？做什么工作的？"))
print("---")
print(chat("根据我的背景，你觉得我学Python需要多久？"))
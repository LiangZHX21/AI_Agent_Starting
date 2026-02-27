import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 定义工具，告诉 AI 有哪些工具可以用
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取某个城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，比如北京、上海"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，比如 2+3*4"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# 工具的实际实现
def get_weather(city: str) -> str:
    # 实际项目这里调真实天气 API，现在先模拟
    return f"{city}今天晴天，25度"

def calculate(expression: str) -> str:
    result = eval(expression)
    return str(result)

# 工具分发，AI 要调哪个就执行哪个
def run_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "get_weather":
        return get_weather(**tool_input)
    elif tool_name == "calculate":
        return calculate(**tool_input)

# Agent 主循环
def run_agent(user_input: str):
    messages = [
        {"role": "system", "content": "你是一个helpful assistant"},
        {"role": "user", "content": user_input}
    ]
    
    print(f"用户：{user_input}")
    
    while True:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools
        )
        
        choice = response.choices[0]

        print(f"AI返回结果：{response}")
        
        # AI 直接回答，不需要工具，结束循环
        if choice.finish_reason == "stop":
            print(f"AI：{choice.message.content}")
            break
        
        # AI 决定调用工具
        if choice.finish_reason == "tool_calls":
            # 把 AI 的决定加入历史
            messages.append(choice.message)
            
            # 执行 AI 要求的每个工具
            for tool_call in choice.message.tool_calls:
                tool_name = tool_call.function.name
                tool_input = json.loads(tool_call.function.arguments)
                
                print(f"调用工具：{tool_name}，参数：{tool_input}")
                result = run_tool(tool_name, tool_input)
                print(f"工具结果：{result}")
                
                # 把工具结果返回给 AI
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            # 继续循环，AI 根据工具结果生成最终回答

# 测试
run_agent("北京今天天气怎么样？另外帮我算一下 (100+200)*3 等于多少")
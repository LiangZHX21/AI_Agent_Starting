import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

@tool
def get_weather(city: str) -> str:
    """当用户询问某个城市的天气、温度、是否带伞时调用此工具"""
    return f"{city}今天晴天，25度"

@tool
def calculate(expression: str) -> str:
    """当用户需要计算数学表达式时调用此工具，输入表达式比如 2+3*4"""
    return str(eval(expression))

tools = [get_weather, calculate]
llm_with_tools = llm.bind_tools(tools)

def run_tool(tool_name: str, tool_input: dict) -> str:
    tool_map = {t.name: t for t in tools}
    return tool_map[tool_name].invoke(tool_input)

# ==============================
# 关键改动：history 移到外面
# 整个对话共享同一份历史
# ==============================
history = [
    SystemMessage(content="""
        你是一个智能助手，可以查天气和做计算。
        你能记住用户说过的所有内容。
    """)
]

def chat(user_input: str):
    # 用户输入加入历史
    history.append(HumanMessage(content=user_input))

    print(f"用户：{user_input}")

    while True:
        response = llm_with_tools.invoke(history)
        history.append(response)

        # AI 直接回答
        if not response.tool_calls:
            print(f"AI：{response.content}")
            break

        # AI 调工具
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]

            print(f"调用工具：{tool_name}，参数：{tool_input}")
            result = run_tool(tool_name, tool_input)
            print(f"工具结果：{result}")

            history.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            ))

# 测试多轮对话
chat("我叫Zephyr，我是.NET开发工程师")
print("---")
chat("我叫什么名字？做什么工作的？")
print("---")
chat("北京天气怎么样？")
print("---")
chat("我之前问的是哪个城市的天气？")  # 考验记忆
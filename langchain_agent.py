import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

load_dotenv()

# 初始化模型
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

# 把工具绑定到模型
llm_with_tools = llm.bind_tools(tools)

# 工具分发
def run_tool(tool_name: str, tool_input: dict) -> str:
    tool_map = {t.name: t for t in tools}
    return tool_map[tool_name].invoke(tool_input)

# Agent 主循环
def run_agent(user_input: str):
    messages = [
        SystemMessage(content="你是一个helpful assistant"),
        HumanMessage(content=user_input)
    ]

    print(f"用户：{user_input}")

    while True:
        # Call API
        print(f"call API with messages: {messages}")
        response = llm_with_tools.invoke(messages)
        print(f"call API response: {response}")
        messages.append(response)

        # AI 直接回答，不需要工具
        if not response.tool_calls:
            print(f"AI：{response.content}")
            break

        # AI 要调工具
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]

            print(f"调用工具：{tool_name}，参数：{tool_input}")
            result = run_tool(tool_name, tool_input)
            print(f"工具结果：{result}")

            messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            ))
        # 继续循环，AI 根据工具结果生成最终回答

# 测试
run_agent("北京天气怎么样？另外 (100+200)*3 等于多少")
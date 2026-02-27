import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ==============================
# 第一步：准备知识库文档
# 实际项目这里可以读取 txt/pdf/word
# ==============================
documents = [
    """
    公司退款政策：
    所有商品支持7天无理由退款。
    退款申请需在购买后7天内提交。
    退款金额将在3个工作日内原路返回。
    特价商品不支持退款。
    """,
    """
    公司工作时间：
    周一到周五 9:00-18:00。
    周六周日休息。
    节假日另行通知。
    """,
    """
    产品价格：
    基础版：99元/月
    专业版：299元/月
    企业版：999元/月
    所有版本支持免费试用7天。
    """
]

# ==============================
# 第二步：把文档切块
# 文档太长要切成小块，方便搜索
# ==============================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,    # 每块最多200字
    chunk_overlap=20   # 块之间重叠20字，防止信息被切断
)
chunks = splitter.create_documents(documents)

# ==============================
# 第三步：向量化存储
# 把文字转成向量，支持语义搜索
# ==============================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)

# ==============================
# 第四步：把知识库搜索变成工具
# ==============================
@tool
def search_knowledge(query: str) -> str:
    """当用户询问公司政策、产品价格、工作时间等问题时调用此工具"""
    docs = vectorstore.similarity_search(query, k=2)  # 找最相关的2段
    return "\n".join([doc.page_content for doc in docs])

@tool
def get_weather(city: str) -> str:
    """当用户询问某个城市天气时调用此工具"""
    return f"{city}今天晴天，25度"

tools = [search_knowledge, get_weather]
llm_with_tools = llm.bind_tools(tools)

def run_tool(tool_name: str, tool_input: dict) -> str:
    tool_map = {t.name: t for t in tools}
    return tool_map[tool_name].invoke(tool_input)

history = [
    SystemMessage(content="""
        你是公司的智能客服助手。
        回答用户问题时必须先调用 search_knowledge 工具查询公司资料。
        不能凭空捏造答案。
    """)
]

def chat(user_input: str):
    history.append(HumanMessage(content=user_input))
    print(f"用户：{user_input}")

    while True:
        response = llm_with_tools.invoke(history)
        history.append(response)
        print(f"AI返回结果：{response}")

        if not response.tool_calls:
            print(f"AI：{response.content}")
            break

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

# 测试
chat("我买东西有没有售后服务？")
print("---")
chat("专业版多少钱？")
print("---")
chat("你们周末上班吗？")

## 整体流程
# ```
# 用户提问
#     ↓
# AI 决定调 search_knowledge 工具
#     ↓
# 工具去向量库搜索最相关的文档片段
#     ↓
# 把搜索结果塞给 AI
#     ↓
# AI 根据真实文档内容回答
# ```

# ---

# ## 向量搜索是什么意思
# ```
# 普通搜索（关键词匹配）：
# 用户问"退货"→ 必须文档里有"退货"这个词才能找到

# 向量搜索（语义匹配）：
# 用户问"退货"→ 能找到包含"退款"的文档
# 因为"退货"和"退款"语义相近
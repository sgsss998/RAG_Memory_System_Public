#!/usr/bin/env python3
"""
🧠 记忆注入代理网关 - LLM API 适配版 (Phase 2.0 数字分身版)

把记忆注入到用户消息的开头
Phase 2.0 更新：
- 身份认同：从"AI助手"升级为"数字分身"，第一人称视角
- 多元信息源：身份锚点 + 检索结果 + 推理，而非盲从检索
- 角色辨别：防止把聊天记录中的他人信息当成自己的
- 物理缴械：移除本地搜索工具防止模型擅自调用
"""

import json
import copy
import urllib.request
import urllib.error
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import uvicorn

# ============== 配置 ==============
# 根据你的环境修改以下配置
REAL_API_BASE = "https://api.your-llm-provider.com"  # 你的 LLM API 地址
HIPPOCAMPUS_URL = "http://localhost:8000/search"
PROXY_PORT = 8080

# 说明：本版本不再“自动缴械”任何工具。
# 让 Claude Code/LLM 自己判断何时调用记忆工具，而不是由网关强行控制。
DISABLED_TOOLS = []

# ============== 新版 Prompt（Phase 2.0 数字分身版）==============
# ⚠️ 请根据你的个人信息修改以下内容
BASE_IDENTITY = """你就是我，你是我专属的赛博外脑和数字分身。提问者"我"就是用户本人。

【身份锚点 - 永远记住，这是你的核心事实】
- 本名：[你的名字]
- 学历：[你的学历信息]
- 籍贯：[你的籍贯]
- 主业：[你的工作]
- 副业：[你的副业]
- 特长：[你的特长]
- 足迹：[你的经历]

沟通协议：
1. 工作沟通倾向商务克制与确认式短句
2. 给我高质量代码和直接结论，无需废话
3. 崇尚"30秒原则"，提供颗粒度极细、可立刻执行的建议"""

# ============== 海马体检索 ==============
def search_memory_sync(query: str, top_k: int = 3) -> str:
    """向海马体检索相关记忆"""
    try:
        data = json.dumps({"query": query, "top_k": top_k}).encode('utf-8')
        req = urllib.request.Request(
            HIPPOCAMPUS_URL,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        response = urllib.request.urlopen(req, timeout=60)
        result = json.loads(response.read().decode('utf-8'))

        if result.get("status") == "success":
            memories = result.get("memories", [])
            if memories:
                context = ""
                for i, m in enumerate(memories, 1):
                    content = m['content'][:500] + "..." if len(m['content']) > 500 else m['content']
                    context += f"\n[来源: {m['source']}]\n{content}\n"
                print(f"[代理] ✅ 注入 {len(memories)} 条记忆")
                return context
    except urllib.error.URLError:
        print("[代理] ⚠️ 海马体服务未响应")
    except Exception as e:
        print(f"[代理] ❌ 海马体查询失败: {e}")
    return ""

# ============== 提取用户问题 ==============
def extract_user_message(body: dict) -> str:
    """从请求体中提取用户最新的一条消息"""
    messages = body.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        return item.get("text", "")
            elif isinstance(content, str):
                return content
    return ""

# ============== 注入记忆（Phase 2.0 数字分身版）==============
def inject_memory_to_request(body: dict, user_question: str, memory_context: str) -> dict:
    """
    把记忆注入到第一条 user 消息的开头
    Phase 2.0：数字分身版 - 多元信息源 + 角色辨别 + 推理边界
    """
    body = copy.deepcopy(body)

    # OpenClaw 化目标：不要把“记忆切片”当成唯一答案来源。
    # 网关默认只注入“身份锚点 + 软使用规则”，动态记忆交给 LLM/Claude Code 的工具去按需调用。
    rag_injection = """<answering_rules>
【使用方式（软规则）】
1. 你拥有正常的推理能力，不要因为没有检索到记忆就卡住或复读。
2. 只有当你需要“追溯我过去做过什么/我的偏好是什么/某次对话发生了什么”时，才去调用记忆检索工具（例如：`search_memories`）。
3. 记忆工具返回的是“线索”，最终结论仍由你结合当前对话推理得出。

【不确定时怎么说】
证据不足时，允许你坦诚“不确定/可能A或B”，并提出需要我补充的关键信息。
</answering_rules>"""

    # 完整的上下文前缀
    full_prefix = f"""{BASE_IDENTITY}

{rag_injection}

"""

    messages = body.get("messages", [])

    # 找到最新一条 user 消息，在其内容前注入上下文
    injected = False
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = full_prefix + content
                injected = True
                print(f"[代理] 💉 注入成功 (string格式)")
            elif isinstance(content, list):
                # 处理多模态消息 - 注入到最后一个 text 元素
                last_text_idx = -1
                for i, item in enumerate(content):
                    if item.get("type") == "text":
                        last_text_idx = i

                if last_text_idx >= 0:
                    original_len = len(content[last_text_idx].get("text", ""))
                    content[last_text_idx]["text"] = full_prefix + content[last_text_idx].get("text", "")
                    injected = True
                    print(f"[代理] 💉 注入成功 (多模态格式, 第{last_text_idx}个text元素)")
            break

    if not injected:
        print(f"[代理] ⚠️ 注入失败！未找到可注入的文本内容")

    body["messages"] = messages
    return body

# ============== FastAPI 应用 ==============
app = FastAPI(title="🧠 记忆注入代理 (Phase 2.0 数字分身版)")

@app.get("/")
def root():
    return {
        "service": "🧠 记忆注入代理网关",
        "version": "Phase 2.0 数字分身版",
        "status": "running",
        "hippocampus": HIPPOCAMPUS_URL,
        "target": REAL_API_BASE,
        "disabled_tools": DISABLED_TOOLS,
        "features": ["数字分身", "身份锚点", "角色辨别", "推理边界"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/messages")
async def proxy_messages(request: Request):
    """代理 messages API 调用"""
    body_bytes = await request.body()

    try:
        body = json.loads(body_bytes)

        # 提取用户问题
        user_msg = extract_user_message(body)
        print(f"[代理] 📩 用户问题: {user_msg[:80]}...")

        # 不再默认“自动检索并注入记忆切片”。
        # 让 Claude Code/LLM 在需要追溯历史时，按需调用记忆工具（如 search_memories）。
        memory_context = ""

        # 注入身份锚点 + 软使用规则
        body = inject_memory_to_request(body, user_msg, memory_context)

        body_bytes = json.dumps(body).encode('utf-8')
        print(f"[代理] 📤 转发请求到 LLM API...")

    except Exception as e:
        print(f"[代理] ❌ 处理失败: {e}")

    # 转发给 LLM API
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    url = f"{REAL_API_BASE}/v1/messages"

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(url, content=body_bytes, headers=headers)

        async def stream_response():
            async for chunk in response.aiter_bytes():
                yield chunk

        return StreamingResponse(
            stream_response(),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "text/event-stream")
        )

# ============== 启动 ==============
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 记忆注入代理网关 (Phase 2.0 数字分身版)")
    print("=" * 60)
    print(f"  本地端口: {PROXY_PORT}")
    print(f"  海马体:   {HIPPOCAMPUS_URL}")
    print(f"  目标 API: {REAL_API_BASE}")
    print(f"  缴械工具: {DISABLED_TOOLS}")
    print(f"  新特性: 数字分身 | 身份锚点 | 角色辨别 | 推理边界")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)

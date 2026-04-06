from unittest.mock import patch, MagicMock
import sys, os
# 让 import 找到 emotion_model 目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'emotion_model'))

from chat_memory import ChatMemory

def test_init_has_system_prompt():
    """初始化后 messages 只有 system prompt"""
    mem = ChatMemory("You are helpful.", max_tokens=3000)
    msgs = mem.get_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are helpful."

def test_add_user_input():
    """添加用户消息后，消息列表正确增长"""
    mem = ChatMemory("system", max_tokens=3000)
    mem.add_user_input("hello")
    msgs = mem.get_messages()
    assert len(msgs) == 2
    assert msgs[-1] == {"role": "user", "content": "hello"}

def test_add_assistant_response():
    mem = ChatMemory("system", max_tokens=3000)
    mem.add_assistant_response("hi there")
    msgs = mem.get_messages()
    assert msgs[-1] == {"role": "assistant", "content": "hi there"}

def test_message_order_preserved():
    """多轮对话顺序正确"""
    mem = ChatMemory("system", max_tokens=3000)
    mem.add_user_input("a")
    mem.add_assistant_response("b")
    mem.add_user_input("c")
    roles = [m["role"] for m in mem.get_messages()]
    assert roles == ["system", "user", "assistant", "user"]

def test_estimate_token_count():
    """token 估算 ≈ 字符数 / 4"""
    mem = ChatMemory("test", max_tokens=3000)  # "test" = 4 chars → ~1 token
    mem.add_user_input("a" * 400)              # 400 chars → ~100 tokens
    # system(1) + user(100) = ~101
    count = mem._estimate_token_count()
    assert 100 <= count <= 110

def test_trim_calls_openai_when_over_limit():
    """超过 max_tokens 时触发裁剪，调用 OpenAI 做摘要"""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Summary of conversation"

    import chat_memory
    original = chat_memory.openai
    chat_memory.openai = MagicMock()
    chat_memory.openai.chat.completions.create.return_value = mock_response

    try:
        mem = ChatMemory("system", max_tokens=10)  # 故意设很小
        mem.add_user_input("x" * 200)              # 必然超限

        # 裁剪后只剩 system + summary
        msgs = mem.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["content"] == "Summary of conversation"
    finally:
        chat_memory.openai = original

def test_trim_handles_api_failure():
    """OpenAI 调用失败时，使用 fallback 摘要"""
    import chat_memory
    original = chat_memory.openai
    chat_memory.openai = MagicMock()
    chat_memory.openai.chat.completions.create.side_effect = Exception("API down")

    try:
        mem = ChatMemory("system", max_tokens=10)
        mem.add_user_input("x" * 200)
        msgs = mem.get_messages()
        assert "[Summary unavailable" in msgs[1]["content"]
    finally:
        chat_memory.openai = original

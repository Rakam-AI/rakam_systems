import os

import pytest

from rakam_systems_agent.components.chat_history.json_chat_history import JSONChatHistory
from rakam_systems_agent.components.chat_history.sql_chat_history import SQLChatHistory

try:
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        UserPromptPart,
        TextPart,
    )
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False

pydantic_ai_required = pytest.mark.skipif(
    not PYDANTIC_AI_AVAILABLE, reason="pydantic-ai not installed"
)


def make_pydantic_messages():
    """Return a minimal user+assistant pydantic-ai message pair."""
    return [
        ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ModelResponse(parts=[TextPart(content="Hi there!")]),
    ]


def make_msg(role="user", content="Hello"):
    return {"role": role, "content": content}


@pytest.fixture
def json_history(tmp_path):
    path = str(tmp_path / "chat.json")
    return JSONChatHistory(storage_path=path)


def test_json_init_defaults(json_history):
    assert json_history.auto_save is True
    assert json_history.indent == 4


def test_json_add_message(json_history):
    json_history.add_message("chat1", make_msg())
    msgs = json_history.get_chat_history("chat1")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_json_add_multiple_messages(json_history):
    json_history.add_message("chat1", make_msg("user", "Hi"))
    json_history.add_message("chat1", make_msg("assistant", "Hello!"))
    msgs = json_history.get_chat_history("chat1")
    assert len(msgs) == 2


def test_json_get_empty_chat(json_history):
    msgs = json_history.get_chat_history("nonexistent")
    assert msgs == []


def test_json_set_messages(json_history):
    messages = [make_msg("user", "A"), make_msg("assistant", "B")]
    json_history.set_messages("chat2", messages)
    result = json_history.get_chat_history("chat2")
    assert len(result) == 2
    assert result[0]["content"] == "A"


def test_json_set_messages_replaces(json_history):
    json_history.add_message("chat3", make_msg("user", "old"))
    json_history.set_messages("chat3", [make_msg("user", "new")])
    msgs = json_history.get_chat_history("chat3")
    assert len(msgs) == 1
    assert msgs[0]["content"] == "new"


def test_json_get_all_chat_ids(json_history):
    json_history.add_message("a", make_msg())
    json_history.add_message("b", make_msg())
    ids = json_history.get_all_chat_ids()
    assert "a" in ids
    assert "b" in ids


def test_json_delete_chat(json_history):
    json_history.add_message("del_me", make_msg())
    result = json_history.delete_chat_history("del_me")
    assert result is True
    assert json_history.get_chat_history("del_me") == []


def test_json_delete_nonexistent_chat(json_history):
    result = json_history.delete_chat_history("ghost")
    assert result is False


def test_json_clear_all(json_history):
    json_history.add_message("a", make_msg())
    json_history.add_message("b", make_msg())
    json_history.clear_all()
    assert json_history.get_all_chat_ids() == []


def test_json_get_readable_chat_history(json_history):
    json_history.add_message("chat", {"role": "user", "content": "Hi", "timestamp": "2024-01-01"})
    json_history.add_message("chat", {"role": "assistant", "content": "Hello!"})
    readable = json_history.get_readable_chat_history("chat")
    assert len(readable) == 2
    assert readable[0]["from"] == "user"
    assert readable[0]["message"] == "Hi"
    assert readable[0]["timestamp"] == "2024-01-01"
    assert readable[1]["from"] == "assistant"
    assert "timestamp" not in readable[1]


def test_json_readable_skips_system(json_history):
    json_history.add_message("chat", {"role": "system", "content": "Setup"})
    json_history.add_message("chat", {"role": "user", "content": "Hi"})
    readable = json_history.get_readable_chat_history("chat")
    assert len(readable) == 1
    assert readable[0]["from"] == "user"


def test_json_persists_to_file(tmp_path):
    path = str(tmp_path / "persist.json")
    h1 = JSONChatHistory(storage_path=path)
    h1.add_message("x", make_msg("user", "persisted"))
    h2 = JSONChatHistory(storage_path=path)
    msgs = h2.get_chat_history("x")
    assert len(msgs) == 1
    assert msgs[0]["content"] == "persisted"


def test_json_reload(json_history):
    json_history.add_message("r", make_msg("user", "reload test"))
    json_history.reload()
    msgs = json_history.get_chat_history("r")
    assert len(msgs) == 1


def test_json_save_manual(tmp_path):
    path = str(tmp_path / "manual.json")
    h = JSONChatHistory(storage_path=path, config={"auto_save": False})
    h.add_message("m", make_msg())
    h.save()
    assert os.path.exists(path)


def test_json_handles_empty_file(tmp_path):
    path = str(tmp_path / "empty.json")
    open(path, "w").close()
    h = JSONChatHistory(storage_path=path)
    assert h.get_all_chat_ids() == []


@pytest.fixture
def sql_history(tmp_path):
    db_path = str(tmp_path / "chat.db")
    return SQLChatHistory(db_path=db_path)


def test_sql_init(sql_history):
    assert sql_history.db_path.endswith("chat.db")


def test_sql_add_message(sql_history):
    sql_history.add_message("chat1", make_msg())
    msgs = sql_history.get_chat_history("chat1")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_sql_add_multiple_messages(sql_history):
    sql_history.add_message("chat1", make_msg("user", "Hi"))
    sql_history.add_message("chat1", make_msg("assistant", "Hello!"))
    msgs = sql_history.get_chat_history("chat1")
    assert len(msgs) == 2
    assert msgs[0]["content"] == "Hi"
    assert msgs[1]["content"] == "Hello!"


def test_sql_get_empty_chat(sql_history):
    msgs = sql_history.get_chat_history("nonexistent")
    assert msgs == []


def test_sql_set_messages(sql_history):
    messages = [make_msg("user", "A"), make_msg("assistant", "B")]
    sql_history.set_messages("chat2", messages)
    result = sql_history.get_chat_history("chat2")
    assert len(result) == 2
    assert result[1]["content"] == "B"


def test_sql_set_messages_replaces(sql_history):
    sql_history.add_message("chat3", make_msg("user", "old"))
    sql_history.set_messages("chat3", [make_msg("user", "new")])
    msgs = sql_history.get_chat_history("chat3")
    assert len(msgs) == 1
    assert msgs[0]["content"] == "new"


def test_sql_get_all_chat_ids(sql_history):
    sql_history.add_message("a", make_msg())
    sql_history.add_message("b", make_msg())
    ids = sql_history.get_all_chat_ids()
    assert "a" in ids
    assert "b" in ids


def test_sql_delete_chat(sql_history):
    sql_history.add_message("del_me", make_msg())
    result = sql_history.delete_chat_history("del_me")
    assert result is True
    assert sql_history.get_chat_history("del_me") == []


def test_sql_delete_nonexistent(sql_history):
    result = sql_history.delete_chat_history("ghost")
    assert result is False


def test_sql_clear_all(sql_history):
    sql_history.add_message("a", make_msg())
    sql_history.add_message("b", make_msg())
    sql_history.clear_all()
    assert sql_history.get_all_chat_ids() == []


def test_sql_get_readable_chat_history(sql_history):
    sql_history.add_message("chat", {"role": "user", "content": "Hi", "timestamp": "2024-01-01"})
    sql_history.add_message("chat", {"role": "assistant", "content": "Hello!"})
    readable = sql_history.get_readable_chat_history("chat")
    assert len(readable) == 2
    assert readable[0]["from"] == "user"
    assert readable[1]["from"] == "assistant"


def test_sql_readable_skips_system(sql_history):
    sql_history.add_message("chat", {"role": "system", "content": "Setup"})
    sql_history.add_message("chat", {"role": "user", "content": "Hi"})
    readable = sql_history.get_readable_chat_history("chat")
    assert len(readable) == 1
    assert readable[0]["from"] == "user"


def test_sql_message_order_preserved(sql_history):
    for i in range(5):
        sql_history.add_message("ordered", make_msg("user", str(i)))
    msgs = sql_history.get_chat_history("ordered")
    assert [m["content"] for m in msgs] == ["0", "1", "2", "3", "4"]


def test_sql_config_db_path(tmp_path):
    db_path = str(tmp_path / "config_test.db")
    h = SQLChatHistory(config={"db_path": db_path})
    assert h.db_path == db_path



@pydantic_ai_required
def test_json_get_message_history_empty_returns_none(json_history):
    """get_message_history returns None when no history exists for the chat."""
    assert json_history.get_message_history("no-such-chat") is None


@pydantic_ai_required
def test_json_save_and_get_message_history_roundtrip(json_history):
    """save_messages + get_message_history preserves message count and types."""
    messages = make_pydantic_messages()
    json_history.save_messages("chat1", messages)

    result = json_history.get_message_history("chat1")

    assert result is not None
    assert len(result) == 2
    assert isinstance(result[0], ModelRequest)
    assert isinstance(result[1], ModelResponse)


@pydantic_ai_required
def test_json_save_messages_overwrites_previous(json_history):
    """save_messages replaces any prior history for the same chat_id."""
    json_history.save_messages("chat1", make_pydantic_messages())
    json_history.save_messages("chat1", make_pydantic_messages()[:1])

    result = json_history.get_message_history("chat1")
    assert len(result) == 1


@pydantic_ai_required
def test_json_message_history_content_preserved(json_history):
    """Roundtrip preserves user prompt content."""
    messages = make_pydantic_messages()
    json_history.save_messages("chat1", messages)

    result = json_history.get_message_history("chat1")
    assert result[0].parts[0].content == "Hello"
    assert result[1].parts[0].content == "Hi there!"


@pydantic_ai_required
def test_sql_get_message_history_empty_returns_none(sql_history):
    """get_message_history returns None when no history exists for the chat."""
    assert sql_history.get_message_history("no-such-chat") is None


@pydantic_ai_required
def test_sql_save_and_get_message_history_roundtrip(sql_history):
    """save_messages + get_message_history preserves message count and types."""
    messages = make_pydantic_messages()
    sql_history.save_messages("chat1", messages)

    result = sql_history.get_message_history("chat1")

    assert result is not None
    assert len(result) == 2
    assert isinstance(result[0], ModelRequest)
    assert isinstance(result[1], ModelResponse)


@pydantic_ai_required
def test_sql_save_messages_overwrites_previous(sql_history):
    """save_messages replaces any prior history for the same chat_id."""
    sql_history.save_messages("chat1", make_pydantic_messages())
    sql_history.save_messages("chat1", make_pydantic_messages()[:1])

    result = sql_history.get_message_history("chat1")
    assert len(result) == 1


@pydantic_ai_required
def test_sql_message_history_content_preserved(sql_history):
    """Roundtrip preserves user prompt content."""
    messages = make_pydantic_messages()
    sql_history.save_messages("chat1", messages)

    result = sql_history.get_message_history("chat1")
    assert result[0].parts[0].content == "Hello"
    assert result[1].parts[0].content == "Hi there!"

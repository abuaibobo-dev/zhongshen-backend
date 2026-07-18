"""
数据库模块 - SQLite 异步操作
管理用户、会话、消息的持久化存储
"""

import aiosqlite
import uuid
import os
from datetime import datetime
from typing import Optional, List, Dict

# 数据库文件路径（与 main.py 同级目录）
DB_PATH = os.path.join(os.path.dirname(__file__), "zhongshen.db")

# 全局数据库连接
_db: Optional[aiosqlite.Connection] = None


async def get_db() -> aiosqlite.Connection:
    """获取数据库连接（单例模式）"""
    global _db
    if _db is None:
        _db = await aiosqlite.connect(DB_PATH)
        _db.row_factory = aiosqlite.Row
        await _db.execute("PRAGMA journal_mode=WAL")  # 启用WAL模式提升并发性能
        await _db.execute("PRAGMA foreign_keys=ON")
        await init_tables(_db)
    return _db


async def init_tables(db: aiosqlite.Connection):
    """初始化数据库表结构"""
    await db.executescript("""
        -- 用户表
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );

        -- 会话表
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '新会话',
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        -- 消息表
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            model_used TEXT,
            tokens_used INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );

        -- 索引优化
        CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, created_at);
    """)
    await db.commit()


async def close_db():
    """关闭数据库连接"""
    global _db
    if _db is not None:
        await _db.close()
        _db = None


def generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4()).replace("-", "")[:16]


# ==================== 用户操作 ====================

async def get_or_create_user(username: str = "default_user") -> dict:
    """获取或创建用户，返回用户信息"""
    db = await get_db()
    cursor = await db.execute(
        "SELECT id, username, created_at FROM users WHERE username = ?",
        (username,)
    )
    row = await cursor.fetchone()
    if row:
        return {"id": row[0], "username": row[1], "created_at": row[2]}

    # 创建新用户
    user_id = generate_id()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await db.execute(
        "INSERT INTO users (id, username, created_at) VALUES (?, ?, ?)",
        (user_id, username, now)
    )
    await db.commit()
    return {"id": user_id, "username": username, "created_at": now}


# ==================== 会话操作 ====================

async def get_or_create_conversation(user_id: str, conversation_id: Optional[str] = None) -> dict:
    """获取或创建会话"""
    db = await get_db()

    if conversation_id:
        cursor = await db.execute(
            "SELECT id, title, created_at FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id)
        )
        row = await cursor.fetchone()
        if row:
            return {"id": row[0], "title": row[1], "created_at": row[2]}

    # 创建新会话
    conv_id = generate_id()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await db.execute(
        "INSERT INTO conversations (id, user_id, title, created_at) VALUES (?, ?, ?, ?)",
        (conv_id, user_id, "新会话", now)
    )
    await db.commit()
    return {"id": conv_id, "title": "新会话", "created_at": now}


# ==================== 消息操作 ====================

async def save_message(
    conversation_id: str,
    role: str,
    content: str,
    model_used: Optional[str] = None,
    tokens_used: int = 0
) -> dict:
    """保存消息到数据库"""
    db = await get_db()
    msg_id = generate_id()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    await db.execute(
        "INSERT INTO messages (id, conversation_id, role, content, model_used, tokens_used, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (msg_id, conversation_id, role, content, model_used, tokens_used, now)
    )
    await db.commit()
    return {
        "id": msg_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "model_used": model_used,
        "tokens_used": tokens_used,
        "created_at": now
    }


async def get_conversation_messages(conversation_id: str, limit: int = 50) -> List[dict]:
    """获取会话历史消息"""
    db = await get_db()
    cursor = await db.execute(
        "SELECT id, role, content, model_used, tokens_used, created_at "
        "FROM messages WHERE conversation_id = ? ORDER BY created_at ASC LIMIT ?",
        (conversation_id, limit)
    )
    rows = await cursor.fetchall()
    return [
        {
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "model_used": row[3],
            "tokens_used": row[4],
            "created_at": row[5]
        }
        for row in rows
    ]


async def update_conversation_title(conversation_id: str, title: str):
    """更新会话标题"""
    db = await get_db()
    await db.execute(
        "UPDATE conversations SET title = ? WHERE id = ?",
        (title, conversation_id)
    )
    await db.commit()

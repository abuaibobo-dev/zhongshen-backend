"""
数据模型定义 - Pydantic Schema
定义所有 API 请求和响应的数据结构
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


# ==================== 请求模型 ====================

class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(..., description="用户发送的消息", min_length=1, max_length=50000)
    conversation_id: Optional[str] = Field(None, description="会话ID，为空则创建新会话")
    force_model: Optional[str] = Field(None, description="强制指定模型，跳过路由引擎")


class ParallelChatRequest(BaseModel):
    """多模型并行请求（竞技场模式）"""
    message: str = Field(..., description="用户消息", min_length=1, max_length=50000)
    models: Optional[List[str]] = Field(None, description="指定模型列表，为空则使用全部文本模型")
    conversation_id: Optional[str] = Field(None, description="会话ID")


class DebateRequest(BaseModel):
    """群体辩论请求"""
    topic: str = Field(..., description="辩论话题", min_length=1, max_length=10000)
    models: Optional[List[str]] = Field(None, description="参与辩论的模型列表")
    rounds: int = Field(1, description="辩论轮数", ge=1, le=5)


# ==================== 响应模型 ====================

class ChatResponse(BaseModel):
    """单模型聊天响应"""
    reply: str = Field(..., description="模型回复内容")
    model_used: str = Field(..., description="实际使用的模型ID")
    role: str = Field(..., description="AI角色名称")
    routing_reason: str = Field(..., description="路由决策原因")
    tokens_used: int = Field(0, description="消耗token数")
    conversation_id: str = Field(..., description="会话ID")
    latency_ms: Optional[float] = Field(None, description="响应延迟（毫秒）")


class ModelResponse(BaseModel):
    """单个模型信息"""
    model_id: str = Field(..., description="模型ID")
    name: str = Field(..., description="模型名称")
    role: str = Field(..., description="角色名称")
    specialty: str = Field(..., description="擅长领域")
    icon: str = Field(..., description="角色图标/emoji")
    description: str = Field(..., description="角色描述")
    is_available: bool = Field(True, description="是否可用")
    category: str = Field("text", description="模型类别: text/image/video")


class ModelsResponse(BaseModel):
    """模型列表响应"""
    models: List[ModelResponse]
    total: int


class RoleCard(BaseModel):
    """角色卡片"""
    id: str
    name: str
    model_id: str
    icon: str
    specialty: str
    description: str
    color: str = Field("#6C5CE7", description="角色主题色")


class RolesResponse(BaseModel):
    """角色列表响应"""
    roles: List[RoleCard]
    total: int


class ParallelResult(BaseModel):
    """单个模型的并行结果"""
    model_id: str
    name: str
    role: str
    reply: str
    tokens_used: int
    latency_ms: Optional[float] = None


class ParallelResponse(BaseModel):
    """并行（竞技场）响应"""
    results: List[ParallelResult]
    topic: str
    conversation_id: str
    total_models: int


class DebateTurn(BaseModel):
    """单轮辩论结果"""
    round_num: int
    statements: List[dict]  # [{model_id, name, role, statement, tokens_used}]


class DebateResponse(BaseModel):
    """群体辩论响应"""
    topic: str
    debate_turns: List[DebateTurn]
    conversation_id: str
    total_rounds: int


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    version: str = "1.0.0"
    mode: str = Field(..., description="运行模式: live / mock")
    models_available: int
    timestamp: str

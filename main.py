"""
众神之域 (ZhongShen) - AI 模型聚合 App 后端
FastAPI 入口文件

启动方式：
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Swagger 文档：
    http://localhost:8000/docs
"""

import asyncio
import time
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# 导入自定义模块
from schemas import (
    ChatRequest, ChatResponse,
    ParallelChatRequest, ParallelResponse, ParallelResult,
    DebateRequest, DebateResponse, DebateTurn,
    ModelsResponse, ModelResponse,
    RolesResponse, RoleCard,
    HealthResponse
)
from models import (
    MODEL_REGISTRY, register_models, create_provider,
    get_running_mode, ModelConfig, MockProvider, OpenRouterProvider,
    AgnesImageProvider, AgnesVideoProvider
)
from router_engine import router
import database as db


# ==================== 应用初始化 ====================

app = FastAPI(
    title="众神之域 - AI模型聚合平台",
    description=(
        "输入需求 → 自动路由到最擅长的AI模型 → 返回结果\n\n"
        "**核心能力：**\n"
        "- 🧠 智能路由：自动识别意图并分发到最合适的模型\n"
        "- ⚡ 竞技场模式：多模型并行回答，让你选择最佳\n"
        "- 🎭 群体辩论：多模型各抒己见，碰撞思想火花\n"
        "- 🎨 全模态：文本 / 图片 / 视频生成"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 配置（开发阶段允许所有来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 启动时注册所有模型
register_models()


@app.on_event("startup")
async def startup_event():
    """应用启动事件：初始化数据库"""
    await db.get_db()


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件：关闭数据库连接"""
    await db.close_db()


# ==================== 核心 API ====================

@app.post("/api/chat", response_model=ChatResponse, tags=["核心对话"])
async def chat(request: ChatRequest, http_request: Request):
    """
    发送消息，自动路由到最合适的 AI 模型
    
    - 输入需求描述，路由引擎自动判断应该调用哪个模型
    - 支持 force_model 强制指定模型（跳过路由）
    - 自动创建或复用会话
    """
    try:
        # 1. 路由决策
        routing = router.route(request.message, request.force_model)
        model_config = MODEL_REGISTRY[routing.model_id]

        # 2. 获取/创建会话
        user = await db.get_or_create_user()
        conversation = await db.get_or_create_conversation(
            user["id"], request.conversation_id
        )
        conv_id = conversation["id"]

        # 3. 保存用户消息
        await db.save_message(conv_id, "user", request.message)

        # 4. 获取用户 API Key（从请求头）
        user_api_key = http_request.headers.get("X-API-Key", "")

        # 5. 调用模型
        provider = create_provider(model_config, api_key=user_api_key)
        messages = [{"role": "user", "content": request.message}]

        # 图像/视频模型走专门的生成方法
        if model_config.category == "image":
            start = time.time()
            result = await provider.generate(request.message)
        elif model_config.category == "video":
            start = time.time()
            result = await provider.generate(request.message)
        else:
            result = await provider.chat(model_config, messages)

        # 6. 如果调用失败，降级到 Mock
        if not result.success:
            mock = MockProvider()
            result = await mock.chat(model_config, messages)

        # 7. 保存助手回复
        await db.save_message(
            conv_id, "assistant", result.content,
            model_used=routing.model_id,
            tokens_used=result.tokens_used
        )

        # 8. 更新会话标题（首次对话时）
        if request.conversation_id is None:
            title = request.message[:30] + ("..." if len(request.message) > 30 else "")
            await db.update_conversation_title(conv_id, title)

        return ChatResponse(
            reply=result.content,
            model_used=routing.model_id,
            role=model_config.role,
            routing_reason=routing.reason,
            tokens_used=result.tokens_used,
            conversation_id=conv_id,
            latency_ms=result.latency_ms
        )

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"模型不存在: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@app.post("/api/chat/parallel", response_model=ParallelResponse, tags=["竞技场模式"])
async def parallel_chat(request: ParallelChatRequest, http_request: Request):
    """
    多模型并行回答同一问题（竞技场模式）
    
    - 同一问题发给多个模型，让它们各显神通
    - 用户可以对比不同模型的回答质量
    - 不指定 models 参数则使用全部文本模型
    """
    try:
        # 确定参与模型列表
        if request.models:
            model_ids = [mid for mid in request.models if mid in MODEL_REGISTRY]
        else:
            model_ids = router.get_all_text_models()

        if not model_ids:
            raise HTTPException(status_code=400, detail="没有可用的文本模型")

        # 获取/创建会话
        user = await db.get_or_create_user()
        conversation = await db.get_or_create_conversation(
            user["id"], request.conversation_id
        )
        conv_id = conversation["id"]

        # 保存用户消息
        await db.save_message(conv_id, "user", request.message)

        # 获取用户 API Key
        user_api_key = http_request.headers.get("X-API-Key", "")

        # 并行调用所有模型
        messages = [{"role": "user", "content": request.message}]
        tasks = []
        for mid in model_ids:
            cfg = MODEL_REGISTRY[mid]
            provider = create_provider(cfg, api_key=user_api_key)
            tasks.append(_call_model(provider, cfg, messages))

        results = await asyncio.gather(*tasks)

        # 保存每个模型的回复
        parallel_results = []
        for r in results:
            if r.success:
                await db.save_message(
                    conv_id, "assistant", r.content,
                    model_used=r.model_id,
                    tokens_used=r.tokens_used
                )
                cfg = MODEL_REGISTRY[r.model_id]
                parallel_results.append(ParallelResult(
                    model_id=r.model_id,
                    name=cfg.name,
                    role=cfg.role,
                    reply=r.content,
                    tokens_used=r.tokens_used,
                    latency_ms=r.latency_ms
                ))

        return ParallelResponse(
            results=parallel_results,
            topic=request.message[:50],
            conversation_id=conv_id,
            total_models=len(parallel_results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"并行调用失败: {str(e)}")


@app.post("/api/chat/debate", response_model=DebateResponse, tags=["群体辩论"])
async def debate(request: DebateRequest, http_request: Request):
    """
    群体辩论：多个模型对同一话题各抒己见
    
    - 模拟多位AI"神明"围绕同一话题展开辩论
    - 每个模型基于自己的"角色定位"给出观点
    - 支持多轮辩论
    """
    try:
        # 确定参与辩论的模型
        if request.models:
            model_ids = [mid for mid in request.models if mid in MODEL_REGISTRY]
        else:
            # 默认选取5个核心文本模型
            model_ids = ["deepseek-r1", "llama-3.3-70b", "glm-4", "groq-llama", "gemini-flash"]

        # 获取/创建会话
        user = await db.get_or_create_user()
        conversation = await db.get_or_create_conversation(
            user["id"], None
        )
        conv_id = conversation["id"]

        debate_turns = []

        # 获取用户 API Key（只取一次）
        user_api_key = http_request.headers.get("X-API-Key", "")

        for round_num in range(1, request.rounds + 1):
            statements = []

            for mid in model_ids:
                cfg = MODEL_REGISTRY[mid]

                # 构造辩论提示词，注入角色身份
                if round_num == 1:
                    debate_prompt = (
                        f"你是「{cfg.role}」{cfg.name}，你的专长是{cfg.specialty}。\n"
                        f"现在众神议会正在辩论以下话题：\n\n"
                        f"「{request.topic}」\n\n"
                        f"请从你的专业角度出发，给出你的观点（200字以内）。要有立场、有论据。"
                    )
                else:
                    # 后续轮次：参考其他模型的发言进行反驳/补充
                    prev_statements = "\n".join(
                        [f"【{s['role']}】：{s['statement'][:100]}..." for s in statements]
                    )
                    debate_prompt = (
                        f"你是「{cfg.role}」{cfg.name}，你的专长是{cfg.specialty}。\n"
                        f"辩论话题：「{request.topic}」\n\n"
                        f"其他神明的观点：\n{prev_statements}\n\n"
                        f"请回应其他观点，可以赞同、反驳或补充（150字以内）。"
                    )

                provider = create_provider(cfg, api_key=user_api_key)
                messages = [{"role": "user", "content": debate_prompt}]
                result = await provider.chat(cfg, messages)

                if result.success:
                    statements.append({
                        "model_id": mid,
                        "name": cfg.name,
                        "role": cfg.role,
                        "statement": result.content,
                        "tokens_used": result.tokens_used
                    })

            debate_turns.append(DebateTurn(
                round_num=round_num,
                statements=statements
            ))

        return DebateResponse(
            topic=request.topic,
            debate_turns=debate_turns,
            conversation_id=conv_id,
            total_rounds=request.rounds
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"辩论失败: {str(e)}")


# ==================== 信息查询 API ====================

@app.get("/api/models", response_model=ModelsResponse, tags=["信息查询"])
async def list_models():
    """
    获取所有可用模型列表及其角色定位
    """
    models = []
    for mid, cfg in MODEL_REGISTRY.items():
        provider = create_provider(cfg)
        is_available = getattr(provider, "is_available", True)
        models.append(ModelResponse(
            model_id=cfg.model_id,
            name=cfg.name,
            role=cfg.role,
            specialty=cfg.specialty,
            icon=cfg.icon,
            description=cfg.description,
            is_available=is_available,
            category=cfg.category
        ))
    return ModelsResponse(models=models, total=len(models))


@app.get("/api/roles", response_model=RolesResponse, tags=["信息查询"])
async def list_roles():
    """
    获取 AI 角色卡片列表（用于前端展示"众神"面板）
    """
    # 定义角色卡片顺序（按"神职"重要性排列）
    role_order = [
        ("deepseek-r1", "#E74C3C"),
        ("llama-3.3-70b", "#3498DB"),
        ("gemini-flash", "#9B59B6"),
        ("glm-4", "#E91E63"),
        ("groq-llama", "#F39C12"),
        ("agnes-image", "#2ECC71"),
        ("agnes-video", "#1ABC9C"),
    ]

    roles = []
    for mid, color in role_order:
        if mid in MODEL_REGISTRY:
            cfg = MODEL_REGISTRY[mid]
            roles.append(RoleCard(
                id=mid,
                name=cfg.role,
                model_id=cfg.model_id,
                icon=cfg.icon,
                specialty=cfg.specialty,
                description=cfg.description,
                color=color
            ))

    # 添加特殊角色：风控质检官（双模型交叉验证）
    roles.append(RoleCard(
        id="quality-checker",
        name="风控质检官",
        model_id="llama-3.3-70b+deepseek-r1",
        icon="🛡️",
        specialty="双模型交叉验证 / 事实核查",
        description="当输出内容涉及关键决策或高风险场景时，双模型交叉验证确保回答的准确性和安全性。",
        color="#34495E"
    ))

    return RolesResponse(roles=roles, total=len(roles))


@app.get("/api/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """
    健康检查接口
    """
    mode = get_running_mode()
    available_count = sum(
        1 for mid, cfg in MODEL_REGISTRY.items()
        if cfg.category == "text"
    )
    return HealthResponse(
        status="ok",
        version="1.0.0",
        mode=mode,
        models_available=available_count,
        timestamp=datetime.now().isoformat()
    )


# ==================== 辅助函数 ====================

async def _call_model(provider, model_config: ModelConfig, messages: List[dict]):
    """调用单个模型的辅助函数（用于 gather 并行调用）"""
    try:
        if model_config.category == "image":
            return await provider.generate(messages[-1]["content"])
        elif model_config.category == "video":
            return await provider.generate(messages[-1]["content"])
        else:
            return await provider.chat(model_config, messages)
    except Exception as e:
        from models import ModelResult
        return ModelResult(
            content=f"[错误] {model_config.name} 调用失败: {str(e)}",
            model_id=model_config.model_id,
            success=False,
            error=str(e)
        )


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["*.db"]
    )

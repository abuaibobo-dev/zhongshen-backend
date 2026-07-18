"""
模型配置和调用封装层
定义统一的 ModelProvider 接口，实现 OpenRouter 调用和 Mock 降级
"""

import os
import time
import json
import random
import asyncio
from typing import Protocol, Optional, List, Dict
from dataclasses import dataclass, field
import httpx


# ==================== 模型注册表 ====================

@dataclass
class ModelConfig:
    """单个模型的配置"""
    model_id: str               # 内部标识符
    name: str                   # 显示名称
    role: str                   # 角色名称
    specialty: str              # 擅长领域
    icon: str                   # 图标/emoji
    description: str            # 角色描述
    category: str = "text"      # text / image / video
    api_model_name: str = ""    # 调用 API 时使用的模型名
    color: str = "#6C5CE7"     # 主题色


# 所有可用模型的注册表
MODEL_REGISTRY: Dict[str, ModelConfig] = {}


def register_models():
    """注册所有可用模型"""
    models = [
        ModelConfig(
            model_id="deepseek-r1",
            name="DeepSeek-R1",
            role="首席科学官",
            specialty="深度推理 / 代码 / 数学证明",
            icon="🧠",
            description="负责深度思考、复杂推理、代码编写和数学证明。当问题需要严密逻辑链条时，由TA出马。",
            category="text",
            api_model_name="deepseek/deepseek-r1:free",
            color="#E74C3C"
        ),
        ModelConfig(
            model_id="llama-3.3-70b",
            name="Llama-3.3-70B",
            role="高级参谋",
            specialty="通用复杂任务 / 综合分析",
            icon="🎯",
            description="全能型选手，擅长处理复杂综合性任务。当路由引擎无法明确分类时，由TA兜底。",
            category="text",
            api_model_name="meta-llama/llama-3.3-70b-instruct:free",
            color="#3498DB"
        ),
        ModelConfig(
            model_id="gemini-flash",
            name="Gemini 2.0 Flash",
            role="首席感知官",
            specialty="超长文档 / 视频分析 / 大上下文",
            icon="👁",
            description="处理超长文本、文档解析和多模态输入。当输入超过5000字符或涉及PDF/长文分析时激活。",
            category="text",
            api_model_name="google/gemini-2.0-flash-exp:free",
            color="#9B59B6"
        ),
        ModelConfig(
            model_id="groq-llama",
            name="Groq Llama-3.3-70B",
            role="执行秘书",
            specialty="速记 / 格式化 / 总结摘要",
            icon="⚡",
            description="以极快速度完成总结、摘要、表格生成、JSON格式化等结构化任务。",
            category="text",
            api_model_name="meta-llama/llama-3.3-70b-instruct:free",
            color="#F39C12"
        ),
        ModelConfig(
            model_id="glm-4",
            name="GLM-4-9B",
            role="首席创意官",
            specialty="文案 / 故事 / 标题 / 诗歌",
            icon="✨",
            description="创意写作专家。小红书文案、抖音标题、诗歌故事、营销文案，都是TA的拿手好戏。",
            category="text",
            api_model_name="thudm/glm-4-9b-chat:free",
            color="#E91E63"
        ),
        ModelConfig(
            model_id="agnes-image",
            name="Agnes Image",
            role="首席画师",
            specialty="图片生成 / 海报设计 / 插画",
            icon="🎨",
            description="负责所有视觉内容的生成。输入文字描述即可产出精美图片。",
            category="image",
            api_model_name="",  # 图像模型单独处理
            color="#2ECC71"
        ),
        ModelConfig(
            model_id="agnes-video",
            name="Agnes Video",
            role="电影导演",
            specialty="视频生成 / 动效制作",
            icon="🎬",
            description="视频生成专家。将文字描述转化为动态视频，支持图生视频。",
            category="video",
            api_model_name="",  # 视频模型单独处理
            color="#1ABC9C"
        ),
    ]
    for m in models:
        MODEL_REGISTRY[m.model_id] = m
    return MODEL_REGISTRY


# ==================== 统一接口 ====================

@dataclass
class ModelResult:
    """模型调用结果"""
    content: str
    model_id: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


class ModelProvider(Protocol):
    """模型调用统一接口"""
    async def chat(self, model_config: ModelConfig, messages: List[dict], **kwargs) -> ModelResult:
        ...


# ==================== OpenRouter Provider ====================

class OpenRouterProvider:
    """
    通过 OpenRouter API 调用免费文本模型
    文档：https://openrouter.ai/docs
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.is_available = bool(self.api_key)

    async def chat(self, model_config: ModelConfig, messages: List[dict], **kwargs) -> ModelResult:
        """调用 OpenRouter API"""
        if not self.is_available:
            return ModelResult(
                content="",
                model_id=model_config.model_id,
                success=False,
                error="OpenRouter API Key 未配置"
            )

        start_time = time.time()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://zhongshen.app",
            "X-Title": "ZhongShen"
        }

        payload = {
            "model": model_config.api_model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(self.BASE_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

                content = data["choices"][0]["message"]["content"]
                tokens = data.get("usage", {}).get("total_tokens", 0)
                latency = (time.time() - start_time) * 1000

                return ModelResult(
                    content=content,
                    model_id=model_config.model_id,
                    tokens_used=tokens,
                    latency_ms=round(latency, 2)
                )
        except httpx.TimeoutException:
            return ModelResult(
                content="", model_id=model_config.model_id,
                success=False, error="请求超时"
            )
        except Exception as e:
            return ModelResult(
                content="", model_id=model_config.model_id,
                success=False, error=str(e)
            )


# ==================== Mock Provider ====================

class MockProvider:
    """
    模拟模型响应，用于无 API Key 时的开发测试
    返回带角色风格的模拟回复
    """

    # 每个角色的模拟回复模板
    MOCK_RESPONSES = {
        "deepseek-r1": [
            "让我仔细分析一下这个问题...\n\n**思考过程：**\n首先，我们需要明确问题的核心要素。通过逐步推理：\n\n1. 第一步：确定已知条件\n2. 第二步：建立逻辑链条\n3. 第三步：推导结论\n\n**最终结论：**\n经过严密的逻辑推理，答案是——{topic}。\n\n> 这个推理过程耗时约{time}步",
            "```\n// 让我为你编写代码\n{code_block}\n```\n\n以上代码的时间复杂度为 O(n)，空间复杂度为 O(1)。如果需要进一步优化，可以考虑使用动态规划。"
        ],
        "llama-3.3-70b": [
            "好的，让我综合各方面来分析一下这个问题。\n\n**分析：**\n- 从技术角度来看：{topic}\n- 从实用角度来看：需要考虑实际场景的约束\n- 从长远来看：建议采用渐进式方案\n\n**建议：**\n综合以上分析，我的建议是先从小规模验证开始，逐步迭代完善。",
            "这是一个很好的问题。让我从多个维度来回答：\n\n1. **背景**：首先需要理解上下文\n2. **核心**：关键在于找到最优路径\n3. **实践**：建议按照以下步骤执行...\n\n希望这个回答对你有帮助！"
        ],
        "gemini-flash": [
            "📄 已接收并处理您的文档/长文本输入。\n\n**内容摘要：**\n- 主要主题：{topic}\n- 关键信息点已提取\n- 文档结构分析完成\n\n**详细分析：**\n根据输入内容，我识别出以下几个核心要点...",
            "我已经仔细阅读了您提供的大量内容。以下是整理后的结构化输出..."
        ],
        "groq-llama": [
            "⚡ 快速整理完毕！\n\n| 项目 | 内容 |\n|------|------|\n| 主题 | {topic} |\n| 要点1 | 核心概念梳理 |\n| 要点2 | 关键数据分析 |\n| 要点3 | 行动建议 |\n\n**总结：** 已为您完成结构化整理。",
            "```json\n{\n  \"summary\": \"已完成摘要\",\n  \"key_points\": [\"要点1\", \"要点2\", \"要点3\"],\n  \"action_items\": [\"待办1\", \"待办2\"]\n}\n```\n已按JSON格式输出。"
        ],
        "glm-4": [
            "✨ 让我为你创作...\n\n---\n\n**{topic}**\n\n在数字与梦境的交界处，\n有一片众神之域在等待。\n每个AI都是一位神明，\n各司其职，各展所能。\n\n---\n\n*（以上为创意输出，可根据需要调整风格）*",
            "📱 小红书文案已就位！\n\n🔥 标题：「这个AI聚合App绝了！一个顶十个！」\n\n正文：\n姐妹们！！！我发现了一个神器App 🎉\n输入一句话就能自动找到最合适的AI来回答\n简直是效率天花板 ✨\n\n#AI #效率工具 #科技好物"
        ],
        "agnes-image": [
            "🎨 图片生成任务已接收！\n\n描述已理解，正在为您创作...\n\n[模拟模式] 实际部署后将调用图像生成API\n生成参数：1024×1024, quality=high\n\n图片URL: https://placeholder.com/generated_image.png",
            "🖼️ 画师已开工！\n\n根据您的描述「{topic}」，我构思了以下画面...\n\n[模拟模式] 图片将在正式环境中生成"
        ],
        "agnes-video": [
            "🎬 视频生成计划制定完毕！\n\n分镜脚本：\n- Scene 1: 开场（0-3s）\n- Scene 2: 主体展示（3-8s）\n- Scene 3: 结尾（8-10s）\n\n[模拟模式] 视频将在正式环境中渲染",
            "📹 导演已就位！\n\n根据您的描述，建议视频时长10-15秒，分辨率1080p。\n\n[模拟模式] 实际视频需调用视频生成API"
        ]
    }

    def __init__(self):
        self.is_available = True

    async def chat(self, model_config: ModelConfig, messages: List[dict], **kwargs) -> ModelResult:
        """返回模拟响应"""
        start_time = time.time()
        await asyncio.sleep(random.uniform(0.3, 1.2))  # 模拟网络延迟

        # 提取用户消息中的关键词作为话题
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg["content"][:50]
                break

        templates = self.MOCK_RESPONSES.get(
            model_config.model_id,
            ["[Mock模式] 这是来自 {model} 的模拟回复。\n\n你的问题：{topic}\n\n实际部署后将由真实模型回答。"]
        )
        template = random.choice(templates)

        code_block = """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
"""

        content = template.format(
            topic=user_msg or "用户问题",
            time=random.randint(3, 15),
            code_block=code_block.strip(),
            model=model_config.name
        )

        latency = (time.time() - start_time) * 1000
        tokens = random.randint(80, 500)

        return ModelResult(
            content=content,
            model_id=model_config.model_id,
            tokens_used=tokens,
            latency_ms=round(latency, 2),
            success=True
        )


# ==================== 图像/视频 Provider（占位） ====================

class AgnesImageProvider:
    """图像生成 Provider（预留接口）"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("AGNES_IMAGE_API_KEY", "")

    async def generate(self, prompt: str, **kwargs) -> ModelResult:
        """生成图片（当前为模拟）"""
        await asyncio.sleep(0.5)
        return ModelResult(
            content=f"🎨 图片生成完成（模拟模式）\n提示词：{prompt[:100]}...\n[正式环境将返回图片URL]",
            model_id="agnes-image",
            tokens_used=0,
            latency_ms=500.0,
            success=True
        )


class AgnesVideoProvider:
    """视频生成 Provider（预留接口）"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("AGNES_VIDEO_API_KEY", "")

    async def generate(self, prompt: str, **kwargs) -> ModelResult:
        """生成视频（当前为模拟）"""
        await asyncio.sleep(1.0)
        return ModelResult(
            content=f"🎬 视频生成任务已创建（模拟模式）\n提示词：{prompt[:100]}...\n[正式环境将返回视频URL]",
            model_id="agnes-video",
            tokens_used=0,
            latency_ms=1000.0,
            success=True
        )


# ==================== Provider 工厂 ====================

def create_provider(model_config: ModelConfig) -> object:
    """根据模型配置创建合适的 Provider"""
    if model_config.category == "image":
        return AgnesImageProvider()
    elif model_config.category == "video":
        return AgnesVideoProvider()
    else:
        # 文本模型：优先 OpenRouter，降级到 Mock
        openrouter = OpenRouterProvider()
        if openrouter.is_available:
            return openrouter
        return MockProvider()


def get_running_mode() -> str:
    """获取当前运行模式"""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    return "live" if api_key else "mock"

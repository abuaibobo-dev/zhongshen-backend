"""
路由引擎 - 意图识别 + 模型路由
基于关键词规则，将用户输入路由到最合适的 AI 模型
"""

import re
from typing import Optional, List, Tuple
from dataclasses import dataclass

from models import MODEL_REGISTRY, register_models

# 确保模型已注册
register_models()


@dataclass
class RoutingResult:
    """路由决策结果"""
    model_id: str
    reason: str
    confidence: str = "high"  # high / medium / low


# ==================== 路由规则定义 ====================

class RouterEngine:
    """
    基于关键词的意图识别与模型路由引擎
    规则按优先级从高到低匹配
    """

    def __init__(self):
        self._build_rules()

    def _build_rules(self):
        """构建路由规则表（按优先级从高到低）"""

        # 【最高优先级】生成类需求 - 按模态细分
        self.rules: List[Tuple[re.Pattern, str, str]] = [
            # 1. 视频生成 - Agnes Video
            (
                re.compile(
                    r"生成视频|做视频|让图动|制作视频|视频生成|"
                    r"clip.*video|video.*gener|文生视频|图生视频|"
                    r"animated|animation|动效", re.IGNORECASE
                ),
                "agnes-video",
                "关键词匹配: 视频生成相关 → 电影导演 (Agnes Video)"
            ),

            # 2. 图片生成 - Agnes Image
            (
                re.compile(
                    r"生成图片|画|配图|海报|插画|图标|logo|设计图|"
                    r"生成图像|image.*gener|draw|paint|photo.*gener|"
                    r"生成一张|给我画|帮我画|创作画|绘图", re.IGNORECASE
                ),
                "agnes-image",
                "关键词匹配: 图片生成相关 → 首席画师 (Agnes Image)"
            ),

            # 3. 代码/算法/证明 - DeepSeek R1
            (
                re.compile(
                    r"代码|code|编程|program|debug|算法|algorithm|证明|"
                    r"推导|数学|logic|logic|函数|class|def |import |"
                    r"写.*程序|python|javascript|java|c\+\+|rust|"
                    r"递归|循环|数据结构|复杂度", re.IGNORECASE
                ),
                "deepseek-r1",
                "关键词匹配: 代码/算法/推理 → 首席科学官 (DeepSeek-R1)"
            ),

            # 4. 超长输入或文档分析 - Gemini Flash
            (
                re.compile(
                    r"pdf|文档|document|paper|论文|文章|长文|"
                    r"pdf.*分析|解析.*pdf|阅读.*文章|总结.*文档", re.IGNORECASE
                ),
                "gemini-flash",
                "关键词匹配: PDF/文档/长文 → 首席感知官 (Gemini Flash)"
            ),

            # 5. 格式化/总结/整理 - Groq Llama
            (
                re.compile(
                    r"总结|摘要|概括|提炼|整理|格式化|表格|json|json化|"
                    r"markdown|转表格|做表格|timeline|时间线|流程|"
                    r"归类|分类整理|大纲|目录", re.IGNORECASE
                ),
                "groq-llama",
                "关键词匹配: 总结/格式化/表格 → 执行秘书 (Groq Llama)"
            ),

            # 6. 创意写作 - GLM-4
            (
                re.compile(
                    r"文案|故事|诗歌|写.*文章|小红书|抖音|微博|公众号|"
                    r"标题|标题党|营销|广告语|slogan|推广|脚本|剧本|"
                    r"小说|散文|日记|情书|贺卡|祝福|创意|写作|润色", re.IGNORECASE
                ),
                "glm-4",
                "关键词匹配: 文案/创意/故事 → 首席创意官 (GLM-4)"
            ),
        ]

    def route(self, message: str, force_model: Optional[str] = None) -> RoutingResult:
        """
        核心路由方法
        :param message: 用户输入消息
        :param force_model: 强制指定的模型ID（跳过路由）
        :return: RoutingResult
        """
        # 强制路由：用户显式指定模型
        if force_model:
            if force_model in MODEL_REGISTRY:
                cfg = MODEL_REGISTRY[force_model]
                return RoutingResult(
                    model_id=force_model,
                    reason=f"强制指定: {cfg.role} ({cfg.name})",
                    confidence="high"
                )
            else:
                return self._route_by_default(f"未知模型 {force_model}，回退默认路由")

        # 规则匹配路由
        for pattern, model_id, reason in self.rules:
            if pattern.search(message):
                # 再次确认模型ID存在
                if model_id in MODEL_REGISTRY:
                    return RoutingResult(
                        model_id=model_id,
                        reason=reason,
                        confidence="high"
                    )

        # 【超长文本兜底】输入超过5000字符
        if len(message) > 5000:
            return RoutingResult(
                model_id="gemini-flash",
                reason=f"输入长度 {len(message)} > 5000，超长文本 → 首席感知官 (Gemini Flash)",
                confidence="medium"
            )

        # 【默认兜底】通用复杂任务
        return self._route_by_default("无关键词匹配，默认路由")

    def _route_by_default(self, reason: str) -> RoutingResult:
        """默认路由：Llama 3.3 70B"""
        return RoutingResult(
            model_id="llama-3.3-70b",
            reason=f"{reason} → 高级参谋 (Llama-3.3-70B)",
            confidence="medium"
        )

    def get_all_text_models(self) -> List[str]:
        """获取所有文本类模型ID（用于竞技场/辩论模式）"""
        return [
            mid for mid, cfg in MODEL_REGISTRY.items()
            if cfg.category == "text"
        ]

    def get_model_info(self, model_id: str) -> Optional[dict]:
        """获取指定模型的详细信息"""
        if model_id in MODEL_REGISTRY:
            cfg = MODEL_REGISTRY[model_id]
            return {
                "model_id": cfg.model_id,
                "name": cfg.name,
                "role": cfg.role,
                "specialty": cfg.specialty,
                "icon": cfg.icon,
                "description": cfg.description,
                "category": cfg.category,
                "color": cfg.color
            }
        return None


# 全局单例路由引擎
router = RouterEngine()

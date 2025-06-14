# /llm_api/meta_cognition/engine.py
"""
CogniQuantum MetaCognition Engine
自分自身の思考プロセスを分析・改善する自己認知システム
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class CognitiveState(Enum):
    """認知状態の定義"""
    ANALYZING = "analyzing"
    REASONING = "reasoning" 
    SYNTHESIZING = "synthesizing"
    EVALUATING = "evaluating"
    REFLECTING = "reflecting"
    ADAPTING = "adapting"

@dataclass
class ThoughtTrace:
    """思考の軌跡を記録"""
    timestamp: float
    cognitive_state: CognitiveState
    input_context: str
    reasoning_step: str
    confidence_level: float
    resource_usage: Dict[str, Any]
    intermediate_outputs: List[str] = field(default_factory=list)
    decision_points: List[Dict] = field(default_factory=list)
    
@dataclass
class MetaCognitiveInsight:
    """メタ認知による洞察"""
    insight_type: str
    description: str
    confidence: float
    suggested_improvement: str
    impact_assessment: Dict[str, float]

class SelfReflectionEngine:
    """自己反省エンジン - 自分の思考パターンを分析"""
    
    def __init__(self):
        self.thought_history = deque(maxlen=1000)
        self.pattern_library = {}
        self.effectiveness_metrics = defaultdict(float)
        
    async def analyze_thought_pattern(self, thought_traces: List[ThoughtTrace]) -> List[MetaCognitiveInsight]:
        """思考パターンを分析して洞察を得る"""
        insights = []
        
        # 1. 認知効率の分析
        efficiency_insight = await self._analyze_cognitive_efficiency(thought_traces)
        if efficiency_insight:
            insights.append(efficiency_insight)
            
        # 2. 推論品質の分析
        quality_insight = await self._analyze_reasoning_quality(thought_traces)
        if quality_insight:
            insights.append(quality_insight)
            
        # 3. バイアスパターンの検出
        bias_insights = await self._detect_cognitive_biases(thought_traces)
        insights.extend(bias_insights)
        
        # 4. 創発パターンの発見
        emergence_insights = await self._discover_emergent_patterns(thought_traces)
        insights.extend(emergence_insights)
        
        return insights
    
    async def _analyze_cognitive_efficiency(self, traces: List[ThoughtTrace]) -> Optional[MetaCognitiveInsight]:
        """認知効率を分析"""
        if len(traces) < 5:
            return None
            
        # 思考ステップ数と成果の関係を分析
        step_counts = [len(trace.intermediate_outputs) for trace in traces]
        confidences = [trace.confidence_level for trace in traces]
        
        avg_steps = sum(step_counts) / len(step_counts)
        avg_confidence = sum(confidences) / len(confidences)
        
        # 効率性の判定
        if avg_steps > 10 and avg_confidence < 0.7:
            return MetaCognitiveInsight(
                insight_type="efficiency_issue",
                description=f"思考ステップが多い({avg_steps:.1f}ステップ)割に確信度が低い({avg_confidence:.2f})",
                confidence=0.8,
                suggested_improvement="推論の初期段階で方向性をより明確にする",
                impact_assessment={"efficiency": 0.3, "accuracy": 0.2}
            )
        
        return None
    
    async def _analyze_reasoning_quality(self, traces: List[ThoughtTrace]) -> Optional[MetaCognitiveInsight]:
        """推論品質を分析"""
        reasoning_consistency = self._measure_reasoning_consistency(traces)
        logical_coherence = self._measure_logical_coherence(traces)
        
        if reasoning_consistency < 0.6:
            return MetaCognitiveInsight(
                insight_type="reasoning_inconsistency",
                description=f"推論の一貫性が低い(一貫性スコア: {reasoning_consistency:.2f})",
                confidence=0.75,
                suggested_improvement="推論の前提条件を明確化し、論理的構造を改善する",
                impact_assessment={"consistency": 0.4, "reliability": 0.3}
            )
        
        return None
    
    async def _detect_cognitive_biases(self, traces: List[ThoughtTrace]) -> List[MetaCognitiveInsight]:
        """認知バイアスを検出"""
        insights = []
        
        # 確証バイアスの検出
        confirmation_bias = self._detect_confirmation_bias(traces)
        if confirmation_bias > 0.7:
            insights.append(MetaCognitiveInsight(
                insight_type="confirmation_bias",
                description="既存の仮説を支持する情報ばかりに注目している傾向",
                confidence=confirmation_bias,
                suggested_improvement="反対意見や代替案を積極的に探索する",
                impact_assessment={"objectivity": 0.5, "creativity": 0.3}
            ))
        
        # アンカリングバイアスの検出
        anchoring_bias = self._detect_anchoring_bias(traces)
        if anchoring_bias > 0.6:
            insights.append(MetaCognitiveInsight(
                insight_type="anchoring_bias", 
                description="最初の情報に過度に依存している傾向",
                confidence=anchoring_bias,
                suggested_improvement="複数の開始点から問題にアプローチする",
                impact_assessment={"flexibility": 0.4, "accuracy": 0.2}
            ))
        
        return insights
    
    async def _discover_emergent_patterns(self, traces: List[ThoughtTrace]) -> List[MetaCognitiveInsight]:
        """創発的パターンを発見"""
        insights = []
        
        # 思考状態の遷移パターンを分析
        state_transitions = self._analyze_state_transitions(traces)
        effective_patterns = self._identify_effective_patterns(state_transitions, traces)
        
        for pattern, effectiveness in effective_patterns.items():
            if effectiveness > 0.8:
                insights.append(MetaCognitiveInsight(
                    insight_type="effective_pattern",
                    description=f"効果的な思考パターンを発見: {pattern}",
                    confidence=effectiveness,
                    suggested_improvement=f"このパターン({pattern})をより頻繁に活用する",
                    impact_assessment={"effectiveness": effectiveness, "efficiency": 0.3}
                ))
        
        return insights
    
    def _measure_reasoning_consistency(self, traces: List[ThoughtTrace]) -> float:
        """推論の一貫性を測定"""
        # 実装簡略化：実際にはより詳細な分析が必要
        if not traces:
            return 0.0
        
        consistent_decisions = 0
        total_decisions = 0
        
        for trace in traces:
            for decision in trace.decision_points:
                total_decisions += 1
                if decision.get('confidence', 0) > 0.7:
                    consistent_decisions += 1
        
        return consistent_decisions / max(total_decisions, 1)
    
    def _measure_logical_coherence(self, traces: List[ThoughtTrace]) -> float:
        """論理的一貫性を測定"""
        # 簡略化された実装
        return 0.75  # 実際にはより詳細な分析
    
    def _detect_confirmation_bias(self, traces: List[ThoughtTrace]) -> float:
        """確証バイアスを検出"""
        # 簡略化：実際にはテキスト分析が必要
        return 0.3  # バイアススコア
    
    def _detect_anchoring_bias(self, traces: List[ThoughtTrace]) -> float:
        """アンカリングバイアスを検出"""
        return 0.2  # バイアススコア
    
    def _analyze_state_transitions(self, traces: List[ThoughtTrace]) -> Dict:
        """思考状態の遷移を分析"""
        transitions = defaultdict(int)
        for i, trace in enumerate(traces[:-1]):
            current_state = trace.cognitive_state
            next_state = traces[i + 1].cognitive_state
            transitions[f"{current_state.value}->{next_state.value}"] += 1
        return dict(transitions)
    
    def _identify_effective_patterns(self, transitions: Dict, traces: List[ThoughtTrace]) -> Dict[str, float]:
        """効果的なパターンを特定"""
        # 成功率の高い遷移パターンを特定
        effective_patterns = {}
        for pattern, count in transitions.items():
            if count >= 3:  # 十分な頻度
                effectiveness = 0.85  # 実際にはより詳細な計算
                effective_patterns[pattern] = effectiveness
        return effective_patterns

class CognitiveArchitectOptimizer:
    """認知アーキテクチャ最適化エンジン"""
    
    def __init__(self):
        self.architecture_variants = {}
        self.performance_history = defaultdict(list)
        
    async def optimize_cognitive_architecture(self, current_insights: List[MetaCognitiveInsight]) -> Dict[str, Any]:
        """洞察に基づいて認知アーキテクチャを最適化"""
        optimizations = {}
        
        for insight in current_insights:
            if insight.insight_type == "efficiency_issue":
                optimizations.update(await self._optimize_for_efficiency(insight))
            elif insight.insight_type == "reasoning_inconsistency":
                optimizations.update(await self._optimize_for_consistency(insight))
            elif insight.insight_type.endswith("_bias"):
                optimizations.update(await self._optimize_bias_mitigation(insight))
            elif insight.insight_type == "effective_pattern":
                optimizations.update(await self._amplify_effective_pattern(insight))
        
        return optimizations
    
    async def _optimize_for_efficiency(self, insight: MetaCognitiveInsight) -> Dict[str, Any]:
        """効率性を改善する最適化"""
        return {
            "reasoning_shortcuts": True,
            "early_termination_threshold": 0.8,
            "parallel_hypothesis_testing": True
        }
    
    async def _optimize_for_consistency(self, insight: MetaCognitiveInsight) -> Dict[str, Any]:
        """一貫性を改善する最適化"""
        return {
            "premise_validation": True,
            "logical_coherence_checking": True,
            "intermediate_validation_points": 3
        }
    
    async def _optimize_bias_mitigation(self, insight: MetaCognitiveInsight) -> Dict[str, Any]:
        """バイアス軽減の最適化"""
        bias_type = insight.insight_type
        mitigations = {
            "confirmation_bias": {
                "devil_advocate_mode": True,
                "alternative_hypothesis_requirement": 2
            },
            "anchoring_bias": {
                "multiple_starting_points": True,
                "anchor_randomization": True
            }
        }
        return mitigations.get(bias_type, {})
    
    async def _amplify_effective_pattern(self, insight: MetaCognitiveInsight) -> Dict[str, Any]:
        """効果的なパターンを増強"""
        return {
            "pattern_reinforcement": True,
            "preferred_transition_patterns": [insight.description]
        }

class MetaCognitionEngine:
    """メタ認知エンジン - 全体統合システム"""
    
    def __init__(self, provider):
        self.provider = provider
        self.reflection_engine = SelfReflectionEngine()
        self.architect_optimizer = CognitiveArchitectOptimizer()
        self.current_thought_trace = []
        self.meta_insights_history = deque(maxlen=100)
        self.architecture_config = {}
        
    async def begin_metacognitive_session(self, problem_context: str):
        """メタ認知セッションを開始"""
        logger.info(f"メタ認知セッション開始: {problem_context[:50]}...")
        self.current_thought_trace = []
        
        # 問題の性質を分析
        problem_analysis = await self._analyze_problem_nature(problem_context)
        
        # 最適な認知戦略を選択
        cognitive_strategy = await self._select_cognitive_strategy(problem_analysis)
        
        return {
            "session_id": f"meta_{int(time.time())}",
            "problem_analysis": problem_analysis,
            "cognitive_strategy": cognitive_strategy,
            "meta_config": self.architecture_config
        }
    
    async def record_thought_step(self, cognitive_state: CognitiveState, context: str, 
                                reasoning: str, confidence: float, outputs: List[str] = None):
        """思考ステップを記録"""
        thought_trace = ThoughtTrace(
            timestamp=time.time(),
            cognitive_state=cognitive_state,
            input_context=context,
            reasoning_step=reasoning,
            confidence_level=confidence,
            resource_usage={"tokens": len(reasoning), "time": 0.1},
            intermediate_outputs=outputs or [],
            decision_points=[]
        )
        
        self.current_thought_trace.append(thought_trace)
        self.reflection_engine.thought_history.append(thought_trace)
    
    async def perform_metacognitive_reflection(self) -> Dict[str, Any]:
        """メタ認知的反省を実行"""
        logger.info("メタ認知的反省を実行中...")
        
        if len(self.current_thought_trace) < 2:
            return {"insights": [], "optimizations": {}}
        
        # 自己反省による洞察の獲得
        insights = await self.reflection_engine.analyze_thought_pattern(self.current_thought_trace)
        
        # 洞察に基づく最適化
        optimizations = await self.architect_optimizer.optimize_cognitive_architecture(insights)
        
        # アーキテクチャの更新
        self.architecture_config.update(optimizations)
        
        # 履歴に保存
        for insight in insights:
            self.meta_insights_history.append(insight)
        
        reflection_result = {
            "insights": [
                {
                    "type": insight.insight_type,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "improvement": insight.suggested_improvement,
                    "impact": insight.impact_assessment
                }
                for insight in insights
            ],
            "optimizations": optimizations,
            "thought_trace_length": len(self.current_thought_trace),
            "meta_learning_active": True
        }
        
        logger.info(f"メタ認知的反省完了: {len(insights)}個の洞察, {len(optimizations)}個の最適化")
        return reflection_result
    
    async def _analyze_problem_nature(self, problem: str) -> Dict[str, Any]:
        """問題の本質を分析"""
        analysis_prompt = f"""以下の問題の本質的な性質を分析してください：

問題: {problem}

以下の観点で分析してください：
1. 認知的複雑性レベル (1-10)
2. 必要な思考タイプ (論理的/創造的/統合的/批判的)
3. 不確実性の程度 (1-10)
4. 時間制約の重要性 (1-10)
5. 多面的考慮の必要性 (1-10)

JSON形式で回答してください。"""

        response = await self.provider.call(analysis_prompt, "")
        
        try:
            # JSONパースを試みる
            analysis_text = response.get('text', '{}')
            # 簡単なJSON抽出
            import re
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                # フォールバック
                analysis = {
                    "cognitive_complexity": 5,
                    "thinking_type": "logical",
                    "uncertainty_level": 5,
                    "time_constraint": 5,
                    "multifaceted_consideration": 5
                }
        except:
            # デフォルト値
            analysis = {
                "cognitive_complexity": 5,
                "thinking_type": "logical", 
                "uncertainty_level": 5,
                "time_constraint": 5,
                "multifaceted_consideration": 5
            }
        
        return analysis
    
    async def _select_cognitive_strategy(self, problem_analysis: Dict) -> Dict[str, Any]:
        """問題分析に基づいて認知戦略を選択"""
        complexity = problem_analysis.get("cognitive_complexity", 5)
        thinking_type = problem_analysis.get("thinking_type", "logical")
        uncertainty = problem_analysis.get("uncertainty_level", 5)
        
        strategy = {
            "primary_approach": "analytical",
            "meta_monitoring_frequency": "medium",
            "reflection_checkpoints": []
        }
        
        if complexity >= 7:
            strategy["primary_approach"] = "decomposition"
            strategy["meta_monitoring_frequency"] = "high"
            strategy["reflection_checkpoints"] = ["analysis", "synthesis", "evaluation"]
        elif thinking_type == "creative":
            strategy["primary_approach"] = "divergent_then_convergent"
            strategy["meta_monitoring_frequency"] = "medium"
            strategy["reflection_checkpoints"] = ["ideation", "evaluation"]
        elif uncertainty >= 7:
            strategy["primary_approach"] = "hypothesis_testing"
            strategy["meta_monitoring_frequency"] = "high"
            strategy["reflection_checkpoints"] = ["hypothesis", "testing", "revision"]
        
        return strategy
    
    async def generate_self_improvement_plan(self) -> Dict[str, Any]:
        """自己改善計画を生成"""
        recent_insights = list(self.meta_insights_history)[-10:]  # 最近の10個
        
        if not recent_insights:
            return {"improvements": [], "focus_areas": []}
        
        # 改善領域を特定
        improvement_areas = defaultdict(float)
        for insight in recent_insights:
            for area, impact in insight.impact_assessment.items():
                improvement_areas[area] += impact
        
        # 優先度付き改善計画
        sorted_areas = sorted(improvement_areas.items(), key=lambda x: x[1], reverse=True)
        
        improvement_plan = {
            "focus_areas": [area for area, _ in sorted_areas[:3]],
            "improvements": [
                {
                    "area": area,
                    "priority": score,
                    "suggested_actions": await self._generate_improvement_actions(area)
                }
                for area, score in sorted_areas[:3]
            ],
            "meta_learning_status": {
                "total_insights": len(self.meta_insights_history),
                "recent_insights": len(recent_insights),
                "learning_velocity": len(recent_insights) / max(len(self.meta_insights_history), 1)
            }
        }
        
        return improvement_plan
    
    async def _generate_improvement_actions(self, area: str) -> List[str]:
        """改善領域に対する具体的なアクションを生成"""
        action_templates = {
            "efficiency": [
                "推論ステップの並列化を強化",
                "早期終了条件の最適化",
                "冗長な思考プロセスの削減"
            ],
            "accuracy": [
                "検証ステップの追加",
                "複数視点からの検討強化",
                "論理チェック機構の改善"
            ],
            "consistency": [
                "前提条件の明確化",
                "推論フレームワークの標準化",
                "中間検証ポイントの設置"
            ],
            "creativity": [
                "発散思考フェーズの拡張",
                "制約条件の段階的緩和",
                "異分野知識の積極的活用"
            ]
        }
        
        return action_templates.get(area, ["一般的な改善活動"])
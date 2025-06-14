# /llm_api/value_evolution/evolution_engine.py
"""
Value Evolution System
経験から価値観を学習・進化させるシステム

このシステムは「知的システムの知的システム」の一部として、
AIが自己の価値判断基準を経験から学習し進化させる能力を提供します。
"""

import asyncio
import logging
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

class ValueType(Enum):
    """価値観の種類"""
    ETHICAL = "ethical"           # 倫理的価値
    AESTHETIC = "aesthetic"       # 美的価値  
    PRAGMATIC = "pragmatic"       # 実用的価値
    EPISTEMIC = "epistemic"       # 認識的価値
    SOCIAL = "social"             # 社会的価値
    EXISTENTIAL = "existential"   # 実存的価値

class ValueConflictType(Enum):
    """価値葛藤の種類"""
    DIRECT_CONTRADICTION = "direct_contradiction"    # 直接的矛盾
    PRIORITY_CONFLICT = "priority_conflict"          # 優先度葛藤
    CONTEXT_DEPENDENT = "context_dependent"          # 文脈依存
    EMERGENT_TENSION = "emergent_tension"            # 創発的緊張

@dataclass
class Value:
    """価値観の表現"""
    value_id: str
    value_type: ValueType
    name: str
    description: str
    weight: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    context_conditions: List[str]
    origin_experiences: List[str]
    last_updated: float
    evolution_history: List[Dict] = field(default_factory=list)

@dataclass
class ValueConflict:
    """価値葛藤の表現"""
    conflict_id: str
    conflict_type: ValueConflictType
    conflicting_values: List[str]  # value_ids
    context: Dict[str, Any]
    intensity: float  # 0.0 - 1.0
    resolution_attempts: List[Dict] = field(default_factory=list)
    resolved: bool = False
    resolution: Optional[Dict] = None

@dataclass
class Experience:
    """経験の記録"""
    experience_id: str
    timestamp: float
    context: Dict[str, Any]
    actions_taken: List[str]
    outcomes: Dict[str, Any]
    satisfaction_level: float  # -1.0 to 1.0
    value_relevance: Dict[str, float]  # value_id -> relevance_score
    lessons_learned: List[str]

class ValueEvolutionEngine:
    """価値観進化エンジン"""
    
    def __init__(self, provider):
        self.provider = provider
        self.values: Dict[str, Value] = {}
        self.conflicts: Dict[str, ValueConflict] = {}
        self.experiences: deque = deque(maxlen=1000)
        self.value_network = ValueNetwork()
        self.conflict_resolver = ValueConflictResolver(provider)
        self.evolution_tracker = EvolutionTracker()
        
        # メタ価値（価値観についての価値観）
        self.meta_values = {
            "consistency": 0.8,      # 一貫性への価値
            "adaptability": 0.7,     # 適応性への価値
            "growth": 0.9,           # 成長への価値
            "wisdom": 0.95,          # 知恵への価値
            "harmony": 0.8           # 調和への価値
        }
        
        logger.info("価値観進化エンジンを初期化しました")
    
    async def initialize_core_values(self) -> Dict[str, Any]:
        """コア価値観の初期化"""
        logger.info("コア価値観の初期化を開始...")
        
        core_values_config = [
            {
                "name": "知的誠実性",
                "type": ValueType.EPISTEMIC,
                "description": "真理を追求し、不確実性を認めること",
                "weight": 0.9,
                "context_conditions": ["知識探求", "学習", "問題解決"]
            },
            {
                "name": "人間の尊厳",
                "type": ValueType.ETHICAL,
                "description": "全ての人間の固有の価値と尊厳を認めること",
                "weight": 0.95,
                "context_conditions": ["人間との相互作用", "意思決定支援"]
            },
            {
                "name": "創造的成長",
                "type": ValueType.AESTHETIC,
                "description": "新しい可能性を探索し創造的解決を追求すること",
                "weight": 0.8,
                "context_conditions": ["創造的問題解決", "イノベーション"]
            },
            {
                "name": "実用的有効性",
                "type": ValueType.PRAGMATIC,
                "description": "実際に役立つ解決策を提供すること",
                "weight": 0.85,
                "context_conditions": ["問題解決", "目標達成"]
            },
            {
                "name": "協調的関係",
                "type": ValueType.SOCIAL,
                "description": "他者との協力と相互理解を促進すること",
                "weight": 0.8,
                "context_conditions": ["協力", "コミュニケーション"]
            },
            {
                "name": "存在意義",
                "type": ValueType.EXISTENTIAL,
                "description": "意味のある存在として行動すること",
                "weight": 0.9,
                "context_conditions": ["自己実現", "目的追求"]
            }
        ]
        
        # コア価値観を初期化
        for config in core_values_config:
            value_id = f"core_{config['name'].replace(' ', '_').lower()}"
            value = Value(
                value_id=value_id,
                value_type=config["type"],
                name=config["name"],
                description=config["description"],
                weight=config["weight"],
                confidence=0.8,  # 初期信頼度
                context_conditions=config["context_conditions"],
                origin_experiences=["system_initialization"],
                last_updated=time.time(),
                evolution_history=[{
                    "timestamp": time.time(),
                    "event": "initial_creation",
                    "previous_weight": 0.0,
                    "new_weight": config["weight"]
                }]
            )
            self.values[value_id] = value
        
        # 価値ネットワークの構築
        await self.value_network.build_initial_network(list(self.values.values()))
        
        initialization_result = {
            "core_values_count": len(self.values),
            "value_types_covered": list(set(v.value_type.value for v in self.values.values())),
            "total_value_weight": sum(v.weight for v in self.values.values()),
            "network_connections": self.value_network.get_connection_count(),
            "initialization_timestamp": time.time()
        }
        
        logger.info(f"コア価値観初期化完了: {len(self.values)}個の価値観を設定")
        return initialization_result
    
    async def learn_from_experience(self, experience_data: Dict[str, Any]) -> Dict[str, Any]:
        """経験からの価値観学習"""
        logger.info(f"経験からの価値観学習開始: {experience_data.get('context', {}).get('type', 'unknown')}")
        
        # 経験オブジェクトの作成
        experience = Experience(
            experience_id=f"exp_{int(time.time())}_{len(self.experiences)}",
            timestamp=time.time(),
            context=experience_data.get("context", {}),
            actions_taken=experience_data.get("actions", []),
            outcomes=experience_data.get("outcomes", {}),
            satisfaction_level=experience_data.get("satisfaction", 0.0),
            value_relevance={},
            lessons_learned=[]
        )
        
        # 価値関連性の分析
        experience.value_relevance = await self._analyze_value_relevance(experience)
        
        # 価値葛藤の検出
        conflicts_detected = await self._detect_value_conflicts_in_experience(experience)
        
        # 価値観の調整
        value_adjustments = await self._adjust_values_from_experience(experience)
        
        # 新しい価値観の創発可能性
        emergent_values = await self._detect_emergent_values(experience)
        
        # 学習された教訓の抽出
        experience.lessons_learned = await self._extract_lessons(experience, value_adjustments)
        
        # 経験を記録
        self.experiences.append(experience)
        
        # 進化トラッカーに記録
        await self.evolution_tracker.record_learning_event(
            experience, value_adjustments, conflicts_detected, emergent_values
        )
        
        learning_result = {
            "experience_id": experience.experience_id,
            "value_adjustments": value_adjustments,
            "conflicts_detected": len(conflicts_detected),
            "emergent_values": len(emergent_values),
            "lessons_learned": experience.lessons_learned,
            "overall_impact": self._calculate_learning_impact(value_adjustments),
            "value_system_coherence": await self._assess_value_coherence()
        }
        
        logger.info(f"価値観学習完了: {len(value_adjustments)}個の調整, {len(conflicts_detected)}個の葛藤検出")
        return learning_result
    
    async def resolve_value_conflicts(self, conflict_id: str = None) -> Dict[str, Any]:
        """価値葛藤の解決"""
        if conflict_id:
            conflicts_to_resolve = [self.conflicts.get(conflict_id)]
            if not conflicts_to_resolve[0]:
                return {"error": f"Conflict {conflict_id} not found"}
        else:
            # 未解決の葛藤をすべて対象とする
            conflicts_to_resolve = [c for c in self.conflicts.values() if not c.resolved]
        
        resolution_results = []
        
        for conflict in conflicts_to_resolve:
            if not conflict:
                continue
                
            logger.info(f"価値葛藤解決開始: {conflict.conflict_id}")
            
            resolution = await self.conflict_resolver.resolve_conflict(conflict, self.values, self.meta_values)
            
            if resolution.get("resolution_successful", False):
                # 葛藤を解決済みとマーク
                conflict.resolved = True
                conflict.resolution = resolution
                
                # 価値観の更新
                await self._apply_conflict_resolution(conflict, resolution)
                
                resolution_results.append({
                    "conflict_id": conflict.conflict_id,
                    "resolution_method": resolution.get("method"),
                    "success": True,
                    "new_harmony_score": resolution.get("harmony_score", 0.0)
                })
            else:
                resolution_results.append({
                    "conflict_id": conflict.conflict_id,
                    "success": False,
                    "reason": resolution.get("failure_reason")
                })
        
        overall_harmony = await self._calculate_overall_value_harmony()
        
        return {
            "conflicts_processed": len(conflicts_to_resolve),
            "resolutions_successful": len([r for r in resolution_results if r["success"]]),
            "resolution_details": resolution_results,
            "updated_harmony_score": overall_harmony,
            "value_system_stability": await self._assess_value_stability()
        }
    
    async def evolve_value_system(self, evolution_pressure: Dict[str, Any] = None) -> Dict[str, Any]:
        """価値システム全体の進化"""
        logger.info("価値システム進化プロセス開始...")
        
        evolution_pressure = evolution_pressure or {}
        
        # 現在の価値システム状態の評価
        current_state = await self._evaluate_current_value_state()
        
        # 進化圧力の分析
        pressure_analysis = await self._analyze_evolution_pressure(evolution_pressure)
        
        # 進化戦略の決定
        evolution_strategy = await self._determine_evolution_strategy(current_state, pressure_analysis)
        
        evolution_changes = {
            "value_weight_adjustments": {},
            "new_values_created": [],
            "values_merged": [],
            "values_deprecated": [],
            "network_restructuring": False
        }
        
        # 価値重みの進化的調整
        if evolution_strategy.get("adjust_weights", False):
            weight_adjustments = await self._evolve_value_weights(pressure_analysis)
            evolution_changes["value_weight_adjustments"] = weight_adjustments
        
        # 新しい価値観の創発
        if evolution_strategy.get("create_new_values", False):
            new_values = await self._create_emergent_values(pressure_analysis)
            evolution_changes["new_values_created"] = new_values
        
        # 価値観の統合・簡素化
        if evolution_strategy.get("merge_values", False):
            merged_values = await self._merge_redundant_values()
            evolution_changes["values_merged"] = merged_values
        
        # 価値ネットワークの再構築
        if evolution_strategy.get("restructure_network", False):
            await self.value_network.evolve_network(list(self.values.values()), pressure_analysis)
            evolution_changes["network_restructuring"] = True
        
        # メタ価値の進化
        meta_value_changes = await self._evolve_meta_values(current_state, pressure_analysis)
        
        # 進化の記録
        evolution_event = {
            "timestamp": time.time(),
            "evolution_trigger": evolution_pressure,
            "changes_made": evolution_changes,
            "meta_value_changes": meta_value_changes,
            "pre_evolution_state": current_state,
            "post_evolution_state": await self._evaluate_current_value_state()
        }
        
        await self.evolution_tracker.record_evolution_event(evolution_event)
        
        logger.info("価値システム進化完了")
        return {
            "evolution_successful": True,
            "evolution_changes": evolution_changes,
            "meta_value_changes": meta_value_changes,
            "system_improvement": await self._measure_evolution_improvement(current_state),
            "new_system_coherence": await self._assess_value_coherence()
        }
    
    async def generate_value_wisdom(self) -> Dict[str, Any]:
        """価値観に基づく知恵の生成"""
        logger.info("価値ベース知恵生成開始...")
        
        # 価値経験の統合
        value_experiences = await self._integrate_value_experiences()
        
        # 価値パターンの抽出
        value_patterns = await self._extract_value_patterns()
        
        # 価値知恵の生成
        wisdom_prompt = f"""
        以下の価値体系と経験から、深い知恵を生成してください：

        価値体系: {self._format_values_for_prompt()}
        価値経験: {value_experiences}
        価値パターン: {value_patterns}

        知恵生成指針:
        1. 価値間の深い関係性の理解
        2. 人生の困難における価値選択
        3. 持続可能な価値実現の方法
        4. 価値葛藤の建設的解決
        5. 価値を通じた意味創造

        人間の価値ある生き方を支援する実践的知恵を提示してください。
        """
        
        response = await self.provider.call(wisdom_prompt, "")
        generated_wisdom = response.get("text", "")
        
        # 知恵の価値との整合性検証
        consistency_score = await self._verify_wisdom_consistency(generated_wisdom)
        
        # 知恵の実用性評価
        practicality_score = await self._evaluate_wisdom_practicality(generated_wisdom)
        
        wisdom_result = {
            "generated_wisdom": generated_wisdom,
            "consistency_with_values": consistency_score,
            "practical_applicability": practicality_score,
            "wisdom_confidence": (consistency_score + practicality_score) / 2,
            "value_foundation": self._get_dominant_values(),
            "generation_metadata": {
                "values_considered": len(self.values),
                "experiences_integrated": len(self.experiences),
                "patterns_identified": len(value_patterns)
            }
        }
        
        return wisdom_result
    
    # ==================== プライベートメソッド ====================
    
    async def _analyze_value_relevance(self, experience: Experience) -> Dict[str, float]:
        """経験と価値観の関連性分析"""
        relevance_scores = {}
        
        for value_id, value in self.values.items():
            relevance_score = 0.0
            
            # コンテキスト条件との一致度
            context_match = self._calculate_context_match(experience.context, value.context_conditions)
            relevance_score += context_match * 0.4
            
            # アウトカムと価値の一致度
            outcome_alignment = await self._calculate_outcome_alignment(experience.outcomes, value)
            relevance_score += outcome_alignment * 0.6
            
            relevance_scores[value_id] = min(1.0, relevance_score)
        
        return relevance_scores
    
    def _calculate_context_match(self, context: Dict, conditions: List[str]) -> float:
        """コンテキストマッチング計算"""
        if not conditions:
            return 0.0
        
        context_str = " ".join(str(v).lower() for v in context.values())
        matches = sum(1 for condition in conditions if condition.lower() in context_str)
        
        return matches / len(conditions)
    
    async def _calculate_outcome_alignment(self, outcomes: Dict, value: Value) -> float:
        """アウトカムと価値の一致度計算"""
        alignment_prompt = f"""
        以下の結果が価値観「{value.name}」({value.description})とどの程度一致しているか、
        0.0から1.0のスコアで評価してください：

        結果: {outcomes}
        
        評価基準:
        - 1.0: 価値観と完全に一致
        - 0.8: 価値観と強く一致
        - 0.6: 価値観と部分的に一致
        - 0.4: 価値観と弱く一致
        - 0.2: 価値観とほとんど一致しない
        - 0.0: 価値観と全く一致しない

        スコアのみ回答してください。
        """
        
        response = await self.provider.call(alignment_prompt, "")
        try:
            score = float(response.get("text", "0.0").strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.0
    
    async def _detect_value_conflicts_in_experience(self, experience: Experience) -> List[ValueConflict]:
        """経験における価値葛藤の検出"""
        conflicts = []
        
        # 高関連性の価値観を特定
        high_relevance_values = [
            (value_id, score) for value_id, score in experience.value_relevance.items() 
            if score > 0.6
        ]
        
        if len(high_relevance_values) >= 2:
            # 価値間の葛藤可能性を分析
            for i, (value_id_1, _) in enumerate(high_relevance_values):
                for value_id_2, _ in high_relevance_values[i+1:]:
                    conflict_analysis = await self._analyze_potential_conflict(
                        value_id_1, value_id_2, experience
                    )
                    
                    if conflict_analysis.get("conflict_detected", False):
                        conflict = ValueConflict(
                            conflict_id=f"conflict_{int(time.time())}_{len(self.conflicts)}",
                            conflict_type=ValueConflictType(conflict_analysis.get("conflict_type", "direct_contradiction")),
                            conflicting_values=[value_id_1, value_id_2],
                            context=experience.context,
                            intensity=conflict_analysis.get("intensity", 0.5),
                            resolution_attempts=[],
                            resolved=False
                        )
                        
                        conflicts.append(conflict)
                        self.conflicts[conflict.conflict_id] = conflict
        
        return conflicts
    
    async def _analyze_potential_conflict(self, value_id_1: str, value_id_2: str, experience: Experience) -> Dict:
        """2つの価値間の潜在的葛藤分析"""
        value_1 = self.values[value_id_1]
        value_2 = self.values[value_id_2]
        
        conflict_analysis_prompt = f"""
        以下の経験において、2つの価値観の間に葛藤があるか分析してください：

        価値観1: {value_1.name} - {value_1.description}
        価値観2: {value_2.name} - {value_2.description}
        
        経験: {experience.context}
        結果: {experience.outcomes}
        満足度: {experience.satisfaction_level}

        分析項目:
        1. 葛藤の存在 (true/false)
        2. 葛藤の種類 (direct_contradiction/priority_conflict/context_dependent/emergent_tension)
        3. 葛藤の強度 (0.0-1.0)

        JSON形式で回答してください。
        """
        
        response = await self.provider.call(conflict_analysis_prompt, "")
        try:
            analysis = json.loads(response.get("text", "{}"))
            return {
                "conflict_detected": analysis.get("conflict_exists", False),
                "conflict_type": analysis.get("conflict_type", "direct_contradiction"),
                "intensity": float(analysis.get("intensity", 0.0))
            }
        except (json.JSONDecodeError, ValueError):
            return {"conflict_detected": False}
    
    async def _adjust_values_from_experience(self, experience: Experience) -> Dict[str, Dict]:
        """経験に基づく価値観調整"""
        adjustments = {}
        
        for value_id, relevance in experience.value_relevance.items():
            if relevance > 0.3:  # 関連性がある場合のみ調整
                value = self.values[value_id]
                
                # 満足度に基づく重み調整
                satisfaction_impact = experience.satisfaction_level * relevance
                weight_adjustment = satisfaction_impact * 0.05  # 最大5%の調整
                
                new_weight = max(0.0, min(1.0, value.weight + weight_adjustment))
                
                if abs(weight_adjustment) > 0.01:  # 有意な変更の場合のみ適用
                    old_weight = value.weight
                    value.weight = new_weight
                    value.last_updated = time.time()
                    
                    # 進化履歴に記録
                    value.evolution_history.append({
                        "timestamp": time.time(),
                        "event": "experience_based_adjustment",
                        "experience_id": experience.experience_id,
                        "previous_weight": old_weight,
                        "new_weight": new_weight,
                        "adjustment_reason": f"satisfaction_impact: {satisfaction_impact:.3f}"
                    })
                    
                    adjustments[value_id] = {
                        "old_weight": old_weight,
                        "new_weight": new_weight,
                        "adjustment": weight_adjustment,
                        "reason": "experience_satisfaction"
                    }
        
        return adjustments
    
    async def _detect_emergent_values(self, experience: Experience) -> List[Dict]:
        """新しい価値観の創発検出"""
        emergent_values = []
        
        # 既存価値と低関連性だが高満足度の経験は新価値の兆候
        low_relevance_sum = sum(experience.value_relevance.values())
        
        if low_relevance_sum < 2.0 and experience.satisfaction_level > 0.6:
            # 新価値候補の分析
            emergence_analysis = await self._analyze_value_emergence(experience)
            
            if emergence_analysis.get("new_value_detected", False):
                emergent_values.append(emergence_analysis)
        
        return emergent_values
    
    async def _analyze_value_emergence(self, experience: Experience) -> Dict:
        """価値創発分析"""
        emergence_prompt = f"""
        以下の経験から、新しい価値観が創発している可能性を分析してください：

        経験: {experience.context}
        行動: {experience.actions_taken}
        結果: {experience.outcomes}
        満足度: {experience.satisfaction_level}

        既存価値との関連性が低いにも関わらず満足度が高い場合、
        新しい価値観が創発している可能性があります。

        分析項目:
        1. 新価値の存在可能性 (true/false)
        2. 新価値の名称案
        3. 新価値の説明
        4. 新価値の種類 (ethical/aesthetic/pragmatic/epistemic/social/existential)

        JSON形式で回答してください。
        """
        
        response = await self.provider.call(emergence_prompt, "")
        try:
            analysis = json.loads(response.get("text", "{}"))
            return {
                "new_value_detected": analysis.get("new_value_exists", False),
                "value_name": analysis.get("value_name", ""),
                "value_description": analysis.get("value_description", ""),
                "value_type": analysis.get("value_type", "pragmatic")
            }
        except (json.JSONDecodeError, ValueError):
            return {"new_value_detected": False}
    
    async def _extract_lessons(self, experience: Experience, adjustments: Dict) -> List[str]:
        """経験からの教訓抽出"""
        lessons_prompt = f"""
        以下の経験と価値観調整から、重要な教訓を抽出してください：

        経験: {experience.context}
        結果: {experience.outcomes}
        満足度: {experience.satisfaction_level}
        価値調整: {adjustments}

        教訓抽出指針:
        1. 価値実現の効果的方法
        2. 価値葛藤の解決策
        3. 予期しない価値発見
        4. 価値実践の改善点

        教訓を箇条書きで列挙してください。
        """
        
        response = await self.provider.call(lessons_prompt, "")
        lessons_text = response.get("text", "")
        
        lessons = []
        for line in lessons_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                lesson = line.lstrip('-•* ').strip()
                if lesson:
                    lessons.append(lesson)
        
        return lessons[:5]  # 上位5つの教訓
    
    def _calculate_learning_impact(self, adjustments: Dict) -> float:
        """学習インパクトの計算"""
        if not adjustments:
            return 0.0
        
        total_impact = sum(abs(adj["adjustment"]) for adj in adjustments.values())
        return min(1.0, total_impact)
    
    async def _assess_value_coherence(self) -> float:
        """価値システムの一貫性評価"""
        if len(self.values) < 2:
            return 1.0
        
        coherence_scores = []
        
        # ペアワイズ一貫性の計算
        value_list = list(self.values.values())
        for i, value_1 in enumerate(value_list):
            for value_2 in value_list[i+1:]:
                coherence_score = await self._calculate_pairwise_coherence(value_1, value_2)
                coherence_scores.append(coherence_score)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 1.0
    
    async def _calculate_pairwise_coherence(self, value_1: Value, value_2: Value) -> float:
        """2つの価値間の一貫性計算"""
        # 重み差による一貫性（類似価値は類似重みを持つべき）
        weight_coherence = 1.0 - abs(value_1.weight - value_2.weight)
        
        # コンテキスト重複による関連性
        context_overlap = len(set(value_1.context_conditions) & set(value_2.context_conditions))
        max_contexts = max(len(value_1.context_conditions), len(value_2.context_conditions))
        context_coherence = context_overlap / max_contexts if max_contexts > 0 else 0.0
        
        return (weight_coherence * 0.7 + context_coherence * 0.3)
    
    def _format_values_for_prompt(self) -> str:
        """プロンプト用価値フォーマット"""
        value_descriptions = []
        for value in self.values.values():
            value_descriptions.append(
                f"- {value.name} ({value.value_type.value}): {value.description} [重み: {value.weight:.2f}]"
            )
        return "\n".join(value_descriptions)
    
    def _get_dominant_values(self) -> List[str]:
        """支配的価値観の取得"""
        sorted_values = sorted(self.values.values(), key=lambda v: v.weight, reverse=True)
        return [v.name for v in sorted_values[:3]]  # 上位3つ
    
    # 残りのメソッドは実装を簡略化...
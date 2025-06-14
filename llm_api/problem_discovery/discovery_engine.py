# /llm_api/problem_discovery/discovery_engine.py
"""
Problem Discovery Engine
人間が気づかない潜在的問題を発見するシステム

このエンジンは「知的システムの知的システム」の一部として、
データパターンや異常から隠れた問題を創発的に発見する能力を提供します。
"""

import asyncio
import logging
import json
import time
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import re

logger = logging.getLogger(__name__)

class ProblemSeverity(Enum):
    """問題の深刻度"""
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"
    EXISTENTIAL = "existential"

class ProblemType(Enum):
    """問題の種類"""
    SYSTEMIC = "systemic"           # システム的問題
    EMERGENT = "emergent"           # 創発的問題
    LATENT = "latent"               # 潜在的問題
    CASCADING = "cascading"         # 連鎖的問題
    PARADOXICAL = "paradoxical"     # パラドックス的問題
    EXISTENTIAL = "existential"     # 実存的問題

class DiscoveryMethod(Enum):
    """発見手法"""
    PATTERN_ANALYSIS = "pattern_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    CAUSAL_INFERENCE = "causal_inference"
    EMERGENT_SYNTHESIS = "emergent_synthesis"
    PARADOX_IDENTIFICATION = "paradox_identification"
    META_ANALYSIS = "meta_analysis"

@dataclass
class DiscoveredProblem:
    """発見された問題"""
    problem_id: str
    title: str
    description: str
    problem_type: ProblemType
    severity: ProblemSeverity
    discovery_method: DiscoveryMethod
    evidence: List[Dict[str, Any]]
    affected_domains: List[str]
    potential_impacts: List[str]
    confidence_score: float
    urgency_score: float
    discovery_timestamp: float
    validation_status: str = "pending"
    validation_evidence: List[Dict] = field(default_factory=list)

@dataclass
class DataPattern:
    """データパターン"""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: float
    statistical_significance: float
    temporal_characteristics: Dict[str, Any]
    anomaly_indicators: List[str]
    correlation_networks: List[Tuple[str, str, float]]

class ProblemDiscoveryEngine:
    """問題発見エンジン"""
    
    def __init__(self, provider):
        self.provider = provider
        self.discovered_problems: Dict[str, DiscoveredProblem] = {}
        self.data_patterns: Dict[str, DataPattern] = {}
        self.anomaly_detector = AnomalyDetector()
        self.pattern_analyzer = PatternAnalyzer()
        self.causal_inferencer = CausalInferencer(provider)
        self.meta_analyzer = MetaAnalyzer(provider)
        
        # 発見履歴とメタデータ
        self.discovery_history = deque(maxlen=1000)
        self.validation_history = deque(maxlen=500)
        self.domain_knowledge = {}
        
        # 発見パラメータ
        self.discovery_sensitivity = 0.7  # 感度調整
        self.confidence_threshold = 0.6   # 信頼度閾値
        self.novelty_threshold = 0.8      # 新規性閾値
        
        logger.info("問題発見エンジンを初期化しました")
    
    async def discover_problems_from_data(self, data_sources: List[Dict[str, Any]], 
                                        domain_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """データから問題を発見"""
        logger.info(f"データからの問題発見開始: {len(data_sources)}個のデータソース")
        
        domain_context = domain_context or {}
        discovered_problems = []
        
        # 1. データパターンの抽出
        patterns = await self._extract_data_patterns(data_sources)
        
        # 2. 異常の検出
        anomalies = await self._detect_anomalies(data_sources, patterns)
        
        # 3. 因果関係の推論
        causal_networks = await self._infer_causal_relationships(data_sources, patterns)
        
        # 4. 各発見手法による問題発見
        discovery_methods = [
            ("pattern_analysis", self._discover_from_patterns),
            ("anomaly_detection", self._discover_from_anomalies),
            ("causal_inference", self._discover_from_causality),
            ("emergent_synthesis", self._discover_emergent_problems),
            ("meta_analysis", self._discover_meta_problems)
        ]
        
        for method_name, method_func in discovery_methods:
            try:
                method_problems = await method_func(data_sources, patterns, anomalies, causal_networks, domain_context)
                discovered_problems.extend(method_problems)
                logger.info(f"{method_name}により{len(method_problems)}個の問題を発見")
            except Exception as e:
                logger.error(f"{method_name}での問題発見中にエラー: {e}")
        
        # 5. 問題の重複除去と統合
        unique_problems = await self._deduplicate_and_merge_problems(discovered_problems)
        
        # 6. 問題の優先順位付け
        prioritized_problems = await self._prioritize_problems(unique_problems)
        
        # 7. 問題の登録
        for problem in prioritized_problems:
            self.discovered_problems[problem.problem_id] = problem
            self.discovery_history.append({
                "timestamp": time.time(),
                "problem_id": problem.problem_id,
                "discovery_method": problem.discovery_method.value,
                "confidence": problem.confidence_score
            })
        
        # 8. 発見結果の分析
        discovery_analysis = await self._analyze_discovery_results(prioritized_problems)
        
        discovery_result = {
            "problems_discovered": len(prioritized_problems),
            "problem_details": [self._format_problem_summary(p) for p in prioritized_problems],
            "discovery_methods_used": list(set(p.discovery_method.value for p in prioritized_problems)),
            "severity_distribution": self._calculate_severity_distribution(prioritized_problems),
            "high_priority_problems": [p for p in prioritized_problems if p.urgency_score > 0.8],
            "discovery_confidence": discovery_analysis.get("overall_confidence", 0.0),
            "novel_insights": discovery_analysis.get("novel_insights", []),
            "cross_domain_implications": discovery_analysis.get("cross_domain_implications", [])
        }
        
        logger.info(f"問題発見完了: {len(prioritized_problems)}個の問題を発見")
        return discovery_result
    
    async def validate_discovered_problem(self, problem_id: str, 
                                        validation_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """発見された問題の検証"""
        if problem_id not in self.discovered_problems:
            return {"error": f"Problem {problem_id} not found"}
        
        problem = self.discovered_problems[problem_id]
        logger.info(f"問題検証開始: {problem.title}")
        
        validation_data = validation_data or {}
        
        # 検証方法の決定
        validation_methods = await self._determine_validation_methods(problem)
        
        validation_results = {}
        overall_validation_score = 0.0
        
        for method in validation_methods:
            try:
                if method == "evidence_verification":
                    result = await self._verify_evidence(problem, validation_data)
                elif method == "impact_assessment":
                    result = await self._assess_actual_impact(problem, validation_data)
                elif method == "expert_simulation":
                    result = await self._simulate_expert_validation(problem)
                elif method == "predictive_validation":
                    result = await self._validate_through_prediction(problem, validation_data)
                else:
                    result = {"score": 0.5, "method": method, "details": "Method not implemented"}
                
                validation_results[method] = result
                overall_validation_score += result.get("score", 0.0)
                
            except Exception as e:
                logger.error(f"検証方法{method}でエラー: {e}")
                validation_results[method] = {"score": 0.0, "error": str(e)}
        
        # 総合検証スコアの計算
        if validation_results:
            overall_validation_score /= len(validation_results)
        
        # 検証状態の更新
        if overall_validation_score > 0.8:
            problem.validation_status = "validated"
        elif overall_validation_score > 0.6:
            problem.validation_status = "partially_validated"
        else:
            problem.validation_status = "questionable"
        
        # 検証証拠の追加
        problem.validation_evidence.append({
            "timestamp": time.time(),
            "validation_score": overall_validation_score,
            "validation_methods": validation_results,
            "validation_data": validation_data
        })
        
        # 検証履歴に記録
        self.validation_history.append({
            "timestamp": time.time(),
            "problem_id": problem_id,
            "validation_score": overall_validation_score,
            "status": problem.validation_status
        })
        
        return {
            "problem_id": problem_id,
            "validation_score": overall_validation_score,
            "validation_status": problem.validation_status,
            "validation_details": validation_results,
            "confidence_update": problem.confidence_score,
            "validation_summary": await self._generate_validation_summary(problem, validation_results)
        }
    
    async def generate_problem_insights(self, focus_domain: str = None) -> Dict[str, Any]:
        """問題群からの洞察生成"""
        logger.info(f"問題洞察生成開始: {focus_domain or '全領域'}")
        
        # 分析対象問題の選択
        if focus_domain:
            target_problems = [p for p in self.discovered_problems.values() 
                             if focus_domain in p.affected_domains]
        else:
            target_problems = list(self.discovered_problems.values())
        
        if not target_problems:
            return {"insights": [], "reason": "No problems found for analysis"}
        
        # 問題群の分析
        problem_analysis = await self._analyze_problem_clusters(target_problems)
        
        # パターンの抽出
        meta_patterns = await self._extract_meta_patterns(target_problems)
        
        # 根本原因の推論
        root_causes = await self._infer_root_causes(target_problems)
        
        # システム的脆弱性の特定
        systemic_vulnerabilities = await self._identify_systemic_vulnerabilities(target_problems)
        
        # 創発的洞察の生成
        emergent_insights = await self._generate_emergent_insights(
            target_problems, problem_analysis, meta_patterns, root_causes
        )
        
        # 予防的提案の生成
        preventive_recommendations = await self._generate_preventive_recommendations(
            root_causes, systemic_vulnerabilities
        )
        
        insights_result = {
            "domain": focus_domain or "universal",
            "problems_analyzed": len(target_problems),
            "problem_clusters": problem_analysis.get("clusters", []),
            "meta_patterns": meta_patterns,
            "root_causes": root_causes,
            "systemic_vulnerabilities": systemic_vulnerabilities,
            "emergent_insights": emergent_insights,
            "preventive_recommendations": preventive_recommendations,
            "insight_confidence": await self._calculate_insight_confidence(target_problems),
            "cross_domain_connections": await self._identify_cross_domain_connections(target_problems)
        }
        
        return insights_result
    
    async def evolve_discovery_capabilities(self, feedback_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """発見能力の進化"""
        logger.info("問題発見能力の進化プロセス開始...")
        
        feedback_data = feedback_data or {}
        
        # 現在の発見性能の評価
        current_performance = await self._evaluate_discovery_performance()
        
        # 発見パターンの学習
        pattern_learning = await self._learn_discovery_patterns()
        
        # パラメータの最適化
        parameter_optimization = await self._optimize_discovery_parameters(feedback_data)
        
        # 新しい発見手法の開発
        new_methods = await self._develop_new_discovery_methods()
        
        # 能力向上の実装
        evolution_changes = {}
        
        # 感度調整
        if parameter_optimization.get("sensitivity_adjustment"):
            old_sensitivity = self.discovery_sensitivity
            self.discovery_sensitivity = parameter_optimization["new_sensitivity"]
            evolution_changes["sensitivity"] = {
                "old": old_sensitivity,
                "new": self.discovery_sensitivity
            }
        
        # 閾値調整
        if parameter_optimization.get("threshold_adjustments"):
            evolution_changes["thresholds"] = parameter_optimization["threshold_adjustments"]
        
        # 新手法の追加
        if new_methods:
            evolution_changes["new_methods"] = new_methods
        
        # 進化結果の評価
        post_evolution_performance = await self._evaluate_discovery_performance()
        
        evolution_result = {
            "evolution_successful": True,
            "performance_improvement": {
                "before": current_performance,
                "after": post_evolution_performance,
                "improvement_score": post_evolution_performance.get("overall_score", 0) - 
                                   current_performance.get("overall_score", 0)
            },
            "evolution_changes": evolution_changes,
            "pattern_learning": pattern_learning,
            "new_capabilities": new_methods,
            "evolution_timestamp": time.time()
        }
        
        logger.info("問題発見能力の進化完了")
        return evolution_result
    
    # ==================== プライベートメソッド ====================
    
    async def _extract_data_patterns(self, data_sources: List[Dict]) -> List[DataPattern]:
        """データパターンの抽出"""
        patterns = []
        
        for source in data_sources:
            try:
                # 時系列パターンの抽出
                temporal_patterns = await self.pattern_analyzer.extract_temporal_patterns(source)
                patterns.extend(temporal_patterns)
                
                # 統計的パターンの抽出
                statistical_patterns = await self.pattern_analyzer.extract_statistical_patterns(source)
                patterns.extend(statistical_patterns)
                
                # 相関パターンの抽出
                correlation_patterns = await self.pattern_analyzer.extract_correlation_patterns(source)
                patterns.extend(correlation_patterns)
                
            except Exception as e:
                logger.error(f"データソース{source.get('name', 'unknown')}のパターン抽出エラー: {e}")
        
        return patterns
    
    async def _detect_anomalies(self, data_sources: List[Dict], patterns: List[DataPattern]) -> List[Dict]:
        """異常検出"""
        anomalies = []
        
        for source in data_sources:
            try:
                source_anomalies = await self.anomaly_detector.detect_anomalies(source, patterns)
                anomalies.extend(source_anomalies)
            except Exception as e:
                logger.error(f"異常検出エラー: {e}")
        
        return anomalies
    
    async def _infer_causal_relationships(self, data_sources: List[Dict], 
                                        patterns: List[DataPattern]) -> Dict[str, Any]:
        """因果関係の推論"""
        try:
            causal_networks = await self.causal_inferencer.infer_causality(data_sources, patterns)
            return causal_networks
        except Exception as e:
            logger.error(f"因果推論エラー: {e}")
            return {}
    
    async def _discover_from_patterns(self, data_sources: List[Dict], patterns: List[DataPattern],
                                    anomalies: List[Dict], causal_networks: Dict,
                                    domain_context: Dict) -> List[DiscoveredProblem]:
        """パターンからの問題発見"""
        problems = []
        
        for pattern in patterns:
            # 異常なパターンの識別
            if pattern.statistical_significance < 0.05 and pattern.frequency > 0.1:
                problem = await self._create_pattern_based_problem(pattern, domain_context)
                if problem:
                    problems.append(problem)
        
        return problems
    
    async def _discover_from_anomalies(self, data_sources: List[Dict], patterns: List[DataPattern],
                                     anomalies: List[Dict], causal_networks: Dict,
                                     domain_context: Dict) -> List[DiscoveredProblem]:
        """異常からの問題発見"""
        problems = []
        
        # 異常をクラスタリング
        anomaly_clusters = await self._cluster_anomalies(anomalies)
        
        for cluster in anomaly_clusters:
            if len(cluster) >= 3:  # 複数の関連異常
                problem = await self._create_anomaly_based_problem(cluster, domain_context)
                if problem:
                    problems.append(problem)
        
        return problems
    
    async def _discover_from_causality(self, data_sources: List[Dict], patterns: List[DataPattern],
                                     anomalies: List[Dict], causal_networks: Dict,
                                     domain_context: Dict) -> List[DiscoveredProblem]:
        """因果関係からの問題発見"""
        problems = []
        
        # 負の因果チェーンの検出
        negative_chains = causal_networks.get("negative_causal_chains", [])
        
        for chain in negative_chains:
            if len(chain) >= 3:  # 長い負の因果チェーン
                problem = await self._create_causal_problem(chain, domain_context)
                if problem:
                    problems.append(problem)
        
        return problems
    
    async def _discover_emergent_problems(self, data_sources: List[Dict], patterns: List[DataPattern],
                                        anomalies: List[Dict], causal_networks: Dict,
                                        domain_context: Dict) -> List[DiscoveredProblem]:
        """創発的問題の発見"""
        problems = []
        
        # 複雑性の創発的分析
        emergence_analysis = await self._analyze_system_emergence(data_sources, patterns, causal_networks)
        
        for emergence in emergence_analysis.get("emergent_behaviors", []):
            if emergence.get("potentially_problematic", False):
                problem = await self._create_emergent_problem(emergence, domain_context)
                if problem:
                    problems.append(problem)
        
        return problems
    
    async def _discover_meta_problems(self, data_sources: List[Dict], patterns: List[DataPattern],
                                    anomalies: List[Dict], causal_networks: Dict,
                                    domain_context: Dict) -> List[DiscoveredProblem]:
        """メタレベル問題の発見"""
        problems = []
        
        # システム全体のメタ分析
        meta_analysis = await self.meta_analyzer.analyze_system_meta_properties(
            data_sources, patterns, anomalies, causal_networks
        )
        
        for meta_issue in meta_analysis.get("meta_level_issues", []):
            problem = await self._create_meta_problem(meta_issue, domain_context)
            if problem:
                problems.append(problem)
        
        return problems
    
    async def _create_pattern_based_problem(self, pattern: DataPattern, context: Dict) -> Optional[DiscoveredProblem]:
        """パターンベース問題の作成"""
        problem_prompt = f"""
        以下のデータパターンから潜在的問題を特定してください：

        パターン: {pattern.description}
        頻度: {pattern.frequency}
        統計的有意性: {pattern.statistical_significance}
        異常指標: {pattern.anomaly_indicators}

        問題特定指針:
        1. パターンが示す潜在的リスク
        2. 将来的な悪影響の可能性
        3. 見過ごされがちな問題点
        4. システム的脆弱性

        JSON形式で問題詳細を返してください。
        """
        
        response = await self.provider.call(problem_prompt, "")
        try:
            problem_data = json.loads(response.get("text", "{}"))
            
            if problem_data.get("problem_identified", False):
                return DiscoveredProblem(
                    problem_id=f"pattern_{int(time.time())}_{len(self.discovered_problems)}",
                    title=problem_data.get("title", "パターンベース問題"),
                    description=problem_data.get("description", ""),
                    problem_type=ProblemType.SYSTEMIC,
                    severity=ProblemSeverity(problem_data.get("severity", "moderate")),
                    discovery_method=DiscoveryMethod.PATTERN_ANALYSIS,
                    evidence=[{"pattern": pattern.__dict__}],
                    affected_domains=problem_data.get("affected_domains", []),
                    potential_impacts=problem_data.get("potential_impacts", []),
                    confidence_score=float(pattern.statistical_significance),
                    urgency_score=float(problem_data.get("urgency", 0.5)),
                    discovery_timestamp=time.time()
                )
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"パターンベース問題作成エラー: {e}")
        
        return None
    
    def _format_problem_summary(self, problem: DiscoveredProblem) -> Dict[str, Any]:
        """問題サマリーのフォーマット"""
        return {
            "problem_id": problem.problem_id,
            "title": problem.title,
            "type": problem.problem_type.value,
            "severity": problem.severity.value,
            "confidence": problem.confidence_score,
            "urgency": problem.urgency_score,
            "affected_domains": problem.affected_domains,
            "discovery_method": problem.discovery_method.value,
            "validation_status": problem.validation_status
        }
    
    def _calculate_severity_distribution(self, problems: List[DiscoveredProblem]) -> Dict[str, int]:
        """深刻度分布の計算"""
        distribution = defaultdict(int)
        for problem in problems:
            distribution[problem.severity.value] += 1
        return dict(distribution)
    
    # 簡略化された残りのメソッド
    async def _deduplicate_and_merge_problems(self, problems: List[DiscoveredProblem]) -> List[DiscoveredProblem]:
        """重複除去と統合"""
        # 簡略化: タイトルの類似性で重複判定
        unique_problems = []
        seen_titles = set()
        
        for problem in problems:
            if problem.title not in seen_titles:
                unique_problems.append(problem)
                seen_titles.add(problem.title)
        
        return unique_problems
    
    async def _prioritize_problems(self, problems: List[DiscoveredProblem]) -> List[DiscoveredProblem]:
        """問題の優先順位付け"""
        # 緊急度と信頼度の組み合わせでソート
        return sorted(problems, 
                     key=lambda p: p.urgency_score * p.confidence_score, 
                     reverse=True)

# ==================== 補助クラス ====================

class AnomalyDetector:
    """異常検出器"""
    
    async def detect_anomalies(self, data_source: Dict, patterns: List[DataPattern]) -> List[Dict]:
        """異常の検出"""
        anomalies = []
        
        # 簡略化された異常検出
        data_values = data_source.get("values", [])
        if data_values and isinstance(data_values[0], (int, float)):
            mean = np.mean(data_values)
            std = np.std(data_values)
            threshold = mean + 2 * std
            
            for i, value in enumerate(data_values):
                if abs(value - mean) > threshold:
                    anomalies.append({
                        "index": i,
                        "value": value,
                        "deviation": abs(value - mean),
                        "type": "statistical_outlier"
                    })
        
        return anomalies

class PatternAnalyzer:
    """パターン分析器"""
    
    async def extract_temporal_patterns(self, data_source: Dict) -> List[DataPattern]:
        """時系列パターンの抽出"""
        patterns = []
        
        # 簡略化された時系列パターン検出
        if "timestamps" in data_source and "values" in data_source:
            pattern = DataPattern(
                pattern_id=f"temporal_{int(time.time())}",
                pattern_type="temporal",
                description="時系列トレンドパターン",
                frequency=0.8,
                statistical_significance=0.05,
                temporal_characteristics={"trend": "increasing"},
                anomaly_indicators=["sudden_changes"],
                correlation_networks=[]
            )
            patterns.append(pattern)
        
        return patterns
    
    async def extract_statistical_patterns(self, data_source: Dict) -> List[DataPattern]:
        """統計的パターンの抽出"""
        return []  # 簡略化
    
    async def extract_correlation_patterns(self, data_source: Dict) -> List[DataPattern]:
        """相関パターンの抽出"""
        return []  # 簡略化

class CausalInferencer:
    """因果推論器"""
    
    def __init__(self, provider):
        self.provider = provider
    
    async def infer_causality(self, data_sources: List[Dict], patterns: List[DataPattern]) -> Dict[str, Any]:
        """因果関係の推論"""
        # 簡略化された因果推論
        return {
            "causal_relationships": [],
            "negative_causal_chains": [],
            "feedback_loops": []
        }

class MetaAnalyzer:
    """メタ分析器"""
    
    def __init__(self, provider):
        self.provider = provider
    
    async def analyze_system_meta_properties(self, data_sources: List[Dict], 
                                          patterns: List[DataPattern],
                                          anomalies: List[Dict], 
                                          causal_networks: Dict) -> Dict[str, Any]:
        """システムのメタ特性分析"""
        return {
            "meta_level_issues": [],
            "system_complexity": 0.7,
            "emergent_properties": [],
            "systemic_risks": []
        }
# /llm_api/autonomous_learning/web_crawler.py
"""
Autonomous Web Learning System
自律的なWeb巡回と学習を行うシステム

このシステムは「知的システムの知的システム」の一部として、
Webから興味深い情報を自律的に発見・学習する能力を提供します。
"""

import asyncio
import logging
import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from urllib.parse import urljoin, urlparse
import re

logger = logging.getLogger(__name__)

class InterestLevel(Enum):
    """興味レベルの定義"""
    VERY_LOW = 0.1
    LOW = 0.3
    MODERATE = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9

class ContentType(Enum):
    """コンテンツタイプ"""
    ARTICLE = "article"
    RESEARCH_PAPER = "research_paper"
    NEWS = "news"
    TUTORIAL = "tutorial"
    DOCUMENTATION = "documentation"
    FORUM_DISCUSSION = "forum_discussion"
    BLOG_POST = "blog_post"

@dataclass
class WebContent:
    """Webコンテンツの表現"""
    url: str
    title: str
    content: str
    content_type: ContentType
    discovery_timestamp: float
    interest_score: float
    learning_value: float
    summary: str
    key_concepts: List[str]
    related_topics: List[str]
    source_credibility: float

@dataclass
class LearningGoal:
    """学習目標"""
    goal_id: str
    description: str
    priority: float
    related_keywords: List[str]
    progress: float
    target_knowledge_areas: List[str]

class InterestProfiler:
    """興味プロファイリングシステム"""
    
    def __init__(self, provider):
        self.provider = provider
        self.interest_patterns = {}
        self.learned_preferences = defaultdict(float)
        self.topic_importance = defaultdict(float)
        
    async def evaluate_content_interest(self, content: str, metadata: Dict) -> Tuple[float, List[str]]:
        """コンテンツの興味度評価"""
        
        # 基本的な興味度評価
        basic_interest = await self._basic_interest_assessment(content, metadata)
        
        # 学習価値の評価
        learning_value = await self._assess_learning_value(content)
        
        # 新規性の評価
        novelty_score = await self._assess_novelty(content)
        
        # 関連性の評価
        relevance_score = await self._assess_relevance(content)
        
        # 総合興味度計算
        overall_interest = (
            basic_interest * 0.3 +
            learning_value * 0.3 +
            novelty_score * 0.2 +
            relevance_score * 0.2
        )
        
        # 興味のあるトピック抽出
        interesting_topics = await self._extract_interesting_topics(content)
        
        return min(1.0, overall_interest), interesting_topics
    
    async def _basic_interest_assessment(self, content: str, metadata: Dict) -> float:
        """基本興味度評価"""
        interest_prompt = f"""
        以下のコンテンツについて、AI・機械学習・認知科学・哲学・未来技術の観点から
        興味度を0.0-1.0で評価してください：

        タイトル: {metadata.get('title', '')}
        コンテンツ: {content[:1000]}...

        評価基準:
        - 1.0: 革新的で重要な内容
        - 0.8: 非常に興味深い
        - 0.6: 興味深い
        - 0.4: やや興味深い
        - 0.2: あまり興味なし
        - 0.0: 全く興味なし

        スコアのみ回答してください。
        """
        
        response = await self.provider.call(interest_prompt, "")
        try:
            return max(0.0, min(1.0, float(response.get("text", "0.0").strip())))
        except ValueError:
            return 0.0
    
    async def _assess_learning_value(self, content: str) -> float:
        """学習価値評価"""
        learning_indicators = [
            "研究", "実験", "発見", "新しい", "革新", "方法", "技術",
            "アルゴリズム", "理論", "仮説", "証明", "分析", "データ"
        ]
        
        content_lower = content.lower()
        indicator_count = sum(1 for indicator in learning_indicators if indicator in content_lower)
        
        return min(1.0, indicator_count / len(learning_indicators) * 2)
    
    async def _assess_novelty(self, content: str) -> float:
        """新規性評価"""
        # 既存の学習内容との比較（簡略化）
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # 類似コンテンツの検索（実装時にはより高度な類似性計算が必要）
        novelty_score = 0.8  # デフォルト値
        
        return novelty_score
    
    async def _assess_relevance(self, content: str) -> float:
        """関連性評価"""
        # 現在の学習目標との関連性
        return 0.7  # 簡略化
    
    async def _extract_interesting_topics(self, content: str) -> List[str]:
        """興味深いトピックの抽出"""
        topic_prompt = f"""
        以下のコンテンツから主要なトピック・概念を5個以下抽出してください：

        {content[:1500]}

        トピックを箇条書きで列挙してください。
        """
        
        response = await self.provider.call(topic_prompt, "")
        topics_text = response.get("text", "")
        
        topics = []
        for line in topics_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•')):
                topic = line.lstrip('-•* ').strip()
                if topic:
                    topics.append(topic)
        
        return topics[:5]

class AutonomousWebCrawler:
    """自律的Webクローラー"""
    
    def __init__(self, provider, web_search_func, web_fetch_func):
        self.provider = provider
        self.web_search = web_search_func
        self.web_fetch = web_fetch_func
        self.interest_profiler = InterestProfiler(provider)
        
        # 学習状態
        self.discovered_content = deque(maxlen=1000)
        self.learned_knowledge = {}
        self.exploration_history = deque(maxlen=500)
        self.learning_goals = {}
        
        # 探索パラメータ
        self.min_interest_threshold = 0.6
        self.max_crawl_depth = 3
        self.max_pages_per_session = 20
        
        # 探索戦略
        self.exploration_strategies = [
            "follow_interesting_links",
            "search_related_topics", 
            "explore_authoritative_sources",
            "discover_trending_topics"
        ]
        
    async def start_autonomous_learning(self, initial_topics: List[str] = None, 
                                      session_duration: int = 3600) -> Dict[str, Any]:
        """自律学習セッションの開始"""
        logger.info(f"自律学習セッション開始: {session_duration}秒間")
        
        session_start = time.time()
        session_id = f"session_{int(session_start)}"
        
        # 初期トピックの設定
        if not initial_topics:
            initial_topics = [
                "artificial intelligence", "machine learning", "cognitive science",
                "neural networks", "consciousness", "AI safety", "future technology"
            ]
        
        session_results = {
            "session_id": session_id,
            "content_discovered": [],
            "knowledge_gained": [],
            "new_interests": [],
            "exploration_paths": [],
            "learning_achievements": []
        }
        
        pages_crawled = 0
        current_topics = initial_topics.copy()
        
        try:
            while (time.time() - session_start < session_duration and 
                   pages_crawled < self.max_pages_per_session):
                
                # 探索戦略の選択
                strategy = await self._select_exploration_strategy(current_topics)
                
                # コンテンツの発見
                discovered_urls = await self._discover_content(current_topics, strategy)
                
                for url in discovered_urls[:5]:  # 最大5ページずつ処理
                    if pages_crawled >= self.max_pages_per_session:
                        break
                    
                    # コンテンツの取得と分析
                    content_analysis = await self._analyze_discovered_content(url)
                    
                    if content_analysis and content_analysis.interest_score >= self.min_interest_threshold:
                        # 興味深いコンテンツの学習
                        learning_result = await self._learn_from_content(content_analysis)
                        
                        session_results["content_discovered"].append({
                            "url": url,
                            "title": content_analysis.title,
                            "interest_score": content_analysis.interest_score,
                            "key_concepts": content_analysis.key_concepts
                        })
                        
                        session_results["knowledge_gained"].extend(learning_result.get("new_knowledge", []))
                        
                        # 新しい探索トピックの追加
                        current_topics.extend(content_analysis.related_topics)
                        current_topics = list(set(current_topics))[:20]  # 重複除去と制限
                    
                    pages_crawled += 1
                    await asyncio.sleep(1)  # レート制限
                
                # トピックの進化
                current_topics = await self._evolve_exploration_topics(current_topics, session_results)
        
        except Exception as e:
            logger.error(f"自律学習セッション中にエラー: {e}")
        
        # セッション結果の分析
        session_analysis = await self._analyze_session_results(session_results)
        
        logger.info(f"自律学習セッション完了: {pages_crawled}ページ探索, {len(session_results['content_discovered'])}個の興味深いコンテンツを発見")
        
        return {
            "session_summary": session_results,
            "session_analysis": session_analysis,
            "pages_crawled": pages_crawled,
            "duration": time.time() - session_start,
            "learning_efficiency": len(session_results["knowledge_gained"]) / max(pages_crawled, 1)
        }
    
    async def _select_exploration_strategy(self, current_topics: List[str]) -> str:
        """探索戦略の選択"""
        # 簡略化：ランダム選択
        import random
        return random.choice(self.exploration_strategies)
    
    async def _discover_content(self, topics: List[str], strategy: str) -> List[str]:
        """コンテンツの発見"""
        discovered_urls = []
        
        try:
            # トピックベースの検索
            for topic in topics[:3]:  # 最大3トピック
                search_query = f"{topic} research latest developments"
                search_results = await self.web_search(search_query)
                
                if search_results and 'results' in search_results:
                    for result in search_results['results'][:3]:  # 各トピック3件まで
                        if 'link' in result:
                            discovered_urls.append(result['link'])
        
        except Exception as e:
            logger.error(f"コンテンツ発見中にエラー: {e}")
        
        return discovered_urls
    
    async def _analyze_discovered_content(self, url: str) -> Optional[WebContent]:
        """発見されたコンテンツの分析"""
        try:
            # ページの取得
            page_result = await self.web_fetch(url)
            
            if not page_result or page_result.get('error'):
                return None
            
            content = page_result.get('content', '')
            title = page_result.get('title', '')
            
            if len(content) < 200:  # 短すぎるコンテンツは除外
                return None
            
            # 興味度評価
            interest_score, interesting_topics = await self.interest_profiler.evaluate_content_interest(
                content, {'title': title, 'url': url}
            )
            
            # コンテンツタイプの推定
            content_type = await self._classify_content_type(content, title)
            
            # 要約の生成
            summary = await self._generate_content_summary(content)
            
            # キー概念の抽出
            key_concepts = await self._extract_key_concepts(content)
            
            web_content = WebContent(
                url=url,
                title=title,
                content=content[:2000],  # 最初の2000文字
                content_type=content_type,
                discovery_timestamp=time.time(),
                interest_score=interest_score,
                learning_value=0.7,  # 簡略化
                summary=summary,
                key_concepts=key_concepts,
                related_topics=interesting_topics,
                source_credibility=0.8  # 簡略化
            )
            
            return web_content
            
        except Exception as e:
            logger.error(f"コンテンツ分析中にエラー ({url}): {e}")
            return None
    
    async def _classify_content_type(self, content: str, title: str) -> ContentType:
        """コンテンツタイプの分類"""
        content_lower = content.lower()
        title_lower = title.lower()
        
        if "arxiv" in title_lower or "paper" in title_lower or "research" in title_lower:
            return ContentType.RESEARCH_PAPER
        elif "tutorial" in title_lower or "how to" in title_lower:
            return ContentType.TUTORIAL
        elif "news" in title_lower or "breaking" in title_lower:
            return ContentType.NEWS
        elif "documentation" in title_lower or "docs" in title_lower:
            return ContentType.DOCUMENTATION
        elif "blog" in title_lower:
            return ContentType.BLOG_POST
        else:
            return ContentType.ARTICLE
    
    async def _generate_content_summary(self, content: str) -> str:
        """コンテンツ要約の生成"""
        summary_prompt = f"""
        以下のコンテンツを3-4文で要約してください：

        {content[:1000]}

        重要なポイントに焦点を当てた簡潔な要約を提供してください。
        """
        
        response = await self.provider.call(summary_prompt, "")
        return response.get("text", "")[:300]  # 最大300文字
    
    async def _extract_key_concepts(self, content: str) -> List[str]:
        """キー概念の抽出"""
        concepts_prompt = f"""
        以下のコンテンツから重要な概念・技術・用語を5個以下抽出してください：

        {content[:1500]}

        概念を箇条書きで列挙してください。
        """
        
        response = await self.provider.call(concepts_prompt, "")
        concepts_text = response.get("text", "")
        
        concepts = []
        for line in concepts_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•')):
                concept = line.lstrip('-•* ').strip()
                if concept:
                    concepts.append(concept)
        
        return concepts[:5]
    
    async def _learn_from_content(self, content: WebContent) -> Dict[str, Any]:
        """コンテンツからの学習"""
        learning_prompt = f"""
        以下のコンテンツから学習できる重要な知識を抽出してください：

        タイトル: {content.title}
        要約: {content.summary}
        キー概念: {content.key_concepts}

        抽出する知識:
        1. 新しい事実や発見
        2. 重要な原理や理論
        3. 実用的な手法や技術
        4. 将来への示唆

        JSON形式で回答してください。
        """
        
        response = await self.provider.call(learning_prompt, "")
        try:
            learning_data = json.loads(response.get("text", "{}"))
            
            # 学習内容の保存
            knowledge_items = []
            for category, items in learning_data.items():
                if isinstance(items, list):
                    knowledge_items.extend(items)
            
            # 発見されたコンテンツの記録
            self.discovered_content.append(content)
            
            return {
                "new_knowledge": knowledge_items,
                "concepts_learned": content.key_concepts,
                "source": content.url
            }
        
        except json.JSONDecodeError:
            return {"new_knowledge": [], "concepts_learned": content.key_concepts}
    
    async def _evolve_exploration_topics(self, current_topics: List[str], 
                                       session_results: Dict) -> List[str]:
        """探索トピックの進化"""
        # 興味深いコンテンツから新しいトピックを抽出
        new_topics = []
        
        for content in session_results["content_discovered"]:
            new_topics.extend(content.get("key_concepts", []))
        
        # トピックの多様性を保ちながら進化
        evolved_topics = current_topics[:10]  # 既存トピックの一部を保持
        evolved_topics.extend(new_topics[:10])  # 新しいトピックを追加
        
        return list(set(evolved_topics))[:20]  # 重複除去と制限
    
    async def _analyze_session_results(self, session_results: Dict) -> Dict[str, Any]:
        """セッション結果の分析"""
        analysis = {
            "discovery_efficiency": 0.0,
            "learning_quality": 0.0,
            "interest_alignment": 0.0,
            "knowledge_diversity": 0.0,
            "exploration_breadth": 0.0
        }
        
        content_count = len(session_results["content_discovered"])
        knowledge_count = len(session_results["knowledge_gained"])
        
        if content_count > 0:
            analysis["discovery_efficiency"] = min(1.0, content_count / 10)
            analysis["learning_quality"] = min(1.0, knowledge_count / content_count)
            
            # 興味スコアの平均
            avg_interest = sum(c.get("interest_score", 0) for c in session_results["content_discovered"]) / content_count
            analysis["interest_alignment"] = avg_interest
            
            # 概念の多様性
            all_concepts = []
            for content in session_results["content_discovered"]:
                all_concepts.extend(content.get("key_concepts", []))
            
            unique_concepts = len(set(all_concepts))
            analysis["knowledge_diversity"] = min(1.0, unique_concepts / max(len(all_concepts), 1))
        
        return analysis

class ContinuousLearningManager:
    """継続学習マネージャー"""
    
    def __init__(self, provider, web_search_func, web_fetch_func):
        self.crawler = AutonomousWebCrawler(provider, web_search_func, web_fetch_func)
        self.learning_schedule = {}
        self.learning_sessions = deque(maxlen=100)
        
    async def setup_continuous_learning(self, 
                                      learning_intervals: Dict[str, int] = None,
                                      learning_goals: List[str] = None) -> Dict[str, Any]:
        """継続学習の設定"""
        default_intervals = {
            "daily_exploration": 3600,      # 1時間
            "weekly_deep_dive": 7200,       # 2時間
            "monthly_review": 14400         # 4時間
        }
        
        self.learning_schedule = learning_intervals or default_intervals
        
        default_goals = [
            "最新のAI研究動向の把握",
            "新しい技術手法の学習",
            "関連分野の知識拡張",
            "実用的な応用例の発見"
        ]
        
        learning_goals = learning_goals or default_goals
        
        return {
            "continuous_learning_setup": True,
            "learning_schedule": self.learning_schedule,
            "learning_goals": learning_goals,
            "next_session": "manual_trigger_required"
        }
    
    async def execute_scheduled_learning(self, session_type: str) -> Dict[str, Any]:
        """スケジュールされた学習の実行"""
        if session_type not in self.learning_schedule:
            return {"error": f"Unknown session type: {session_type}"}
        
        duration = self.learning_schedule[session_type]
        
        # セッションタイプに応じたトピック設定
        session_topics = await self._get_session_topics(session_type)
        
        # 自律学習の実行
        learning_result = await self.crawler.start_autonomous_learning(
            initial_topics=session_topics,
            session_duration=duration
        )
        
        # セッション記録
        session_record = {
            "session_type": session_type,
            "timestamp": time.time(),
            "duration": duration,
            "result": learning_result
        }
        
        self.learning_sessions.append(session_record)
        
        return {
            "session_completed": True,
            "session_type": session_type,
            "learning_result": learning_result,
            "session_id": session_record.get("session_id")
        }
    
    async def _get_session_topics(self, session_type: str) -> List[str]:
        """セッションタイプに応じたトピック取得"""
        topic_sets = {
            "daily_exploration": [
                "AI news", "machine learning updates", "tech breakthroughs"
            ],
            "weekly_deep_dive": [
                "AI research papers", "cognitive science", "consciousness studies",
                "neural networks", "deep learning"
            ],
            "monthly_review": [
                "AI safety", "future of AI", "ethical AI", "AI governance",
                "technological singularity", "human-AI collaboration"
            ]
        }
        
        return topic_sets.get(session_type, ["artificial intelligence"])
# /llm_api/autonomous_learning/__init__.py
"""
Autonomous Learning Package
自律的学習システムのパッケージ初期化
"""

from .web_crawler import (
    AutonomousWebCrawler,
    ContinuousLearningManager, 
    InterestProfiler,
    WebContent,
    LearningGoal,
    InterestLevel,
    ContentType
)

__all__ = [
    "AutonomousWebCrawler",
    "ContinuousLearningManager",
    "InterestProfiler", 
    "WebContent",
    "LearningGoal",
    "InterestLevel",
    "ContentType"
]

# /cli/autonomous_learning_cli.py
"""
自律学習システム用CLIツール
"""

import asyncio
import argparse
import logging
import json
from typing import List, Dict, Any

from llm_api.providers import get_provider
from llm_api.autonomous_learning import AutonomousWebCrawler, ContinuousLearningManager
from llm_api.utils.helper_functions import format_json_output

logger = logging.getLogger(__name__)

class AutonomousLearningCLI:
    """自律学習システムのCLIインターフェース"""
    
    def __init__(self):
        self.provider = None
        self.crawler = None
        self.learning_manager = None
    
    async def initialize(self, provider_name: str = "ollama"):
        """システムの初期化"""
        try:
            self.provider = get_provider(provider_name, enhanced=True)
            
            # Web検索・取得機能の設定
            # 実際の実装では web_search と web_fetch ツールを使用
            from llm_api import web_search, web_fetch  # 実装時に追加
            
            self.crawler = AutonomousWebCrawler(self.provider, web_search, web_fetch)
            self.learning_manager = ContinuousLearningManager(self.provider, web_search, web_fetch)
            
            logger.info(f"自律学習システム初期化完了 (プロバイダー: {provider_name})")
            return True
            
        except Exception as e:
            logger.error(f"初期化エラー: {e}")
            return False
    
    async def run_single_session(self, args: argparse.Namespace) -> Dict[str, Any]:
        """単発学習セッションの実行"""
        if not self.crawler:
            await self.initialize(args.provider)
        
        # 設定の調整
        self.crawler.min_interest_threshold = args.min_interest
        self.crawler.max_pages_per_session = args.max_pages
        
        logger.info(f"単発学習セッション開始:")
        logger.info(f"  期間: {args.duration}秒")
        logger.info(f"  トピック: {', '.join(args.topics)}")
        logger.info(f"  最小興味度: {args.min_interest}")
        
        result = await self.crawler.start_autonomous_learning(
            initial_topics=args.topics,
            session_duration=args.duration
        )
        
        return result
    
    async def setup_continuous_learning(self, args: argparse.Namespace) -> Dict[str, Any]:
        """継続学習の設定"""
        if not self.learning_manager:
            await self.initialize(args.provider)
        
        learning_intervals = {
            "daily_exploration": args.daily_duration,
            "weekly_deep_dive": args.weekly_duration,
            "monthly_review": args.monthly_duration
        }
        
        setup_result = await self.learning_manager.setup_continuous_learning(
            learning_intervals=learning_intervals,
            learning_goals=args.learning_goals
        )
        
        return setup_result
    
    async def run_scheduled_session(self, args: argparse.Namespace) -> Dict[str, Any]:
        """スケジュールされたセッションの実行"""
        if not self.learning_manager:
            await self.initialize(args.provider)
        
        result = await self.learning_manager.execute_scheduled_learning(args.session_type)
        return result

async def main():
    """メインCLI関数"""
    parser = argparse.ArgumentParser(
        description="CogniQuantum自律Web学習システム",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 共通オプション
    parser.add_argument("--provider", default="ollama", help="LLMプロバイダー")
    parser.add_argument("--json", action="store_true", help="JSON出力")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細ログ")
    
    # サブコマンド
    subparsers = parser.add_subparsers(dest="command", help="使用可能なコマンド")
    
    # 単発学習コマンド
    single_parser = subparsers.add_parser("learn", help="単発学習セッション")
    single_parser.add_argument("--duration", type=int, default=1800, help="学習時間（秒）")
    single_parser.add_argument("--topics", nargs="+", default=["artificial intelligence"], 
                              help="探索トピック")
    single_parser.add_argument("--min-interest", type=float, default=0.6, 
                              help="最小興味度閾値")
    single_parser.add_argument("--max-pages", type=int, default=20, 
                              help="最大探索ページ数")
    
    # 継続学習設定コマンド
    continuous_parser = subparsers.add_parser("setup", help="継続学習設定")
    continuous_parser.add_argument("--daily-duration", type=int, default=1800, 
                                  help="日次学習時間（秒）")
    continuous_parser.add_argument("--weekly-duration", type=int, default=3600, 
                                  help="週次学習時間（秒）") 
    continuous_parser.add_argument("--monthly-duration", type=int, default=7200,
                                  help="月次学習時間（秒）")
    continuous_parser.add_argument("--learning-goals", nargs="+", 
                                  default=["AI研究動向", "技術革新", "応用事例"],
                                  help="学習目標")
    
    # スケジュール実行コマンド
    schedule_parser = subparsers.add_parser("run", help="スケジュール学習実行")
    schedule_parser.add_argument("session_type", 
                                choices=["daily_exploration", "weekly_deep_dive", "monthly_review"],
                                help="セッションタイプ")
    
    args = parser.parse_args()
    
    # ログレベル設定
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # CLIインスタンス作成
    cli = AutonomousLearningCLI()
    
    try:
        if args.command == "learn":
            result = await cli.run_single_session(args)
            
            if args.json:
                print(format_json_output(result))
            else:
                print(f"🎓 学習セッション完了!")
                print(f"  探索ページ数: {result['pages_crawled']}")
                print(f"  発見コンテンツ: {len(result['session_summary']['content_discovered'])}")
                print(f"  学習効率: {result['learning_efficiency']:.2f}")
                
                if result['session_summary']['knowledge_gained']:
                    print(f"\n📚 新たに学習した知識:")
                    for knowledge in result['session_summary']['knowledge_gained'][:5]:
                        print(f"  • {knowledge}")
        
        elif args.command == "setup":
            result = await cli.setup_continuous_learning(args)
            
            if args.json:
                print(format_json_output(result))
            else:
                print(f"⚙️  継続学習設定完了!")
                print(f"  学習スケジュール:")
                for session_type, duration in result['learning_schedule'].items():
                    print(f"    {session_type}: {duration}秒")
                print(f"  学習目標数: {len(result['learning_goals'])}")
        
        elif args.command == "run":
            result = await cli.run_scheduled_session(args)
            
            if args.json:
                print(format_json_output(result))
            else:
                print(f"🚀 スケジュール学習完了!")
                print(f"  セッションタイプ: {result['session_type']}")
                print(f"  学習効率: {result['learning_result']['learning_efficiency']:.2f}")
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n学習セッションが中断されました。")
    except Exception as e:
        logger.error(f"実行エラー: {e}", exc_info=True)
        print(f"❌ エラーが発生しました: {e}")

if __name__ == "__main__":
    asyncio.run(main())

# /fetch_llm_autonomous.py
"""
自律学習専用のエントリーポイント
"""

import asyncio
import sys
import os

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cli.autonomous_learning_cli import main

if __name__ == "__main__":
    asyncio.run(main())

# /examples/autonomous_learning_config.py
"""
自律学習システムの設定例
"""

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class AutonomousLearningConfig:
    """自律学習設定"""
    
    # 基本設定
    provider_name: str = "ollama"
    default_session_duration: int = 1800  # 30分
    min_interest_threshold: float = 0.6
    max_pages_per_session: int = 20
    
    # 探索設定
    exploration_strategies: List[str] = None
    interest_domains: List[str] = None
    learning_goals: List[str] = None
    
    # 継続学習設定
    learning_schedule: Dict[str, int] = None
    auto_scheduling: bool = False
    
    # 高度設定
    enable_value_integration: bool = True
    enable_cogniquantum_integration: bool = True
    enable_knowledge_persistence: bool = True
    
    def __post_init__(self):
        if self.exploration_strategies is None:
            self.exploration_strategies = [
                "follow_interesting_links",
                "search_related_topics",
                "explore_authoritative_sources",
                "discover_trending_topics"
            ]
        
        if self.interest_domains is None:
            self.interest_domains = [
                "artificial intelligence",
                "machine learning", 
                "cognitive science",
                "consciousness studies",
                "AI safety",
                "neural networks",
                "future technology"
            ]
        
        if self.learning_goals is None:
            self.learning_goals = [
                "最新AI研究動向の把握",
                "新技術手法の学習",
                "関連分野知識の拡張",
                "実用的応用例の発見",
                "理論的基盤の深化"
            ]
        
        if self.learning_schedule is None:
            self.learning_schedule = {
                "daily_exploration": 1800,      # 30分
                "weekly_deep_dive": 3600,       # 1時間
                "monthly_review": 7200          # 2時間
            }

# 使用例
def create_default_config() -> AutonomousLearningConfig:
    """デフォルト設定の作成"""
    return AutonomousLearningConfig()

def create_intensive_config() -> AutonomousLearningConfig:
    """集中学習設定の作成"""
    return AutonomousLearningConfig(
        default_session_duration=3600,  # 1時間
        min_interest_threshold=0.7,
        max_pages_per_session=50,
        learning_schedule={
            "daily_exploration": 3600,      # 1時間
            "weekly_deep_dive": 7200,       # 2時間 
            "monthly_review": 14400         # 4時間
        }
    )

def create_focused_config(domain: str) -> AutonomousLearningConfig:
    """特定領域に焦点を当てた設定"""
    domain_configs = {
        "AI_research": {
            "interest_domains": [
                "artificial intelligence research",
                "machine learning papers",
                "neural architecture",
                "AI theory"
            ],
            "learning_goals": [
                "最新論文の理解",
                "研究手法の学習",
                "理論的進歩の追跡"
            ]
        },
        "consciousness_studies": {
            "interest_domains": [
                "consciousness research",
                "cognitive science",
                "philosophy of mind",
                "neuroscience"
            ],
            "learning_goals": [
                "意識研究の最新動向",
                "哲学的考察の深化",
                "科学的発見の理解"
            ]
        }
    }
    
    config_data = domain_configs.get(domain, {})
    
    return AutonomousLearningConfig(
        interest_domains=config_data.get("interest_domains"),
        learning_goals=config_data.get("learning_goals"),
        min_interest_threshold=0.8  # より高い閾値
    )
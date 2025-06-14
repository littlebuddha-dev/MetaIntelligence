# /examples/autonomous_learning_demo.py
"""
自律Web学習システムの使用例とデモンストレーション
"""

import asyncio
import logging
from llm_api.providers import get_provider
from llm_api.autonomous_learning.web_crawler import (
    AutonomousWebCrawler, 
    ContinuousLearningManager,
    InterestProfiler
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_autonomous_learning():
    """自律学習システムのデモ"""
    
    # 1. プロバイダーの初期化
    provider = get_provider("ollama", enhanced=True)
    
    # 2. Web検索・取得機能の準備（実際にはweb_searchとweb_fetch関数が必要）
    async def mock_web_search(query):
        return {
            'results': [
                {'link': 'https://example.com/ai-research-1', 'title': 'Latest AI Research'},
                {'link': 'https://example.com/ml-breakthrough', 'title': 'ML Breakthrough'},
                {'link': 'https://example.com/cognitive-science', 'title': 'Cognitive Science'}
            ]
        }
    
    async def mock_web_fetch(url):
        return {
            'content': f"This is sample content from {url}. It contains information about artificial intelligence, machine learning, and cognitive processes.",
            'title': f"Sample Title for {url}"
        }
    
    # 3. 自律学習システムの初期化
    crawler = AutonomousWebCrawler(provider, mock_web_search, mock_web_fetch)
    
    # 4. 単発の自律学習セッション
    logger.info("=== 単発自律学習セッションのデモ ===")
    
    initial_topics = [
        "artificial intelligence", 
        "consciousness studies", 
        "neural networks",
        "AI safety"
    ]
    
    session_result = await crawler.start_autonomous_learning(
        initial_topics=initial_topics,
        session_duration=300  # 5分間のデモ
    )
    
    print(f"学習セッション完了:")
    print(f"- 探索ページ数: {session_result['pages_crawled']}")
    print(f"- 発見したコンテンツ: {len(session_result['session_summary']['content_discovered'])}")
    print(f"- 獲得した知識: {len(session_result['session_summary']['knowledge_gained'])}")
    print(f"- 学習効率: {session_result['learning_efficiency']:.2f}")
    
    # 5. 継続学習マネージャーのデモ
    logger.info("\n=== 継続学習システムのデモ ===")
    
    learning_manager = ContinuousLearningManager(provider, mock_web_search, mock_web_fetch)
    
    # 継続学習の設定
    setup_result = await learning_manager.setup_continuous_learning(
        learning_intervals={
            "daily_exploration": 1800,    # 30分
            "weekly_deep_dive": 3600,     # 1時間
            "monthly_review": 7200        # 2時間
        },
        learning_goals=[
            "最新のAI研究動向の把握",
            "認知科学の新発見の学習", 
            "AI安全性に関する議論の追跡",
            "実用的なAI技術の発見"
        ]
    )
    
    print(f"継続学習設定完了:")
    print(f"- スケジュール: {setup_result['learning_schedule']}")
    print(f"- 学習目標: {len(setup_result['learning_goals'])}個")
    
    # 日次探索セッションの実行
    daily_session = await learning_manager.execute_scheduled_learning("daily_exploration")
    
    print(f"\n日次探索セッション完了:")
    print(f"- セッションタイプ: {daily_session['session_type']}")
    print(f"- 学習効率: {daily_session['learning_result']['learning_efficiency']:.2f}")

async def demo_interest_profiling():
    """興味プロファイリングのデモ"""
    logger.info("=== 興味プロファイリングのデモ ===")
    
    provider = get_provider("ollama", enhanced=True)
    profiler = InterestProfiler(provider)
    
    # サンプルコンテンツの評価
    sample_contents = [
        {
            'text': "最新の大規模言語モデルにおける創発的能力について研究した結果、モデルサイズが一定の閾値を超えると予期しない能力が現れることが判明した。",
            'metadata': {'title': 'LLMの創発的能力に関する研究'}
        },
        {
            'text': "今日のランチメニューはカレーライスでした。とても美味しかったです。",
            'metadata': {'title': 'ランチの記録'}
        },
        {
            'text': "意識の難しい問題について、統合情報理論の観点から新しいアプローチを提案する。",
            'metadata': {'title': '意識研究の新アプローチ'}
        }
    ]
    
    for i, content_data in enumerate(sample_contents, 1):
        interest_score, topics = await profiler.evaluate_content_interest(
            content_data['text'], 
            content_data['metadata']
        )
        
        print(f"コンテンツ {i}:")
        print(f"  タイトル: {content_data['metadata']['title']}")
        print(f"  興味度: {interest_score:.2f}")
        print(f"  関連トピック: {topics}")
        print()

async def demo_advanced_learning_features():
    """高度な学習機能のデモ"""
    logger.info("=== 高度な学習機能のデモ ===")
    
    provider = get_provider("ollama", enhanced=True)
    
    # Web検索とフェッチ機能（実際の実装例）
    from llm_api.providers import get_provider
    
    async def real_web_search(query):
        """実際のWeb検索機能の例"""
        try:
            # 実際にはweb_searchツールを使用
            # ここではモックデータを返す
            return {
                'results': [
                    {
                        'link': f'https://arxiv.org/abs/2301.{i:05d}',
                        'title': f'Research Paper on {query} - {i}',
                        'snippet': f'Abstract about {query} research findings...'
                    }
                    for i in range(1, 6)
                ]
            }
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {'results': []}
    
    async def real_web_fetch(url):
        """実際のWebページ取得機能の例"""
        try:
            # 実際にはweb_fetchツールを使用
            # ここではモックコンテンツを返す
            return {
                'content': f"""
                This is a detailed research paper from {url}.
                
                Abstract: This paper presents novel findings in artificial intelligence
                and machine learning. We demonstrate new approaches to neural network
                architectures that show improved performance on various tasks.
                
                Introduction: Recent advances in deep learning have shown remarkable
                progress in various domains. However, several challenges remain...
                
                Methodology: We propose a new neural architecture that combines
                attention mechanisms with novel regularization techniques...
                
                Results: Our experiments show 15% improvement in accuracy compared
                to baseline methods. The proposed approach demonstrates strong
                generalization capabilities...
                
                Conclusion: This work opens new directions for future research
                in artificial intelligence and cognitive modeling.
                """,
                'title': f'Advanced AI Research - {url.split("/")[-1]}'
            }
        except Exception as e:
            logger.error(f"Web fetch error: {e}")
            return {'content': '', 'title': ''}
    
    # 高度な自律学習システムの初期化
    advanced_crawler = AutonomousWebCrawler(provider, real_web_search, real_web_fetch)
    
    # 学習目標の設定
    learning_goals = [
        "最新のTransformerアーキテクチャの理解",
        "マルチモーダルAIの進展",
        "AI安全性研究の動向",
        "神経科学とAIの融合",
        "量子コンピューティングとAI"
    ]
    
    print("学習目標:")
    for i, goal in enumerate(learning_goals, 1):
        print(f"  {i}. {goal}")
    
    # 高度な学習セッションの実行
    advanced_session = await advanced_crawler.start_autonomous_learning(
        initial_topics=learning_goals,
        session_duration=600  # 10分間
    )
    
    print(f"\n高度学習セッション結果:")
    print(f"- 総探索時間: {advanced_session['duration']:.1f}秒")
    print(f"- 発見コンテンツ数: {len(advanced_session['session_summary']['content_discovered'])}")
    print(f"- 新規知識項目: {len(advanced_session['session_summary']['knowledge_gained'])}")
    print(f"- 平均興味度: {_calculate_average_interest(advanced_session['session_summary']):.2f}")
    
    # 学習した知識の表示
    if advanced_session['session_summary']['knowledge_gained']:
        print(f"\n学習した主要知識:")
        for knowledge in advanced_session['session_summary']['knowledge_gained'][:5]:
            print(f"  - {knowledge}")
    
    # 発見した興味深いトピック
    if advanced_session['session_summary']['new_interests']:
        print(f"\n新たに発見した興味深いトピック:")
        for topic in advanced_session['session_summary']['new_interests'][:5]:
            print(f"  - {topic}")

def _calculate_average_interest(session_summary):
    """平均興味度の計算"""
    content_discovered = session_summary.get('content_discovered', [])
    if not content_discovered:
        return 0.0
    
    total_interest = sum(content.get('interest_score', 0.0) for content in content_discovered)
    return total_interest / len(content_discovered)

async def demo_learning_integration():
    """学習システムとCogniQuantumの統合デモ"""
    logger.info("=== CogniQuantum統合学習のデモ ===")
    
    provider = get_provider("ollama", enhanced=True)
    
    # CogniQuantumシステムとの統合例
    from llm_api.cogniquantum.system import CogniQuantumSystemV2
    from llm_api.value_evolution.evolution_engine import ValueEvolutionEngine
    
    # 統合システムの初期化
    cq_system = CogniQuantumSystemV2(provider, {})
    value_system = ValueEvolutionEngine(provider)
    
    # 価値システムの初期化
    await value_system.initialize_core_values()
    
    # 学習体験のシミュレーション
    learning_experience = {
        "context": {
            "type": "autonomous_web_learning",
            "session_duration": 600,
            "topics_explored": ["AI consciousness", "neural networks", "cognitive science"],
            "content_quality": "high"
        },
        "actions": [
            "探索的Web検索",
            "興味度評価",
            "コンテンツ分析",
            "知識抽出",
            "概念統合"
        ],
        "outcomes": {
            "knowledge_acquired": 15,
            "concepts_learned": 8,
            "insights_generated": 3,
            "curiosity_satisfied": True
        },
        "satisfaction": 0.85  # 高い学習満足度
    }
    
    # 価値システムに学習経験をフィードバック
    value_learning = await value_system.learn_from_experience(learning_experience)
    
    print("価値システム学習結果:")
    print(f"- 価値調整数: {len(value_learning['value_adjustments'])}")
    print(f"- 学習インパクト: {value_learning['overall_impact']:.3f}")
    print(f"- システム一貫性: {value_learning['value_system_coherence']:.3f}")
    
    # 学習から得られた教訓
    if value_learning['lessons_learned']:
        print(f"\n学習した教訓:")
        for lesson in value_learning['lessons_learned']:
            print(f"  - {lesson}")
    
    # CogniQuantumシステムでの統合的問題解決
    integration_prompt = """
    自律的Web学習システムから以下の知識を獲得しました：
    
    1. 最新のAI研究では意識の計算理論が注目されている
    2. 神経科学とAIの融合により新しい認知モデルが提案されている
    3. マルチモーダルAIが人間類似の学習を実現している
    
    この新しい知識を統合して、次の探索すべき研究領域を提案してください。
    """
    
    integration_result = await cq_system.solve_problem(
        integration_prompt,
        mode="adaptive",
        use_rag=False
    )
    
    print(f"\nCogniQuantum統合分析:")
    print(f"推奨探索領域: {integration_result.get('final_solution', '')[:200]}...")

# CLIツールとしての使用例
async def cli_autonomous_learning():
    """CLIツールとしての自律学習"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CogniQuantum自律学習システム")
    parser.add_argument("--mode", choices=["single", "continuous"], default="single",
                       help="学習モード")
    parser.add_argument("--duration", type=int, default=1800,
                       help="学習セッション時間（秒）")
    parser.add_argument("--topics", nargs="+", default=["artificial intelligence"],
                       help="初期探索トピック")
    parser.add_argument("--min-interest", type=float, default=0.6,
                       help="最小興味度閾値")
    
    # 実際のCLI使用時はargparse.parse_args()を使用
    # ここではデモ用にデフォルト値を使用
    args = argparse.Namespace(
        mode="single",
        duration=300,  # 5分
        topics=["consciousness", "AI safety", "neural networks"],
        min_interest=0.7
    )
    
    provider = get_provider("ollama", enhanced=True)
    
    # モック関数の定義
    async def mock_search(q): 
        return {'results': [{'link': f'https://example.com/{i}', 'title': f'Result {i}'} for i in range(3)]}
    async def mock_fetch(url): 
        return {'content': f'Content from {url}', 'title': f'Title for {url}'}
    
    crawler = AutonomousWebCrawler(provider, mock_search, mock_fetch)
    crawler.min_interest_threshold = args.min_interest
    
    if args.mode == "single":
        print(f"開始: 単発学習セッション ({args.duration}秒)")
        print(f"探索トピック: {', '.join(args.topics)}")
        print(f"最小興味度: {args.min_interest}")
        
        result = await crawler.start_autonomous_learning(
            initial_topics=args.topics,
            session_duration=args.duration
        )
        
        print(f"\n学習完了:")
        print(f"  探索ページ: {result['pages_crawled']}")
        print(f"  興味深いコンテンツ: {len(result['session_summary']['content_discovered'])}")
        print(f"  学習効率: {result['learning_efficiency']:.2f}")
        
    elif args.mode == "continuous":
        manager = ContinuousLearningManager(provider, mock_search, mock_fetch)
        
        print("継続学習モードの設定中...")
        setup = await manager.setup_continuous_learning()
        print(f"設定完了: {setup['continuous_learning_setup']}")
        
        # デモ用に日次セッションを実行
        print("日次探索セッション実行中...")
        session = await manager.execute_scheduled_learning("daily_exploration")
        print(f"セッション完了: {session['session_completed']}")

# メイン実行関数
async def main():
    """メイン実行関数"""
    print("🌟 CogniQuantum自律Web学習システム デモ 🌟\n")
    
    try:
        # 基本的な自律学習デモ
        await demo_autonomous_learning()
        print("\n" + "="*60 + "\n")
        
        # 興味プロファイリングデモ
        await demo_interest_profiling()
        print("\n" + "="*60 + "\n")
        
        # 高度な機能デモ
        await demo_advanced_learning_features()
        print("\n" + "="*60 + "\n")
        
        # システム統合デモ
        await demo_learning_integration()
        print("\n" + "="*60 + "\n")
        
        # CLIツールデモ
        await cli_autonomous_learning()
        
        print("\n🎉 全てのデモが完了しました！")
        
    except Exception as e:
        logger.error(f"デモ実行中にエラー: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
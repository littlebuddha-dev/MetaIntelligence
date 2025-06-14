# MetaIntelligence プロジェクトファイル構造


```text
MetaIntelligence/
├── .env.example                # 環境変数のテンプレートファイル
├── README.md                     # プロジェクト全体の概要、特徴、セットアップ方法
├── requirements.txt              # Pythonの依存ライブラリリスト
├── fetch_llm_v2.py               # CLI（コマンドラインインターフェース）のメインエントリーポイント
├── quick_test_v2.py              # 環境と基本機能の動作を素早く確認する診断スクリプト
├── test_all_v2_providers.py      # 全てのプロバイダーとV2モードを対象とした包括的なテストスクリプト
│
├── llm_api/                      # AIのコアロジックを格納するメインパッケージ
│   ├── __init__.py               # パッケージの初期化、ロギング設定
│   ├── config.py                 # APIキーやモデル設定など、プロジェクト全体の設定を集中管理
│   │
│   ├── core/                     # (新設) 複数の機能で共有される共通モジュール
│   │   ├── __init__.py           #
│   │   ├── exceptions.py         # (推奨) カスタム例外クラスを定義
│   │   └── types.py              # (推奨) 複数のパッケージで使われる共通のEnumやデータクラスを定義
│   │
│   ├── providers/                # 各LLMプロバイダーとの連携を担当
│   │   ├── __init__.py           # プロバイダーの動的読み込みとファクトリー機能
│   │   ├── base.py               # 全プロバイダーの抽象基底クラス（LLMProvider, EnhancedLLMProvider）を定義
│   │   ├── openai.py             # OpenAI (GPT) APIとの通信を行う標準プロバイダー
│   │   ├── claude.py             # Anthropic (Claude) APIとの通信を行う標準プロバイダー
│   │   ├── gemini.py             # Google (Gemini) APIとの通信を行う標準プロバイダー
│   │   ├── ollama.py             # Ollama（ローカルモデル）との通信を行う標準プロバイダー
│   │   ├── huggingface.py        # HuggingFace Inference APIとの通信を行う標準プロバイダー
│   │   ├── llamacpp.py           # Llama.cppサーバーとの通信を行う標準プロバイダー
│   │   ├── enhanced_openai_v2.py # OpenAIプロバイダーにCogniQuantum V2機能を追加する拡張ラッパー
│   │   └── ...                   # (他の拡張プロバイダーも同様)
│   │
│   ├── cogniquantum/             # 思考の品質と効率を最適化する認知量子推論システム
│   │   ├── __init__.py           # CogniQuantum V2 コアシステムのパッケージ初期化
│   │   ├── system.py             # 問題の複雑性に応じて最適な推論パイプラインへ処理を振り分けるディスパッチャー
│   │   ├── analyzer.py           # プロンプトの複雑性を多言語対応で分析
│   │   ├── engine.py             # 分解・統合など、実際の推論プロセスを実行するエンジン
│   │   ├── learner.py            # 過去の実行結果からプロンプトの複雑性を学習し、推論を最適化
│   │   ├── enums.py              # 複雑性レジーム（LOW, MEDIUM, HIGH）のEnumを定義
│   │   ├── tracker.py            # 推論プロセスのメトリクスを追跡するためのデータクラス
│   │   └── pipelines/            # 各推論モードの具体的な処理フローを実装
│   │       ├── __init__.py       # 全パイプラインの統一インターフェース
│   │       ├── adaptive.py       # 適応型推論パイプラインの処理
│   │       ├── parallel.py       # 並列推論パイプラインの処理
│   │       ├── quantum_inspired.py # 量子インスパイアード推論パイプラインの処理
│   │       └── speculative.py    # 投機的思考パイプラインの処理
│   │
│   ├── master_system/            # 全ての高度な機能を統合し、システム全体を統括する最高レベルのシステム
│   │   ├── __init__.py           # マスターシステムパッケージの初期化
│   │   ├── facade.py             # (推奨) CogniQuantumMasterクラス。外部からの統一インターフェースを提供
│   │   ├── orchestrator.py       # 各サブシステム間の協調動作を調整し、統合的な問題解決を実行
│   │   ├── consciousness.py      # (推奨) システム全体の意識レベルの進化を管理
│   │   ├── problem_solver.py     # (推奨) 究極的問題解決メソッドのロジックを実装
│   │   ├── wisdom.py             # (推奨) 複数の解決策や経験から、普遍的な知恵を生成・蒸留
│   │   └── types.py              # (推奨) マスターシステム固有のデータクラスやEnumを定義
│   │
│   ├── autonomous_learning/      # Webからの自律的な情報収集と学習
│   │   ├── __init__.py           # 自律学習パッケージの初期化
│   │   ├── crawler.py            # Web巡回のコアロジック
│   │   ├── enhanced_web_crawler.py # JavaScriptレンダリング対応の拡張クローラー
│   │   └── ...                   # (推奨) manager.py, renderer.py, profiler.py, types.py に分割
│   │
│   ├── meta_cognition/           # 自身の思考プロセスを分析・改善する自己認識（メタ認知）システム
│   ├── dynamic_architecture/     # タスクに応じて自身の構成を動的に最適化するシステム
│   ├── super_intelligence/       # 複数のAIシステムを統合し、集合知を創発させるシステム
│   ├── value_evolution/          # 経験から倫理観や価値判断基準を学習・進化させるシステム
│   ├── problem_discovery/        # データから人間が気づかない潜在的な問題を発見するシステム
│   ├── rag/                      # RAG（検索拡張生成）機能
│   ├── reasoning/                # 自己発見に基づく推論戦略モジュール
│   ├── tools/                    # 画像検索などの外部ツール
│   └── utils/                    # パフォーマンスモニターなどの補助的ツール
│
├── cli/                          # コマンドラインインターフェース関連のコード
│   ├── __init__.py               # パッケージ初期化
│   ├── main.py                   # argparseを用いた引数の解析と、handlerへの処理委譲
│   └── handler.py                # CLIコマンドの具体的な処理ロジックを実装
│
├── doc/                          # プロジェクトのドキュメント
│   ├── directory_structure.md	# このファイル
│   ├── architecture.md           # システムアーキテクチャ設計書
│   ├── cli_guide.md              # CLIの詳細なコマンドリファレンス
│   ├── installation_guide.md     # インストールとセットアップガイド
│   ├── meta_ai_system_concept.md # 「知的システムの知的システム」の概念設計書
│   ├── roadmap.md                # プロジェクトの進化ロードマップ
│   ├── api_reference.md          # Python APIリファレンス
│   └── usage_examples.md         # CLIとAPIの具体的な使用例集
│
├── examples/                     # APIやCLIの具体的な使用方法を示すサンプルコード
│   ├── master_system_api_usage.py # マスター統合システムのPython API使用例
│   ├── autonomous_learning_demo.py # 自律学習システムのデモ
│   ├── v2_demo_script.sh         # V2の各モードを試すデモスクリプト
│   └── sample_questions.txt      # 各複雑度に応じたサンプル質問集
│
└── tests/                        # 単体テストおよび結合テストコード
    ├── __init__.py               # パッケージ初期化
    ├── test_providers.py         # プロバイダー機能のテスト
    ├── test_cogniquantum.py      # CogniQuantumシステムのテスト
    └── test_cli.py               # CLI機能のテスト
    

```
# /llm_api/cogniquantum/system.py
# タイトル: CogniQuantum System V2 (Final Refactored Version)
# 役割: パイプライン管理とコーディネーションに特化。全ての処理を各パイプラインに委譲。

import logging
from typing import Any, Dict, Optional

from .enums import ComplexityRegime
from .pipelines import AdaptivePipeline, ParallelPipeline, QuantumInspiredPipeline, SpeculativePipeline
from ..providers.base import LLMProvider

logger = logging.getLogger(__name__)

class CogniQuantumSystemV2:
    """CogniQuantum V2 メインシステム（最終リファクタリング版）"""
    
    def __init__(self, provider: LLMProvider, base_model_kwargs: Dict[str, Any]):
        logger.info("CogniQuantumシステムV2（最終リファクタリング版）を初期化中")
        if not provider:
            raise ValueError("有効なLLMプロバイダーがCogniQuantumSystemV2に必要です。")
        
        self.provider = provider
        self.base_model_kwargs = base_model_kwargs
        
        # パイプライン初期化
        self.adaptive_pipeline = AdaptivePipeline(provider, base_model_kwargs)
        self.parallel_pipeline = ParallelPipeline(provider, base_model_kwargs)
        self.quantum_pipeline = QuantumInspiredPipeline(provider, base_model_kwargs)
        self.speculative_pipeline = SpeculativePipeline(provider, base_model_kwargs)
        
        logger.info("CogniQuantumシステムV2の初期化完了 - 全パイプライン利用可能")
    
    async def solve_problem(
        self,
        prompt: str,
        system_prompt: str = "",
        force_regime: Optional[ComplexityRegime] = None,
        use_rag: bool = False,
        knowledge_base_path: Optional[str] = None,
        use_wikipedia: bool = False,
        real_time_adjustment: bool = True,
        mode: str = 'adaptive'
    ) -> Dict[str, Any]:
        """問題解決のメインエントリーポイント"""
        logger.info(f"問題解決プロセス開始（V2最終版, モード: {mode}）: {prompt[:80]}...")
        
        # モード別のパイプライン選択
        try:
            if mode in ['adaptive', 'efficient', 'balanced', 'decomposed', 'edge', 'paper_optimized']: # paper_optimizedもadaptiveで処理されるように追加
                # 適応型パイプライン
                logger.info("適応型パイプラインを選択")
                return await self.adaptive_pipeline.execute(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    force_regime=force_regime,
                    use_rag=use_rag,
                    knowledge_base_path=knowledge_base_path,
                    use_wikipedia=use_wikipedia,
                    real_time_adjustment=real_time_adjustment,
                    mode=mode
                )
            
            elif mode == 'parallel':
                # 並列パイプライン
                logger.info("並列パイプラインを選択")
                return await self.parallel_pipeline.execute(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    use_rag=use_rag,
                    knowledge_base_path=knowledge_base_path,
                    use_wikipedia=use_wikipedia
                )
            
            elif mode == 'quantum_inspired':
                # 量子インスパイアードパイプライン
                logger.info("量子インスパイアードパイプラインを選択")
                return await self.quantum_pipeline.execute(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    use_rag=use_rag,
                    knowledge_base_path=knowledge_base_path,
                    use_wikipedia=use_wikipedia
                )
            
            elif mode == 'speculative_thought': # speculative_thoughtモードを追加
                # 投機的思考パイプライン
                logger.info("投機的思考パイプラインを選択")
                return await self.speculative_pipeline.execute(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    use_rag=use_rag,
                    knowledge_base_path=knowledge_base_path,
                    use_wikipedia=use_wikipedia
                )
            
            else:
                # 未知のモード - 適応型にフォールバック
                logger.warning(f"未知のモード '{mode}' です。適応型パイプラインにフォールバックします。")
                return await self.adaptive_pipeline.execute(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    force_regime=force_regime,
                    use_rag=use_rag,
                    knowledge_base_path=knowledge_base_path,
                    use_wikipedia=use_wikipedia,
                    real_time_adjustment=real_time_adjustment,
                    mode='adaptive'
                )
                
        except Exception as e:
            logger.error(f"パイプライン実行中にエラー（モード: {mode}）: {e}", exc_info=True)
            return {
                'success': False,
                'final_solution': None,
                'image_url': None,
                'thought_process': {'error': f'パイプライン実行エラー: {str(e)}'},
                'v2_improvements': {'mode': mode, 'error_occurred': True},
                'version': 'v2',
                'error': str(e)
            }
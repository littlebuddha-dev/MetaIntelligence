# /tests/test_cogniquantum.py

import pytest
from unittest.mock import MagicMock, patch, call
import asyncio # asyncioをインポート

# Test target modules
from llm_api.cogniquantum import ( # パスを修正
    CogniQuantumSystemV2,
    ComplexityRegime,
)
from llm_api.cogniquantum.analyzer import AdaptiveComplexityAnalyzer # analyzerを個別にインポート
from llm_api.cogniquantum.engine import EnhancedReasoningEngine # engineを個別にインポート
from llm_api.providers.base import LLMProvider, EnhancedLLMProvider # LLMProviderとEnhancedLLMProviderをインポート

# --- Test Fixtures ---

@pytest.fixture
def mock_standard_provider():
    """A pytest fixture that creates a mock LLMProvider (standard)."""
    provider = MagicMock(spec=LLMProvider)
    # Configure the mock to return a simple response when call is called
    provider.call.side_effect = lambda prompt, system_prompt="", **kwargs: {"text": f"Mocked standard response for: {prompt}", "error": None}
    provider.provider_name = "mock_standard_provider"
    provider.get_capabilities.return_value = { # capabilitiesをモック
        LLMProvider.ProviderCapability.STANDARD_CALL: True
    }
    return provider

@pytest.fixture
def mock_enhanced_provider(mock_standard_provider):
    """A pytest fixture that creates a mock EnhancedLLMProvider."""
    # EnhancedLLMProvider は内部で standard_provider を使用するため、それをラップする
    enhanced_provider = MagicMock(spec=EnhancedLLMProvider)
    enhanced_provider.standard_provider = mock_standard_provider # 内部のstandard_providerを設定
    enhanced_provider.provider_name = mock_standard_provider.provider_name # 名前を同期
    enhanced_provider.call.side_effect = lambda prompt, system_prompt="", **kwargs: {"text": f"Mocked enhanced response for: {prompt}", "error": None}
    enhanced_provider.get_capabilities.return_value = { # capabilitiesをモック
        EnhancedLLMProvider.ProviderCapability.STANDARD_CALL: True,
        EnhancedLLMProvider.ProviderCapability.ENHANCED_CALL: True
    }
    return enhanced_provider


@pytest.fixture
def cq_system(mock_enhanced_provider):
    """A pytest fixture that creates an instance of CogniQuantumSystemV2 with a mock enhanced provider."""
    # CogniQuantumSystemV2 はEnhancedLLMProviderを受け取る
    return CogniQuantumSystemV2(provider=mock_enhanced_provider, base_model_kwargs={})

# --- Test Cases ---

class TestAdaptiveComplexityAnalyzer:
    """Tests for the AdaptiveComplexityAnalyzer class."""

    @pytest.mark.asyncio
    async def test_analyze_complexity(self): # メソッド名を変更し、async に
        """Test the prompt complexity analysis."""
        analyzer = AdaptiveComplexityAnalyzer() # LearnerはここではNoneでOK
        
        # mock internal _keyword_based_analysis for isolation
        with patch.object(analyzer, '_keyword_based_analysis', return_value=10):
            # mock _get_spacy_model to return None or a dummy to force keyword-based
            with patch.object(analyzer, '_get_spacy_model', return_value=None):
                score, regime = analyzer.analyze_complexity("Simple question") # 戻り値の順序がスコア, レジーム
                assert regime == ComplexityRegime.LOW
                assert score == 10

        with patch.object(analyzer, '_keyword_based_analysis', return_value=50):
            with patch.object(analyzer, '_get_spacy_model', return_value=None):
                score, regime = analyzer.analyze_complexity("Medium question that requires some thought process")
                assert regime == ComplexityRegime.MEDIUM
                assert score == 50

        with patch.object(analyzer, '_keyword_based_analysis', return_value=80):
            with patch.object(analyzer, '_get_spacy_model', return_value=None):
                score, regime = analyzer.analyze_complexity("Very long and complex question asking multiple things with various constraints and conditions that need careful consideration and step-by-step problem-solving approach to ensure accuracy and completeness in the final output.")
                assert regime == ComplexityRegime.HIGH
                assert score == 80

    @pytest.mark.asyncio
    async def test_nlp_enhanced_analysis_fallback(self):
        """Test that NLP analysis falls back to keyword-based if spacy fails."""
        analyzer = AdaptiveComplexityAnalyzer()
        
        # spacy.load がエラーを発生させるようにモック
        with patch('spacy.load', side_effect=OSError("Spacy model not found")):
            with patch.object(analyzer, '_keyword_based_analysis', return_value=50) as mock_keyword:
                score, regime = analyzer.analyze_complexity("This is a moderately complex prompt for NLP analysis.")
                # NLPロードが失敗し、キーワードベースにフォールバックすることを確認
                mock_keyword.assert_called_once()
                assert regime == ComplexityRegime.MEDIUM
                assert score == 50


class TestCogniQuantumSystemV2:
    """Tests for the main CogniQuantumSystemV2 class and its dispatch logic."""

    @pytest.mark.asyncio
    async def test_solve_problem_mode_dispatch(self, cq_system): # メソッド名を変更し、async に
        """Test that solve_problem correctly dispatches to the right pipeline based on the mode."""
        prompt = "test prompt"
        
        # AdaptivePipeline の execute をモック
        with patch.object(cq_system.adaptive_pipeline, 'execute') as mock_adaptive:
            mock_adaptive.return_value = {"success": True, "final_solution": "Adaptive solution", "v2_improvements": {"regime": "adaptive"}}
            await cq_system.solve_problem(prompt, mode='adaptive')
            mock_adaptive.assert_called_once_with(prompt=prompt, system_prompt="", force_regime=None, use_rag=False, knowledge_base_path=None, use_wikipedia=False, real_time_adjustment=True, mode='adaptive')

        # ParallelPipeline の execute をモック
        with patch.object(cq_system.parallel_pipeline, 'execute') as mock_parallel:
            mock_parallel.return_value = {"success": True, "final_solution": "Parallel solution", "v2_improvements": {"parallel_execution": True}}
            await cq_system.solve_problem(prompt, mode='parallel')
            mock_parallel.assert_called_once_with(prompt=prompt, system_prompt="", use_rag=False, knowledge_base_path=None, use_wikipedia=False)
            
        # QuantumInspiredPipeline の execute をモック
        with patch.object(cq_system.quantum_pipeline, 'execute') as mock_quantum:
            mock_quantum.return_value = {"success": True, "final_solution": "Quantum solution", "v2_improvements": {"quantum_inspired": True}}
            await cq_system.solve_problem(prompt, mode='quantum_inspired')
            mock_quantum.assert_called_once_with(prompt=prompt, system_prompt="", use_rag=False, knowledge_base_path=None, use_wikipedia=False)

        # SpeculativePipeline の execute をモック
        with patch.object(cq_system.speculative_pipeline, 'execute') as mock_speculative:
            mock_speculative.return_value = {"success": True, "final_solution": "Speculative solution", "v2_improvements": {"speculative_execution_enabled": True}}
            await cq_system.solve_problem(prompt, mode='speculative_thought')
            mock_speculative.assert_called_once_with(prompt=prompt, system_prompt="", use_rag=False, knowledge_base_path=None, use_wikipedia=False)


class TestEnhancedReasoningEngine:
    """Tests for the core reasoning logic in EnhancedReasoningEngine."""

    @pytest.mark.asyncio
    async def test_high_complexity_decomposes_and_integrates(self, mock_standard_provider): # fixture名を修正
        """Verify the high complexity flow: decompose -> solve sub-problems -> integrate."""
        engine = EnhancedReasoningEngine(provider=mock_standard_provider, base_model_kwargs={}) # provider引数を修正
        complex_prompt = "Explain quantum computing and its impact on cryptography."

        # Define the mock responses for each step of the high-complexity flow
        # .call メソッドの side_effect に変更
        mock_standard_provider.call.side_effect = [
            # 1. Response for the decomposition prompt (JSON format expected by _decompose_complex_problem)
            {"text": '{"sub_problems": ["Explain quantum bits (qubits).", "Explain Shor\'s algorithm."]}', "error": None},
            # 2. Response for solving sub-problem 1
            {"text": "Qubits can exist in a superposition of 0 and 1.", "error": None},
            # 3. Response for solving sub-problem 2
            {"text": "Shor's algorithm can factor large numbers, breaking RSA encryption.", "error": None},
            # 4. Response for the sequential integration prompts (first integration)
            {"text": "Integrated: Qubits can exist in a superposition of 0 and 1. Shor's algorithm can factor large numbers, breaking RSA encryption.", "error": None},
            # 5. Response for the final polish prompt
            {"text": "Final integrated answer about quantum computing and cryptography polished.", "error": None}
        ]
        
        result_dict = await engine._execute_high_complexity_reasoning(complex_prompt, system_prompt="") # await と system_promptを追加

        # Assert the final result is the integrated and polished one
        assert result_dict['solution'] == "Final integrated answer about quantum computing and cryptography polished."
        assert not result_dict['error']
        assert result_dict['complexity_regime'] == ComplexityRegime.HIGH.value
        assert "decomposition" in result_dict
        assert "sub_solutions" in result_dict
        
        # Assert that the provider was called multiple times
        # 1 (decompose) + 2 (solve sub-problems) + 1 (integrate step) + 1 (final polish) = 5 calls
        assert mock_standard_provider.call.call_count == 5

        # Optionally, check the prompts sent to the mock provider
        actual_calls = mock_standard_provider.call_args_list
        assert "以下の複雑な問題を、解決可能な独立したサブ問題に分解してください。" in actual_calls[0].args[0]
        assert "解決すべきサブ問題\nExplain quantum bits (qubits)." in actual_calls[1].args[0]
        assert "解決すべきサブ問題\nExplain Shor's algorithm." in actual_calls[2].args[0]
        assert "これまでの統合結果" in actual_calls[3].args[0] # 逐次統合
        assert "統合された文章" in actual_calls[4].args[0] # 最終仕上げ
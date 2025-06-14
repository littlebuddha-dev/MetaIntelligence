# /tests/test_cli.py

import pytest
from unittest.mock import patch, MagicMock
import sys
import asyncio # asyncioをインポート

# main関数をcli.mainからインポートするように修正
from cli.main import main as cli_main

# 全てのテストでasyncio.run()をラップするためにpytest-asyncioを使用
# pytestのfixtureでイベントループを管理するため、テスト関数はasyncキーワードで定義する
@pytest.mark.asyncio
async def test_cli_simple_prompt():
    """Test a simple CLI call with a provider and prompt."""
    with patch('cli.handler.get_provider') as mock_get_provider:
        mock_provider_instance = MagicMock()
        mock_provider_instance.call.return_value = {"text": "CLI test successful", "error": None}
        mock_get_provider.return_value = mock_provider_instance
        
        test_args = ["fetch_llm_v2.py", "openai", "Hello, world!"] # 引数の渡し方を修正
        with patch.object(sys, 'argv', test_args):
            await cli_main() # await を追加

        # Assertions
        # get_providerの呼び出しがenhanced=Trueで行われることを確認
        mock_get_provider.assert_called_with("openai", enhanced=True)
        # provider.call が正しく呼び出されることを確認
        mock_provider_instance.call.assert_called_once_with("Hello, world!", mode='adaptive', json=False, system_prompt=None, temperature=None, max_tokens=None, use_rag=False, knowledge_base_path=None, use_wikipedia=False, real_time_adjustment=True, force_v2=False, no_fallback=False)


@pytest.mark.asyncio
async def test_cli_v2_mode_argument():
    """Test that the --mode argument is passed correctly to the provider."""
    with patch('cli.handler.get_provider') as mock_get_provider:
        mock_provider_instance = MagicMock()
        mock_provider_instance.call.return_value = {"text": "Efficient mode test", "error": None}
        mock_get_provider.return_value = mock_provider_instance
        
        test_args = ["fetch_llm_v2.py", "openai", "Be quick", "--mode", "efficient"] # 引数の渡し方を修正
        with patch.object(sys, 'argv', test_args):
            await cli_main() # await を追加

        mock_get_provider.assert_called_once_with("openai", enhanced=True)
        # provider.call が正しいmodeで呼び出されることを確認
        mock_provider_instance.call.assert_called_once_with("Be quick", mode='efficient', json=False, system_prompt=None, temperature=None, max_tokens=None, use_rag=False, knowledge_base_path=None, use_wikipedia=False, real_time_adjustment=True, force_v2=False, no_fallback=False)


@pytest.mark.asyncio
@patch('cli.handler.CogniQuantumCLIV2Fixed.check_system_health')
@patch('cli.main.get_provider') # cli.mainからget_providerをパッチ
async def test_cli_health_check(mock_get_provider, mock_health_check): # 引数の順序を修正
    """Test that the --health-check argument calls the correct function and exits."""
    mock_health_check.return_value = {} # ダミーの戻り値
    test_args = ["fetch_llm_v2.py", "ollama", "--health-check"] # プロバイダーも指定が必要
    with patch.object(sys, 'argv', test_args):
        # SystemExitをキャッチ
        with pytest.raises(SystemExit):
            await cli_main()
    
    mock_health_check.assert_called_once_with("ollama") # プロバイダー名が渡されることを確認
    mock_get_provider.assert_not_called() # main関数内の get_provider は呼ばれないことを確認


@pytest.mark.asyncio
async def test_cli_missing_prompt():
    """Test that the CLI exits if --prompt is missing for a standard request."""
    test_args = ["fetch_llm_v2.py", "openai"] # プロバイダーは必要
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit): # SystemExitをキャッチ
            await cli_main()


@pytest.mark.asyncio
@patch('cli.handler.get_provider') # cli.handler.get_provider をパッチ
async def test_cli_fallback_mechanism(mock_get_provider):
    """Test the provider fallback logic from V2 enhanced -> standard."""
    mock_v2_enhanced_provider_instance = MagicMock()
    mock_standard_provider_instance = MagicMock()

    # Enhanced v2 provider will return an error
    mock_v2_enhanced_provider_instance.call.return_value = {"text": "", "error": "V2 enhanced failed"}
    # Standard provider will succeed
    mock_standard_provider_instance.call.return_value = {"text": "Fallback successful", "error": None}

    # get_provider の side_effect を設定して、2回の呼び出しをシミュレート
    # 1回目は enhanced=True で呼ばれ、2回目は enhanced=False で呼ばれる
    mock_get_provider.side_effect = [
        mock_v2_enhanced_provider_instance, # 1回目 (enhanced=True)
        mock_standard_provider_instance     # 2回目 (enhanced=False)
    ]
    
    test_args = ["fetch_llm_v2.py", "openai", "fallback test"]
    with patch.object(sys, 'argv', test_args):
        await cli_main()
        
    assert mock_get_provider.call_count == 2 # 2回呼ばれることを確認
    calls = mock_get_provider.call_args_list
    # 1回目の呼び出しは enhanced=True であることを確認
    assert calls[0].args == ("openai",) and calls[0].kwargs.get('enhanced') == True
    # 2回目の呼び出しは enhanced=False であることを確認
    assert calls[1].args == ("openai",) and calls[1].kwargs.get('enhanced') == False
    
    # enhanced_call はエラーを返すが、standard_callは成功するため、standard_callが呼ばれる
    mock_v2_enhanced_provider_instance.call.assert_called_once()
    mock_standard_provider_instance.call.assert_called_once()
# /llm_api/providers/ollama.py
# タイトル: OllamaProvider with Centralized Settings and Functional Retry Logic
# 役割: Ollamaと対話するための標準プロバイダー。設定値をconfigモジュールから正しく取得し、堅牢なリトライ機能を提供する。

import logging
import asyncio
from typing import Any, Dict

import httpx
from .base import LLMProvider, ProviderCapability
from ..config import settings

logger = logging.getLogger(__name__)

class OllamaProvider(LLMProvider):
    """
    Ollamaと対話するための標準プロバイダー
    """
    def __init__(self):
        self.api_base_url = settings.OLLAMA_API_BASE_URL
        self.default_model = settings.OLLAMA_DEFAULT_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT
        super().__init__()
        logger.info(f"Ollama provider initialized with API URL: {self.api_base_url} and default model: {self.default_model}")

    def get_capabilities(self) -> Dict[ProviderCapability, bool]:
        """このプロバイダーのケイパビリティを返す。"""
        return {
            ProviderCapability.STANDARD_CALL: True,
            ProviderCapability.ENHANCED_CALL: False,
            ProviderCapability.STREAMING: False,
            ProviderCapability.SYSTEM_PROMPT: True,
            ProviderCapability.TOOLS: False,
            ProviderCapability.JSON_MODE: True,
        }

    # ★★★ 修正箇所: `call` メソッドをオーバーライドしてリトライ機能を追加 ★★★
    async def call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """
        Ollama APIを呼び出す。指数関数的バックオフによるリトライロジックを実装。
        """
        last_exception = None
        model = kwargs.get("model", self.default_model)

        for attempt in range(settings.OLLAMA_MAX_RETRIES):
            try:
                # 例外を送出する可能性のある standard_call を直接呼び出す
                return await self.standard_call(prompt, system_prompt, **kwargs)
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_exception = e
                # 5xx系のサーバーエラーまたは接続エラーの場合にリトライ
                should_retry = False
                if isinstance(e, httpx.HTTPStatusError) and 500 <= e.response.status_code < 600:
                    should_retry = True
                elif isinstance(e, httpx.RequestError):
                    should_retry = True

                if should_retry and attempt < settings.OLLAMA_MAX_RETRIES - 1:
                    wait_time = settings.OLLAMA_BACKOFF_FACTOR ** attempt
                    logger.warning(
                        f"Ollama API call failed: {e}. Retrying in {wait_time:.2f}s... "
                        f"(Attempt {attempt + 1}/{settings.OLLAMA_MAX_RETRIES})"
                    )
                    await asyncio.sleep(wait_time)
                    continue  # 次の試行へ
                else:
                    # リトライ対象外のエラーまたはリトライ回数上限
                    error_msg = f"Ollama API request failed after {attempt + 1} attempts: {e}"
                    logger.error(error_msg)
                    return {"text": "", "model": model, "usage": {}, "error": error_msg}
            except Exception as e:
                # その他の予期せぬ例外
                error_msg = f"An unexpected error occurred during Ollama call: {e}"
                logger.error(error_msg, exc_info=True)
                return {"text": "", "model": model, "usage": {}, "error": error_msg}

        # ループを抜けた場合（リトライ上限に達した）
        final_error_msg = f"Failed to get a response from Ollama after {settings.OLLAMA_MAX_RETRIES} attempts. Last exception: {last_exception}"
        logger.error(final_error_msg)
        return {"text": "", "model": model, "usage": {}, "error": final_error_msg}


    async def standard_call(self, prompt: str, system_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """
        Ollama APIを呼び出し、標準化された辞書形式で結果を返す。
        リトライ可能なエラーが発生した場合は、呼び出し元で処理できるよう例外を送出する。
        """
        api_url = f"{self.api_base_url}/api/chat"
        model = kwargs.get("model", self.default_model)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        options = {}
        supported_options = [
            'temperature', 'top_p', 'top_k', 'mirostat', 'mirostat_eta',
            'mirostat_tau', 'num_ctx', 'repeat_last_n', 'repeat_penalty',
            'seed', 'stop', 'tfs_z', 'num_predict'
        ]
        for key in supported_options:
            if key in kwargs:
                options[key] = kwargs[key]
        if options:
            payload['options'] = options
        
        if kwargs.get('json_mode'):
            payload['format'] = 'json'

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(api_url, json=payload)
                response.raise_for_status() # 2xx以外のステータスコードで例外を送出
                response_data = response.json()

            full_response = response_data.get('message', {}).get('content', '')
            
            prompt_tokens = response_data.get("prompt_eval_count") or 0
            completion_tokens = response_data.get("eval_count") or 0

            return {
                "text": full_response,
                "model": model,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                "error": None
            }
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            # リトライ可能なエラーはそのまま呼び出し元（callメソッド）に送出する
            logger.warning(f"Ollama API call failed, propagating exception: {e}")
            raise
        except Exception as e:
            # 予期せぬエラーは捕捉し、リトライ不能なエラーとして扱うために例外を再度raiseする
            error_msg = f"Ollama API呼び出し中に予期せぬエラー: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def should_use_enhancement(self, prompt: str, **kwargs) -> bool:
        """標準プロバイダーは拡張機能を使用しない。"""
        return False

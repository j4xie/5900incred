"""
Llama 模型客户端
使用 Hugging Face Llama-3.1-8B-Instruct 模型（免费）
移植自 backend-ai-chat 项目
"""
import os
import requests
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class LlamaClient:
    """
    Llama-3.1-8B-Instruct 模型客户端
    通过 Hugging Face Router API 调用
    """

    def __init__(self, hf_token: Optional[str] = None):
        """
        初始化 Llama 客户端

        Args:
            hf_token: Hugging Face API Token (如果为空则从环境变量读取)
        """
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        self.hf_token = hf_token or os.environ.get('HF_TOKEN', '')
        self.model = "meta-llama/Llama-3.1-8B-Instruct:fireworks-ai"

        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN 未配置。请设置环境变量或传入 hf_token 参数。\n"
                "获取 Token: https://huggingface.co/settings/tokens"
            )

    def query(self,
              messages: List[Dict[str, str]],
              max_tokens: int = 500,
              temperature: float = 0.0,
              stream: bool = False) -> str:
        """
        调用 Llama 模型

        Args:
            messages: 消息列表，格式：[{"role": "user", "content": "..."}]
            max_tokens: 最大返回 token 数
            temperature: 随机性 (0=确定性, 1=创造性)
            stream: 是否使用流式返回

        Returns:
            模型返回的文本内容

        Raises:
            requests.exceptions.RequestException: API 调用失败
        """
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "messages": messages,
            "model": self.model,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            if stream:
                # 流式返回
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=60
                )
                response.raise_for_status()

                full_content = ""
                for line in response.iter_lines():
                    if not line or not line.startswith(b"data:"):
                        continue
                    if line.strip() == b"data: [DONE]":
                        break

                    try:
                        chunk = json.loads(line.decode("utf-8").lstrip("data:"))
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        full_content += content
                    except json.JSONDecodeError:
                        continue

                return full_content
            else:
                # 非流式返回
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            raise Exception("Llama API 调用超时（60秒）")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Llama API 调用失败: {str(e)}")

    def simple_query(self, prompt: str, **kwargs) -> str:
        """
        简化的单轮对话接口

        Args:
            prompt: 用户提示词
            **kwargs: 传递给 query() 的其他参数

        Returns:
            模型回复
        """
        messages = [{"role": "user", "content": prompt}]
        return self.query(messages, **kwargs)


# 便捷函数
def query_llama(prompt: str, hf_token: Optional[str] = None, **kwargs) -> str:
    """
    便捷函数：快速调用 Llama 模型

    Args:
        prompt: 用户提示词
        hf_token: Hugging Face Token (可选)
        **kwargs: 其他参数（max_tokens, temperature等）

    Returns:
        模型回复
    """
    client = LlamaClient(hf_token)
    return client.simple_query(prompt, **kwargs)

"""
LLM客户端工厂模块 - 用于创建不同类型的LLM客户端
"""

import requests
import time
from openai import OpenAI
from config import API_KEYS, LLM_CONFIG

class LLMClientFactory:
    """LLM客户端工厂类，用于创建不同类型的LLM客户端"""
    
    @staticmethod
    def create_client(provider=None):
        """
        创建LLM客户端
        
        Args:
            provider: LLM提供商名称，如果为None则使用默认提供商
            
        Returns:
            LLM客户端实例
        """
        # 如果未指定提供商，使用默认提供商
        if provider is None:
            provider = LLM_CONFIG["default_provider"]
        
        # 确保提供商存在
        if provider not in LLM_CONFIG["providers"]:
            raise ValueError(f"未知的LLM提供商: {provider}")
        
        # 获取提供商配置
        provider_config = LLM_CONFIG["providers"][provider]
        api_key = API_KEYS.get(provider)
        
        # 根据客户端类型创建客户端
        client_type = provider_config.get("client_type", "openai")
        
        if client_type == "openai":
            # 创建OpenAI客户端
            return OpenAIClient(provider, api_key, provider_config)
        elif client_type == "requests":
            # 创建Requests客户端
            return RequestsClient(provider, api_key, provider_config)
        else:
            raise ValueError(f"未知的客户端类型: {client_type}")
    
    @staticmethod
    def set_default_provider(provider):
        """
        设置默认LLM提供商
        
        Args:
            provider: LLM提供商名称
        """
        if provider not in LLM_CONFIG["providers"]:
            raise ValueError(f"未知的LLM提供商: {provider}")
        
        # 更新默认提供商
        LLM_CONFIG["default_provider"] = provider
        
        # 更新兼容层
        from config import update_compat_layer
        update_compat_layer()
    
    @staticmethod
    def get_available_providers():
        """
        获取可用的LLM提供商列表
        
        Returns:
            可用的LLM提供商列表
        """
        return list(LLM_CONFIG["providers"].keys())


class LLMClientBase:
    """LLM客户端基类，提供共享的错误处理逻辑"""
    
    def __init__(self, provider, api_key, config):
        """
        初始化LLM客户端基类
        
        Args:
            provider: LLM提供商名称
            api_key: API密钥
            config: 提供商配置
        """
        self.provider = provider
        self.api_key = api_key
        self.config = config
        self.max_retries = 3
        self.retry_delay_factor = 2  # 重试延迟因子
    
    def _handle_api_error(self, func, *args, **kwargs):
        """
        通用API错误处理器
        
        Args:
            func: 要执行的API调用函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            API调用结果或错误消息
        """
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                # 调用API
                return func(*args, **kwargs)
                
            except Exception as e:
                retry_count += 1
                print(f"API调用失败 (尝试 {retry_count}/{self.max_retries}): {str(e)}")
                
                if retry_count >= self.max_retries:
                    print(f"达到最大重试次数，返回错误信息")
                    
                    # 处理API密钥打印（如果需要调试）
                    if hasattr(self, 'api_key') and self.api_key:
                        masked_key = f"{self.api_key[:5]}...{self.api_key[-4:]}"
                        print(f"API密钥: {masked_key}")
                    
                    return f"抱歉，{self.provider.capitalize()} API暂时无法访问，请稍后再试。错误: {str(e)}"
                
                # 等待一段时间后重试，使用指数退避策略
                delay = self.retry_delay_factor * retry_count
                time.sleep(delay)


class OpenAIClient(LLMClientBase):
    """使用OpenAI库的LLM客户端"""
    
    def __init__(self, provider, api_key, config):
        """
        初始化OpenAI客户端
        
        Args:
            provider: LLM提供商名称
            api_key: API密钥
            config: 提供商配置
        """
        super().__init__(provider, api_key, config)
        
        # 创建OpenAI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=config["base_url"],
            timeout=60,  # 增加超时时间到60秒
            max_retries=3
        )
    
    def chat_completion(self, messages, **kwargs):
        """
        调用聊天补全API
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            API响应
        """
        def api_call():
            # 合并默认参数和自定义参数
            params = {
                "model": self.config["default_model"],
                "temperature": self.config["temperature"],
                "max_tokens": self.config["max_tokens"],
                "stream": False
            }
            params.update(kwargs)
            
            # 调用API
            response = self.client.chat.completions.create(
                messages=messages,
                **params
            )
            
            return response.choices[0].message.content
        
        # 使用基类的错误处理器
        return self._handle_api_error(api_call)


class RequestsClient(LLMClientBase):
    """使用requests库的LLM客户端"""
    
    def __init__(self, provider, api_key, config):
        """
        初始化Requests客户端
        
        Args:
            provider: LLM提供商名称
            api_key: API密钥
            config: 提供商配置
        """
        super().__init__(provider, api_key, config)
        self.base_url = config["base_url"]
    
    def chat_completion(self, messages, **kwargs):
        """
        调用聊天补全API
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            API响应
        """
        def api_call():
            # 合并默认参数和自定义参数
            params = {
                "model": self.config["default_model"],
                "messages": messages,
                "temperature": self.config["temperature"],
                "max_tokens": self.config["max_tokens"],
                "stream": False
            }
            
            # 添加特定于提供商的参数
            if self.provider == "siliconflow":
                params.update({
                    "top_p": self.config.get("top_p", 0.7),
                    "top_k": self.config.get("top_k", 50),
                    "frequency_penalty": self.config.get("frequency_penalty", 0.5),
                    "n": 1,
                    "response_format": {"type": "text"},
                    "stop": None
                })
            
            # 添加自定义参数
            for key, value in kwargs.items():
                if key not in ["model", "messages"]:  # 避免覆盖关键参数
                    params[key] = value
            
            # 准备请求头
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建URL
            url = f"{self.base_url}/chat/completions"
            
            # 发送请求，增加超时时间
            response = requests.post(url, json=params, headers=headers, timeout=60)
            response.raise_for_status()  # 抛出HTTP错误
            
            # 解析响应
            result = response.json()
            
            # 提取内容
            if self.provider == "siliconflow":
                return result["choices"][0]["message"]["content"]
            else:
                return result["choices"][0]["message"]["content"]
        
        # 使用基类的错误处理器
        return self._handle_api_error(api_call) 
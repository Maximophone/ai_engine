import os
from typing import Optional
from .wrappers import ClaudeWrapper, GeminiWrapper, GPTWrapper, MockWrapper, AIWrapper, DeepSeekWrapper, PerplexityWrapper

_MODELS_DICT = {
    "mock": "mock-",
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-sonnet-20240229",
    "opus": "claude-3-opus-20240229",
    "sonnet3.5": "claude-3-5-sonnet-latest",
    "sonnet3.7": "claude-3-7-sonnet-latest",
    "haiku3.5": "claude-3-5-haiku-latest",
    "gemini1.0": "gemini-1.0-pro-latest",
    "gemini1.5": "gemini-1.5-pro-latest",
    "gemini2.0flash": "gemini-2.0-flash",
    "gemini2.0flashlite": "gemini-2.0-flash-lite",
    "gemini2.0flashthinking": "gemini-2.0-flash-thinking-exp",
    "gemini2.0exp": "gemini-exp-1206",
    "gemini2.5exp": "gemini-2.5-pro-exp-03-25",
    "gemini2.5pro": "gemini-2.5-pro-preview-03-25",
    "gpt3.5": "gpt-3.5-turbo",
    "gpt4": "gpt-4-turbo-preview",
    "gpt4o": "gpt-4o",
    "mini": "gpt-4o-mini",
    "o1-preview": "o1-preview",
    "o1-mini": "o1-mini",
    "o1": "o1-2024-12-17",
    "o3": "o3",
    "o4-mini": "o4-mini",
    "gpt4.1": "gpt-4.1",
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner",
    "sonar": "sonar",
    "sonar-pro": "sonar-pro",
}

DEFAULT_MODEL = "sonnet3.7"

def get_model(model_name: str) -> str:
    return _MODELS_DICT.get(model_name, model_name)

def get_client(
    model_name: str, 
    claude_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openai_org: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    perplexity_api_key: Optional[str] = None,
) -> AIWrapper:
    """
    Get the appropriate AI client wrapper for the specified model.
    Requires the corresponding API key to be provided for the chosen model type.
    
    Args:
        model_name: Name of the model to get client for (e.g., 'haiku', 'gemini1.5', 'gpt4o')
        claude_api_key: API key for Anthropic Claude models
        gemini_api_key: API key for Google Gemini models
        openai_api_key: API key for OpenAI GPT models
        openai_org: Optional OpenAI organization ID
        deepseek_api_key: API key for DeepSeek models
        perplexity_api_key: API key for Perplexity models
        
    Returns:
        AIWrapper instance for the specified model
        
    Raises:
        ValueError: If the required API key for the model type is not provided.
    """
    model_key = model_name # Keep original for lookup if needed
    model_name = get_model(model_name)
    client_name, _ = model_name.split("-", 1) if "-" in model_name else (model_name, "")
    
    if client_name == "claude":
        if not claude_api_key:
            claude_api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not claude_api_key:
                 raise ValueError("Claude API key must be provided via argument or ANTHROPIC_API_KEY env var for Claude models.")
        return ClaudeWrapper(claude_api_key)
    elif client_name == "gemini":
        if not gemini_api_key:
            gemini_api_key = os.environ.get("GOOGLE_API_KEY")
            if not gemini_api_key:
                raise ValueError("Gemini API key must be provided via argument or GOOGLE_API_KEY env var for Gemini models.")
        # Pass model_name (resolved full name) to GeminiWrapper
        # Rate limiting setup moved inside the wrapper or handled by application
        return GeminiWrapper(
            api_key=gemini_api_key, 
            model_name=model_name 
        )
    elif client_name in ["gpt", "o1", "o3", "o4-mini"]:
        if not openai_api_key:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                 raise ValueError("OpenAI API key must be provided via argument or OPENAI_API_KEY env var for GPT/o1/o3 models.")
        # Org ID is optional for GPTWrapper
        openai_org_id = openai_org or os.environ.get("OPENAI_ORG_ID")
        return GPTWrapper(openai_api_key, openai_org_id)
    elif client_name == "deepseek":
        if not deepseek_api_key:
            deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not deepseek_api_key:
                 raise ValueError("DeepSeek API key must be provided via argument or DEEPSEEK_API_KEY env var for DeepSeek models.")
        return DeepSeekWrapper(deepseek_api_key)
    elif client_name == "sonar":
         if not perplexity_api_key:
            perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY")
            if not perplexity_api_key:
                 raise ValueError("Perplexity API key must be provided via argument or PERPLEXITY_API_KEY env var for Perplexity models.")
         return PerplexityWrapper(perplexity_api_key)
    elif client_name == "mock":
        return MockWrapper()
        
    raise ValueError(f"Unsupported model provider for model key: {model_key}") 
import os
from typing import Optional, Tuple
from .wrappers import ClaudeWrapper, GeminiWrapper, GPTWrapper, MockWrapper, AIWrapper, DeepSeekWrapper, PerplexityWrapper

# Explicit set of supported providers
_SUPPORTED_PROVIDERS = {
    "anthropic",
    "google",
    "openai",
    "deepseek",
    "perplexity",
    "mock",
}

# Mapping from user-friendly aliases to a canonical "provider:model_name" format.
# Canonical names themselves are handled directly by the resolution logic.
# Note: Some canonical model names (esp. for 'latest', Google experimental, Perplexity) might need verification against provider docs.
_MODEL_ALIASES = {
    # Anthropic Aliases
    "haiku": "anthropic:claude-3-haiku-20240307",
    "sonnet": "anthropic:claude-3-sonnet-20240229",
    "opus3": "anthropic:claude-3-opus-20240229",
    "sonnet3.5": "anthropic:claude-3-5-sonnet-latest",
    "sonnet3.7": "anthropic:claude-3-7-sonnet-latest", # Verify actual name if available
    "haiku3.5": "anthropic:claude-3-5-haiku-latest", # Verify actual name if available
    "opus4": "anthropic:claude-opus-4-20250514",
    "sonnet4": "anthropic:claude-sonnet-4-20250514",
    "opus4.1": "anthropic:claude-opus-4-1-20250805",
    "sonnet4.5": "anthropic:claude-sonnet-4-5-20250929",

    # Google Aliases
    "gemini1.0": "google:gemini-1.0-pro-latest",
    "gemini1.5": "google:gemini-1.5-pro-latest",
    "gemini2.0flash": "google:gemini-flash", # Assuming simpler name, verify
    "gemini2.0flashlite": "google:gemini-flash-lite", # Verify name
    "gemini2.0flashthinking": "google:gemini-flash-thinking-exp", # Verify name
    "gemini2.0exp": "google:gemini-exp-1206", # Verify name
    "gemini2.5exp": "google:gemini-2.5-pro-exp-03-25", # Verify name
    "gemini2.5pro": "google:gemini-2.5-pro-preview-03-25", # Verify name
    "gemini2.5flash": "google:gemini-2.5-flash", # Verify name
    "gemini2.5flashlite": "google:gemini-2.5-flash-lite", # Verify name
    "gemini3.0pro": "google:gemini-3-pro-preview", # Verify name

    # OpenAI Aliases
    "gpt5.1": "openai:gpt-5.1",
    "pgt5.1instant": "openai:gpt-5.1-chat-latest",
    "gpt5": "openai:gpt-5",
    "gpt5-mini": "openai:gpt-5-mini",
    "gpt5-nano": "openai:gpt-5-nano",
    "gpt3.5": "openai:gpt-3.5-turbo",
    "gpt4": "openai:gpt-4-turbo-preview",
    "gpt4o": "openai:gpt-4o",
    "mini": "openai:gpt-4o-mini",
    "o1-preview": "openai:o1-preview",
    "o1-mini": "openai:o1-mini",
    "o1": "openai:o1-2024-12-17",
    "o3": "openai:o3",
    "o4-mini": "openai:o4-mini",
    "gpt4.1": "openai:gpt-4.1", # Verify name

    # DeepSeek Aliases
    "deepseek-chat": "deepseek:deepseek-chat",
    "deepseek-reasoner": "deepseek:deepseek-reasoner", # Verify name, original had 'reasoner'

    # Perplexity Aliases
    "sonar": "perplexity:sonar", # Verify canonical name
    "sonar-pro": "perplexity:sonar-pro", # Verify canonical name

    # Mock Alias
    "mock": "mock:mock-model",
}

DEFAULT_MODEL_IDENTIFIER = "sonnet4" # Use an alias by default


def resolve_model_info(model_identifier: str) -> Tuple[str, str]:
    """
    Resolves a user-provided model identifier (alias or "provider:model_name")
    into its provider and canonical model name.

    Args:
        model_identifier: The alias (e.g., 'sonnet3.7') or full name (e.g., 'anthropic:claude-3-opus-20240229').

    Returns:
        A tuple containing (provider, model_name).

    Raises:
        ValueError: If the identifier is not recognized, malformed, or uses an unsupported provider.
    """
    canonical_identifier = _MODEL_ALIASES.get(model_identifier)

    if canonical_identifier:
        # Identifier is an alias, use the canonical name from the dict
        pass # canonical_identifier is already set
    elif ":" in model_identifier:
        # Identifier is potentially a direct canonical name "provider:model_name"
        try:
            provider_check, model_name_check = model_identifier.split(":", 1)
            if not provider_check or not model_name_check:
                raise ValueError("Malformed identifier") # Contains colon but empty provider or model
            if provider_check in _SUPPORTED_PROVIDERS:
                canonical_identifier = model_identifier # It's a valid canonical name for a supported provider
            else:
                raise ValueError(f"Unsupported provider '{provider_check}' in identifier: {model_identifier}")
        except ValueError as e:
             # Catch split errors or errors from our checks
             if "Malformed identifier" in str(e) or "Unsupported provider" in str(e):
                 raise
             else: # Assume error from split() due to no colon, which shouldn't happen here, but belts and suspenders
                 raise ValueError(f"Invalid format for model identifier: {model_identifier}") from e
    else:
        # Identifier is not in aliases and doesn't contain a colon, so it's an unknown alias
        raise ValueError(f"Unknown or unsupported model identifier alias: {model_identifier}")

    # Now parse the resolved canonical_identifier
    try:
        provider, model_name = canonical_identifier.split(":", 1)
        # Final check, should always pass if logic above is correct
        if not provider or not model_name or provider not in _SUPPORTED_PROVIDERS:
             raise ValueError("Internal Error: Invalid canonical format or unsupported provider generated")
        return provider, model_name
    except ValueError as e:
         # This catch is primarily for the split(":", 1) failing on a value from _MODEL_ALIASES or internal errors
         raise ValueError(f"Internal Error: Failed to parse canonical identifier '{canonical_identifier}' derived from '{model_identifier}': {e}") from e


def get_wrapper(
    model_identifier: str = DEFAULT_MODEL_IDENTIFIER,
    claude_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openai_org: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    perplexity_api_key: Optional[str] = None,
) -> AIWrapper:
    """
    Get the appropriate AI provider wrapper for the specified model identifier.
    Accepts either a known alias (e.g., 'sonnet3.7') or a full identifier
    in the format "provider:model_name" (e.g., "anthropic:claude-3-opus-20240229").

    Uses the DEFAULT_MODEL_IDENTIFIER if none is provided.

    Requires the corresponding API key to be provided via argument or environment
    variable for the resolved provider.

    Args:
        model_identifier: Alias or full "provider:model_name" string. Defaults to DEFAULT_MODEL_IDENTIFIER.
        claude_api_key: API key for Anthropic models.
        gemini_api_key: API key for Google models.
        openai_api_key: API key for OpenAI models.
        openai_org: Optional OpenAI organization ID.
        deepseek_api_key: API key for DeepSeek models.
        perplexity_api_key: API key for Perplexity models.

    Returns:
        AIWrapper instance for the specified model.

    Raises:
        ValueError: If the identifier is invalid, the provider is unsupported,
                    or the required API key for the provider is missing.
    """
    try:
        provider, model_name = resolve_model_info(model_identifier)
    except ValueError as e:
         # Re-raise with a more context-specific message if desired, or just let it propagate
         raise ValueError(f"Failed to resolve model identifier '{model_identifier}': {e}") from e

    if provider == "anthropic":
        api_key = claude_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key must be provided via argument or ANTHROPIC_API_KEY env var for Anthropic models.")
        # ClaudeWrapper typically doesn't need the specific model name passed during init
        # It usually takes it during the completion call, but check its implementation.
        return ClaudeWrapper(api_key)
    elif provider == "google":
        api_key = gemini_api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key must be provided via argument or GOOGLE_API_KEY env var for Google models.")
        # GeminiWrapper needs the resolved model name
        return GeminiWrapper(api_key=api_key, model_name=model_name)
    elif provider == "openai": # Handles gpt*, o* models
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided via argument or OPENAI_API_KEY env var for OpenAI models.")
        org_id = openai_org or os.environ.get("OPENAI_ORG_ID")
        # GPTWrapper typically doesn't need the model name at init.
        return GPTWrapper(api_key, org_id)
    elif provider == "deepseek":
        api_key = deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key must be provided via argument or DEEPSEEK_API_KEY env var for DeepSeek models.")
        # DeepSeekWrapper likely just needs the key.
        return DeepSeekWrapper(api_key)
    elif provider == "perplexity":
        api_key = perplexity_api_key or os.environ.get("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("Perplexity API key must be provided via argument or PERPLEXITY_API_KEY env var for Perplexity models.")
        # PerplexityWrapper might need the model name depending on its implementation.
        # Assuming it just needs the key for now. If needed: return PerplexityWrapper(api_key, model_name=model_name)
        return PerplexityWrapper(api_key)
    elif provider == "mock":
        return MockWrapper()

    # This should be unreachable if resolve_model_info works correctly and covers all providers in _MODEL_ALIASES
    raise ValueError(f"Internal Error: Unhandled supported provider '{provider}' derived from identifier '{model_identifier}'.") 
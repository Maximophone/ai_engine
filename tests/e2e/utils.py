import os

def e2e_available(provider_env_var: str) -> bool:
    """Checks if E2E tests are enabled and the specific provider key is set."""
    return os.getenv("RUN_E2E") == "1" and os.getenv(provider_env_var)

# Environment variable names used by the library
ANTHROPIC_API_KEY_VAR = "ANTHROPIC_API_KEY"
GOOGLE_API_KEY_VAR = "GOOGLE_API_KEY"
OPENAI_API_KEY_VAR = "OPENAI_API_KEY"
DEEPSEEK_API_KEY_VAR = "DEEPSEEK_API_KEY"
PERPLEXITY_API_KEY_VAR = "PERPLEXITY_API_KEY"

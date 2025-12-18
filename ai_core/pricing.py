pricing_data = { # in $ per 1M tokens
    "opus4": {
        "input": 15,
        "output": 75,
    },
    "opus4.1": {
        "input": 15,
        "output": 75,
    },
    "sonnet4": {
        "input": 3,
        "output": 15,
    },
    "haiku3.5": {
        "input": 0.8,
        "output": 4,
    },
    "opus3": {
        "input": 15,
        "output": 75,
    },
    "sonnet3.7": {
        "input": 3,
        "output": 15,
    },
    "sonnet4.5": {
        "input": 3,
        "output": 15,
    },
    "opus4.5": {
        "input": 5,
        "output": 25,
    },
    "haiku3": {
        "input": 0.25,
        "output": 1.25,
    },
    "gemini2.5pro": { # TODO: For Gemini, the pricing depends on the number of tokens, so I'll have to find a way to put that in there.
        "input": 1.25,
        "output": 10.00,
    },
    "gemini2.5flash": {
        "input": 0.3,
        "output": 2.5,
    },
    "gemini3.0pro": {
        "input": 2.00,
        "output": 12.00,
    },
    "gemini3.0flash": {
        "input": 0.5,
        "output": 3.0,
    },
    "gpt5": {
        "input": 1.25,
        "output": 10.00,
    },
    "gpt5-mini": {
        "input": 0.25,
        "output": 2,
    },
    "gpt5-nano": {
        "input": 0.05,
        "output": 0.4,
    },
}


def compute_request_price(
    tokens_in: int,
    tokens_out: int,
    model_alias: str
) -> float:
    """
    Compute the price of a request based on token usage and model.

    Args:
        tokens_in: Number of input tokens consumed.
        tokens_out: Number of output tokens generated.
        model_alias: The model alias (e.g., 'sonnet4', 'opus4', 'gemini2.5pro').

    Returns:
        The total cost in USD as a float.

    Raises:
        ValueError: If the model alias is not found in the pricing data.

    Example:
        >>> compute_request_price(tokens_in=1000, tokens_out=500, model_alias="sonnet4")
        0.0105
    """
    if model_alias not in pricing_data:
        available_models = ", ".join(sorted(pricing_data.keys()))
        raise ValueError(
            f"Unknown model alias '{model_alias}'. "
            f"Available models: {available_models}"
        )

    model_pricing = pricing_data[model_alias]
    input_price_per_token = model_pricing["input"] / 1_000_000
    output_price_per_token = model_pricing["output"] / 1_000_000

    total_price = (tokens_in * input_price_per_token) + (tokens_out * output_price_per_token)
    return total_price

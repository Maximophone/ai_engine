# AI Core Library

A Python library providing a unified interface for interacting with various Large Language Models (LLMs).

## Features

*   Supports multiple AI providers (OpenAI, Anthropic, Google Gemini, DeepSeek, Perplexity) through a single interface.
*   Handles message history for conversational context.
*   Provides a decorator (`@tool`) for easily defining functions callable by AI models.
*   Includes image input support.
*   Basic token counting utilities.

## Installation

```bash
pip install .
# Or if you upload to PyPI
# pip install ai-core
```

## Basic Usage

```python
from ai_core import AI, Tool, ToolParameter, tool
import os

# Example Tool
@tool(
    description="Get the current weather in a given location",
    city="The city and state, e.g. San Francisco, CA",
    unit="Temperature unit (celsius or fahrenheit)",
    safe=True
)
def get_current_weather(city: str, unit: str = "fahrenheit") -> str:
    """Gets the current weather"""
    # Replace with actual weather API call
    return f"The weather in {city} is 70 degrees {unit}."

# Initialize the client (pass necessary API keys)
# Keys can also be read from environment variables:
# ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, PERPLEXITY_API_KEY
# If an API key argument (e.g., `claude_api_key`) is not provided during initialization, 
# the library will automatically attempt to load it from its corresponding environment variable.
# Explicitly passed keys take precedence over environment variables.
ai = AI(
    model_name="sonnet3.7", # Or gpt4o, gemini1.5, etc.
    system_prompt="You are a helpful assistant.",
    tools=[get_current_weather], 
    claude_api_key=os.environ.get("ANTHROPIC_API_KEY"), # Example for Claude
    # Provide keys for other models as needed
)

# Send a message
response = ai.message("What is the weather in Boston?")
print(response.content)

# If the model decides to use a tool, it will be in response.tool_calls
# The application would then execute the tool and send the result back.
```

## Defining Toolsets

Applications using this library can define their own sets of tools using the `@tool` decorator and pass them to the `AI` client during initialization.

```python
# my_app_tools.py
from ai_core import tool

@tool(
    description="Custom tool specific to my application",
    # ... parameters ...
)
def my_custom_tool(param1: str) -> str:
    # ... implementation ...
    return "Result from custom tool"

# main_app.py
from ai_core import AI
from my_app_tools import my_custom_tool
import os

my_tools = [my_custom_tool]

ai_client = AI(
    model_name="gpt4o",
    tools=my_tools,
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

# Use the client...
response = ai_client.message("Use the custom tool with parameter 'abc'")
print(response)
```

## Contributing

[Instructions for contributing...]

## License

MIT License
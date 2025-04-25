import unittest
import os
from unittest.mock import patch
from ai_core import AI, tool
from tests.e2e.utils import e2e_available, OPENAI_API_KEY_VAR

# Define a simple tool for testing
@tool(description="Get the length of a string", text="The string to measure")
def get_string_length(text: str) -> int:
    return len(text)

# Patch log_token_use globally for this test module
@patch('ai_core.wrappers.base.log_token_use')
class TestOpenAIEndToEnd(unittest.TestCase):

    @unittest.skipUnless(e2e_available(OPENAI_API_KEY_VAR),
                         f"E2E test requires RUN_E2E=1 and {OPENAI_API_KEY_VAR} set")
    def setUp(self):
        # Initialize AI client specifically for OpenAI within the test
        # This ensures it only tries to initialize if the test runs
        self.ai_gpt4o = AI(model_identifier="gpt4o", tools=[get_string_length])
        self.ai_gpt35 = AI(model_identifier="gpt3.5") # Test without tools

    def test_openai_gpt4o_basic_chat(self, mock_log_tokens):
        """Test basic chat completion with GPT-4o"""
        response = self.ai_gpt4o.message("Say hello!")
        self.assertIsInstance(response.content, str)
        self.assertTrue(len(response.content) > 0)
        self.assertIn("hello", response.content.lower())
        self.assertIsNone(response.tool_calls)
        self.assertTrue(mock_log_tokens.called)

    def test_openai_gpt35_basic_chat(self, mock_log_tokens):
        """Test basic chat completion with GPT-3.5 Turbo"""
        response = self.ai_gpt35.message("What is 1 + 1?")
        self.assertIsInstance(response.content, str)
        self.assertTrue(len(response.content) > 0)
        self.assertIn("2", response.content)
        self.assertIsNone(response.tool_calls)
        self.assertTrue(mock_log_tokens.called)

    @unittest.skip("Tool use tests can be flaky and cost more. Enable manually if needed.")
    def test_openai_gpt4o_tool_use(self, mock_log_tokens):
        """Test if GPT-4o can use a simple tool"""
        prompt = "Use the tool to find the length of the word 'supercalifragilisticexpialidocious'"
        response = self.ai_gpt4o.message(prompt)

        # Option 1: Model uses the tool
        if response.tool_calls:
            self.assertEqual(len(response.tool_calls), 1)
            tool_call = response.tool_calls[0]
            self.assertEqual(tool_call.name, "get_string_length")
            self.assertIn("text", tool_call.arguments)
            # Allow for minor variations in how the model extracts the text
            self.assertIn("supercalifragilisticexpialidocious", tool_call.arguments['text'].lower())
            # We would normally execute the tool and send back the result here
        # Option 2: Model answers directly (less likely for GPT-4o with tools)
        else:
            self.assertIsInstance(response.content, str)
            self.assertIn("34", response.content) # Check if it calculated the length itself
            print("\nWarning: GPT-4o answered directly instead of using the tool.")

        self.assertTrue(mock_log_tokens.called)

    # Add more tests: conversation history, image inputs (for gpt4o), error handling etc.

if __name__ == '__main__':
    unittest.main() 
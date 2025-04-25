import unittest
import os
import tempfile
import base64
from PIL import Image
from unittest.mock import patch
from ai_core import AI, tool, Message, MessageContent
from tests.e2e.utils import e2e_available, ANTHROPIC_API_KEY_VAR

# Define a simple tool for testing (can be reused)
@tool(description="Get the length of a string", text="The string to measure")
def get_string_length(text: str) -> int:
    return len(text)

# Helper to create a dummy image file
def create_dummy_image(path, size=(10, 10), format="PNG"):
    img = Image.new('RGB', size, color = 'red')
    img.save(path, format)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Patch log_token_use globally for this test module
@patch('ai_core.wrappers.base.log_token_use')
class TestClaudeEndToEnd(unittest.TestCase):

    test_image_path = None

    @classmethod
    def setUpClass(cls):
        # Create a dummy image file once for all tests in this class
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_image_path = os.path.join(cls.temp_dir.name, "test_claude.png")
        create_dummy_image(cls.test_image_path)

    @classmethod
    def tearDownClass(cls):
        # Clean up the temporary directory
        cls.temp_dir.cleanup()

    @unittest.skipUnless(e2e_available(ANTHROPIC_API_KEY_VAR),
                         f"E2E test requires RUN_E2E=1 and {ANTHROPIC_API_KEY_VAR} set")
    def setUp(self):
        # Initialize AI clients for different Claude models
        self.ai_haiku = AI(model_identifier="haiku", tools=[get_string_length])
        # Sonnet 3.5 supports images and thinking
        self.ai_sonnet37 = AI(model_identifier="sonnet3.7", tools=[get_string_length])

    def test_claude_haiku_basic_chat(self, mock_log_tokens):
        """Test basic chat completion with Claude Haiku"""
        response = self.ai_haiku.message("Please write back the words SCATTERING and ATMOSPHERE in all lowercase. Super simple task (this is just a test). Don't call any tools.")
        self.assertIsInstance(response.content, str)
        self.assertTrue(len(response.content) > 0)

        self.assertTrue("scattering" in response.content.lower() or "atmosphere" in response.content.lower())
        self.assertIsNone(response.tool_calls)
        self.assertIsNone(response.reasoning) # Haiku doesn't support 'thinking'
        self.assertTrue(mock_log_tokens.called)

    @unittest.skip("Tool use tests can be flaky and cost more. Enable manually if needed.")
    def test_claude_sonnet_tool_use(self, mock_log_tokens):
        """Test if Claude Sonnet can use a simple tool"""
        prompt = "Use the tool to find the length of the word 'antidisestablishmentarianism'"
        response = self.ai_sonnet37.message(prompt) # Using Sonnet as it's generally better at tool use

        # Option 1: Model uses the tool
        if response.tool_calls:
            self.assertEqual(len(response.tool_calls), 1)
            tool_call = response.tool_calls[0]
            self.assertEqual(tool_call.name, "get_string_length")
            self.assertIn("text", tool_call.arguments)
            self.assertIn("antidisestablishmentarianism", tool_call.arguments.get('text', '').lower())
        # Option 2: Model answers directly
        else:
            self.assertIsInstance(response.content, str)
            self.assertIn("28", response.content)
            print("\nWarning: Claude answered directly instead of using the tool.")

        self.assertTrue(mock_log_tokens.called)

    def test_claude_sonnet_thinking(self, mock_log_tokens):
        """Test Claude Sonnet's thinking parameter"""
        prompt = "Explain the main steps to bake a simple cake."
        response = self.ai_sonnet37.message(prompt, thinking=True, temperature=1.0) # Temp must be 1.0 for thinking

        self.assertIsInstance(response.content, str)
        self.assertTrue(len(response.content) > 0)
        self.assertIsNotNone(response.reasoning, "Thinking response should have reasoning")
        self.assertTrue(len(response.reasoning) > 10) # Expect some reasoning text
        # Check if response looks like steps
        self.assertTrue("mix" in response.content.lower() or "bake" in response.content.lower() or "ingredients" in response.content.lower())
        self.assertTrue(mock_log_tokens.called)

    def test_claude_sonnet_image_input(self, mock_log_tokens):
        """Test sending an image to Claude Sonnet 3.5"""
        prompt = "Describe this image. Don't use any tools."
        response = self.ai_sonnet37.message(prompt, image_paths=[self.test_image_path])

        self.assertIsInstance(response.content, str)
        self.assertTrue(len(response.content) > 0)
        # Expect some description, maybe mentioning color or shape
        self.assertTrue("red" in response.content.lower() or "square" in response.content.lower() or "image" in response.content.lower())
        self.assertIsNone(response.tool_calls)
        self.assertTrue(mock_log_tokens.called)


if __name__ == '__main__':
    unittest.main() 
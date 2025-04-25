import unittest
import os
import tempfile
import base64
from PIL import Image
from unittest.mock import patch
from ai_core import AI, tool, Message, MessageContent
from tests.e2e.utils import e2e_available, GOOGLE_API_KEY_VAR

# Define a simple tool for testing (can be reused)
@tool(description="Get the length of a string", text="The string to measure")
def get_string_length(text: str) -> int:
    return len(text)

# Helper to create a dummy image file (can be shared or redefined)
def create_dummy_image(path, size=(10, 10), format="PNG"):
    img = Image.new('RGB', size, color = 'blue') # Different color for variety
    img.save(path, format)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Patch log_token_use globally for this test module
@patch('ai_core.wrappers.base.log_token_use')
class TestGeminiEndToEnd(unittest.TestCase):

    test_image_path = None

    @classmethod
    def setUpClass(cls):
        # Create a dummy image file once
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_image_path = os.path.join(cls.temp_dir.name, "test_gemini.png")
        create_dummy_image(cls.test_image_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    @unittest.skipUnless(e2e_available(GOOGLE_API_KEY_VAR),
                         f"E2E test requires RUN_E2E=1 and {GOOGLE_API_KEY_VAR} set")
    def setUp(self):
        # Initialize AI client for Gemini 1.5 Pro
        # Note: get_client requires model name to initialize GeminiWrapper correctly
        self.ai_gemini15 = AI(model_identifier="gemini1.5", tools=[get_string_length])

    def test_gemini15_basic_chat(self, mock_log_tokens):
        """Test basic chat completion with Gemini 1.5 Pro"""
        response = self.ai_gemini15.message("What is the capital of France?")
        self.assertIsInstance(response.content, str)
        self.assertTrue(len(response.content) > 0)
        self.assertIn("Paris", response.content)
        self.assertIsNone(response.tool_calls) # Basic wrapper may not implement tool calls well
        self.assertTrue(mock_log_tokens.called)

    @unittest.skip("Gemini tool use in the current wrapper is basic/not fully implemented. Skipping.")
    def test_gemini15_tool_use(self, mock_log_tokens):
        """Test if Gemini 1.5 Pro can use a simple tool (Expected Fail/Skip)"""
        prompt = "Use the tool to find the length of 'hello world'"
        response = self.ai_gemini15.message(prompt)

        # Gemini's tool support via this wrapper might be limited.
        # This test might fail or the model might just answer directly.
        if response.tool_calls:
             self.assertEqual(len(response.tool_calls), 1)
             # ... assertions based on expected ToolCall structure from Gemini ...
        else:
             self.assertIn("11", response.content)
             print("\nWarning: Gemini likely answered directly (tool use wrapper limitations).")

        self.assertTrue(mock_log_tokens.called)

    def test_gemini15_image_input(self, mock_log_tokens):
        """Test sending an image to Gemini 1.5 Pro"""
        prompt = "Describe this image in one word."
        response = self.ai_gemini15.message(prompt, image_paths=[self.test_image_path])

        self.assertIsInstance(response.content, str)
        self.assertTrue(len(response.content) > 0)
        # Expect short description, maybe color
        self.assertIn("blue", response.content.lower())
        self.assertIsNone(response.tool_calls)
        self.assertTrue(mock_log_tokens.called)

    def test_gemini15_conversation(self, mock_log_tokens):
        """Test conversation history with Gemini 1.5 Pro"""
        ai = AI(model_identifier="gemini1.5") # Fresh instance for conversation test
        response1 = ai.conversation("My favorite color is blue.")
        self.assertIsInstance(response1.content, str)

        response2 = ai.conversation("What is my favorite color?")
        self.assertIsInstance(response2.content, str)
        self.assertIn("blue", response2.content.lower())
        self.assertEqual(mock_log_tokens.call_count, 4) # 2 calls for each conversation turn

if __name__ == '__main__':
    unittest.main() 
import unittest
import os
from unittest.mock import patch, MagicMock
from ai_core.models import get_client, get_model
from ai_core.wrappers import (
    ClaudeWrapper, GeminiWrapper, GPTWrapper, DeepSeekWrapper,
    PerplexityWrapper, MockWrapper
)

class TestGetClient(unittest.TestCase):

    def test_get_model_mapping(self):
        self.assertEqual(get_model("haiku"), "claude-3-haiku-20240307")
        self.assertEqual(get_model("gpt4o"), "gpt-4o")
        self.assertEqual(get_model("gemini1.5"), "gemini-1.5-pro-latest")
        self.assertEqual(get_model("deepseek-chat"), "deepseek-chat")
        self.assertEqual(get_model("sonar-pro"), "sonar-pro")
        self.assertEqual(get_model("mock"), "mock-") # Mock uses trailing hyphen convention
        self.assertEqual(get_model("unknown-model"), "unknown-model") # Pass through unknown

    # Helper to patch environment variables
    def _patch_env(self, vars_to_set):
        return patch.dict(os.environ, vars_to_set, clear=True)

    def test_get_client_claude(self):
        with self._patch_env({"ANTHROPIC_API_KEY": "fake_claude_key"}):
            client = get_client("haiku")
            self.assertIsInstance(client, ClaudeWrapper)
        # Test with direct key
        client = get_client("sonnet", claude_api_key="direct_claude_key")
        self.assertIsInstance(client, ClaudeWrapper)
        # Test missing key
        with self._patch_env({}): # Clear environment
            with self.assertRaisesRegex(ValueError, "Claude API key must be provided"):
                get_client("opus")

    def test_get_client_gemini(self):
        with self._patch_env({"GOOGLE_API_KEY": "fake_gemini_key"}):
            client = get_client("gemini1.5")
            self.assertIsInstance(client, GeminiWrapper)
            self.assertEqual(client.model.model_name, 'models/gemini-1.5-pro-latest') # Check model name passed
        # Test with direct key
        client = get_client("gemini1.0", gemini_api_key="direct_gemini_key")
        self.assertIsInstance(client, GeminiWrapper)
        # Test missing key
        with self._patch_env({}):
            with self.assertRaisesRegex(ValueError, "Gemini API key must be provided"):
                get_client("gemini1.5")

    def test_get_client_openai(self):
        with self._patch_env({"OPENAI_API_KEY": "fake_openai_key"}):
            client = get_client("gpt4o")
            self.assertIsInstance(client, GPTWrapper)
            # Test org from env
            with self._patch_env({"OPENAI_API_KEY": "fake_key", "OPENAI_ORG_ID": "org_from_env"}):
                client_with_org = get_client("gpt4")
                self.assertIsInstance(client_with_org, GPTWrapper)
                # Assuming GPTWrapper stores org, add check here if it does
                # self.assertEqual(client_with_org.org, "org_from_env") # Example
        # Test with direct key and org
        client = get_client("gpt3.5", openai_api_key="direct_openai_key", openai_org="direct_org")
        self.assertIsInstance(client, GPTWrapper)
        # self.assertEqual(client.org, "direct_org") # Example
        # Test missing key
        with self._patch_env({}):
            with self.assertRaisesRegex(ValueError, "OpenAI API key must be provided"):
                get_client("gpt4o")
            # Also test o1/o3 models
            with self.assertRaisesRegex(ValueError, "OpenAI API key must be provided"):
                get_client("o1-preview")

    def test_get_client_deepseek(self):
        with self._patch_env({"DEEPSEEK_API_KEY": "fake_deepseek_key"}):
            client = get_client("deepseek-chat")
            self.assertIsInstance(client, DeepSeekWrapper)
        # Test with direct key
        client = get_client("deepseek-reasoner", deepseek_api_key="direct_deepseek_key")
        self.assertIsInstance(client, DeepSeekWrapper)
        # Test missing key
        with self._patch_env({}):
            with self.assertRaisesRegex(ValueError, "DeepSeek API key must be provided"):
                get_client("deepseek-chat")

    def test_get_client_perplexity(self):
        with self._patch_env({"PERPLEXITY_API_KEY": "fake_perplexity_key"}):
            client = get_client("sonar")
            self.assertIsInstance(client, PerplexityWrapper)
        # Test with direct key
        client = get_client("sonar-pro", perplexity_api_key="direct_perplexity_key")
        self.assertIsInstance(client, PerplexityWrapper)
        # Test missing key
        with self._patch_env({}):
            with self.assertRaisesRegex(ValueError, "Perplexity API key must be provided"):
                get_client("sonar")

    def test_get_client_mock(self):
        # Mock doesn't need keys
        with self._patch_env({}):
             client = get_client("mock")
             self.assertIsInstance(client, MockWrapper)

    def test_get_client_unsupported(self):
        with self._patch_env({}): # No keys needed for this check
            with self.assertRaisesRegex(ValueError, "Unsupported model provider"):
                get_client("unsupported-provider-model")

if __name__ == '__main__':
    unittest.main()

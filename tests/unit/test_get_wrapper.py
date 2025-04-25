import unittest
import os
from unittest.mock import patch, MagicMock
from ai_core.models import get_wrapper, resolve_model_info
from ai_core.wrappers import (
    ClaudeWrapper, GeminiWrapper, GPTWrapper, DeepSeekWrapper,
    PerplexityWrapper, MockWrapper
)

class TestGetWrapper(unittest.TestCase):

    def test_resolve_model_info(self):
        self.assertEqual(resolve_model_info("haiku"), ("anthropic", "claude-3-haiku-20240307"))
        self.assertEqual(resolve_model_info("gpt4o"), ("openai", "gpt-4o"))
        self.assertEqual(resolve_model_info("gemini1.5"), ("google", "gemini-1.5-pro-latest"))
        self.assertEqual(resolve_model_info("deepseek-chat"), ("deepseek", "deepseek-chat"))
        self.assertEqual(resolve_model_info("sonar-pro"), ("perplexity", "sonar-pro"))
        self.assertEqual(resolve_model_info("mock"), ("mock", "mock-model"))

        self.assertRaises(ValueError, resolve_model_info, "unknown-model")

    # Helper to patch environment variables
    def _patch_env(self, vars_to_set):
        return patch.dict(os.environ, vars_to_set, clear=True)

    def test_get_wrapper_claude(self):
        with self._patch_env({"ANTHROPIC_API_KEY": "fake_claude_key"}):
            wrapper = get_wrapper("haiku")
            self.assertIsInstance(wrapper, ClaudeWrapper)
        # Test with direct key
        wrapper = get_wrapper("sonnet", claude_api_key="direct_claude_key")
        self.assertIsInstance(wrapper, ClaudeWrapper)
        # Test missing key
        with self._patch_env({}): # Clear environment
            with self.assertRaisesRegex(ValueError, "Anthropic API key must be provided"):
                get_wrapper("opus")

    def test_get_wrapper_gemini(self):
        with self._patch_env({"GOOGLE_API_KEY": "fake_gemini_key"}):
            wrapper = get_wrapper("gemini1.5")
            self.assertIsInstance(wrapper, GeminiWrapper)
            self.assertEqual(wrapper.model.model_name, 'models/gemini-1.5-pro-latest') # Check model name passed
        # Test with direct key
        wrapper = get_wrapper("gemini1.0", gemini_api_key="direct_gemini_key")
        self.assertIsInstance(wrapper, GeminiWrapper)
        # Test missing key
        with self._patch_env({}):
            with self.assertRaisesRegex(ValueError, "Google API key must be provided"):
                get_wrapper("gemini1.5")

    def test_get_wrapper_openai(self):
        with self._patch_env({"OPENAI_API_KEY": "fake_openai_key"}):
            wrapper = get_wrapper("gpt4o")
            self.assertIsInstance(wrapper, GPTWrapper)
            # Test org from env
            with self._patch_env({"OPENAI_API_KEY": "fake_key", "OPENAI_ORG_ID": "org_from_env"}):
                wrapper_with_org = get_wrapper("gpt4")
                self.assertIsInstance(wrapper_with_org, GPTWrapper)
                # Assuming GPTWrapper stores org, add check here if it does
                # self.assertEqual(wrapper_with_org.org, "org_from_env") # Example
        # Test with direct key and org
        wrapper = get_wrapper("gpt3.5", openai_api_key="direct_openai_key", openai_org="direct_org")
        self.assertIsInstance(wrapper, GPTWrapper)
        # self.assertEqual(wrapper.org, "direct_org") # Example
        # Test missing key
        with self._patch_env({}):
            with self.assertRaisesRegex(ValueError, "OpenAI API key must be provided"):
                get_wrapper("gpt4o")
            # Also test o1/o3 models
            with self.assertRaisesRegex(ValueError, "OpenAI API key must be provided"):
                get_wrapper("o1-preview")

    def test_get_wrapper_deepseek(self):
        with self._patch_env({"DEEPSEEK_API_KEY": "fake_deepseek_key"}):
            wrapper = get_wrapper("deepseek-chat")
            self.assertIsInstance(wrapper, DeepSeekWrapper)
        # Test with direct key
        wrapper = get_wrapper("deepseek-reasoner", deepseek_api_key="direct_deepseek_key")
        self.assertIsInstance(wrapper, DeepSeekWrapper)
        # Test missing key
        with self._patch_env({}):
            with self.assertRaisesRegex(ValueError, "DeepSeek API key must be provided"):
                get_wrapper("deepseek-chat")

    def test_get_wrapper_perplexity(self):
        with self._patch_env({"PERPLEXITY_API_KEY": "fake_perplexity_key"}):
            wrapper = get_wrapper("sonar")
            self.assertIsInstance(wrapper, PerplexityWrapper)
        # Test with direct key
        wrapper = get_wrapper("sonar-pro", perplexity_api_key="direct_perplexity_key")
        self.assertIsInstance(wrapper, PerplexityWrapper)
        # Test missing key
        with self._patch_env({}):
            with self.assertRaisesRegex(ValueError, "Perplexity API key must be provided"):
                get_wrapper("sonar")

    def test_get_wrapper_mock(self):
        # Mock doesn't need keys
        with self._patch_env({}):
             wrapper = get_wrapper("mock")
             self.assertIsInstance(wrapper, MockWrapper)

    def test_get_wrapper_unsupported(self):
        with self._patch_env({}): # No keys needed for this check
            with self.assertRaisesRegex(ValueError, "Unknown or unsupported model identifier alias"):
                get_wrapper("unsupported-provider-model")

if __name__ == '__main__':
    unittest.main()

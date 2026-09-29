import unittest
from unittest.mock import MagicMock
from ai_core.wrappers import ClaudeWrapper
from ai_core.types import Message, MessageContent


class TestClaudeWrapperArguments(unittest.TestCase):

    def _call(self, model, **kwargs):
        wrapper = ClaudeWrapper(api_key="fake")
        wrapper.client = MagicMock()
        wrapper.client.messages.create.return_value.content = []
        message = Message(role="user", content=[MessageContent(type="text", text="hi")])
        wrapper._messages(model, [message], "", None, 0.0, **kwargs)
        return wrapper.client.messages.create.call_args.kwargs

    def test_legacy_model_sends_temperature(self):
        args = self._call("claude-sonnet-4-6")
        self.assertEqual(args["temperature"], 0.0)

    def test_legacy_model_uses_budget_thinking(self):
        args = self._call("claude-opus-4-6", thinking=True)
        self.assertEqual(args["thinking"]["type"], "enabled")
        self.assertEqual(args["temperature"], 1.0)

    def test_new_model_omits_temperature(self):
        for model in ["claude-opus-4-7", "claude-sonnet-5-5", "claude-opus-5-5"]:
            args = self._call(model)
            self.assertNotIn("temperature", args, model)

    def test_new_model_uses_adaptive_thinking(self):
        args = self._call("claude-sonnet-5-5", thinking=True)
        self.assertEqual(args["thinking"], {"type": "adaptive"})
        self.assertNotIn("temperature", args)


if __name__ == "__main__":
    unittest.main()

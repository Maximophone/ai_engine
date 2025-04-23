import unittest
from unittest.mock import patch
from ai_core import AI, Message, MessageContent, tool, ToolResult, ToolCall

# Use a tool definition from unit tests or define a simple one here
@tool(description="Simple test tool", param1="A parameter")
def simple_tool_mock(param1: str) -> str:
    return f"Tool executed with {param1}"

# Patch log_token_use globally for this test module to avoid file IO
@patch('ai_core.wrappers.base.log_token_use')
class TestAIMockIntegration(unittest.TestCase):

    def setUp(self):
        self.ai = AI(
            model_name="mock",
            system_prompt="Mock system prompt.",
            tools=[simple_tool_mock]
        )

    def test_message_simple(self, mock_log_tokens):
        response = self.ai.message("Hello mock AI")
        self.assertIsInstance(response.content, str)
        self.assertIn("---SYSTEM PROMPT START---", response.content)
        self.assertIn("Mock system prompt.", response.content)
        self.assertIn("---MESSAGES START---", response.content)
        self.assertIn("role: user", response.content)
        self.assertIn("Hello mock AI", response.content)
        self.assertIsNone(response.tool_calls)
        self.assertIsNone(response.reasoning)
        # Check token logging was called (input and output)
        self.assertEqual(mock_log_tokens.call_count, 2)

    def test_message_with_tool_schema(self, mock_log_tokens):
        # Check if the tool schema is included in the mock output
        response = self.ai.message("Use the tool")
        self.assertIn("---TOOLS START---", response.content)
        self.assertIn("Tool: simple_tool_mock", response.content)
        self.assertIn("Description: Simple test tool", response.content)
        self.assertIn("param1:", response.content)
        self.assertIn("type: string", response.content)
        self.assertTrue(mock_log_tokens.called)

    def test_conversation_history(self, mock_log_tokens):
        response1 = self.ai.conversation("First message")
        self.assertEqual(len(self.ai._history), 2) # user + assistant
        self.assertEqual(self.ai._history[0].role, "user")
        self.assertEqual(self.ai._history[0].content[0].text, "First message")
        self.assertEqual(self.ai._history[1].role, "assistant")
        self.assertIn("First message", self.ai._history[1].content[0].text) # Mock includes input

        response2 = self.ai.conversation("Second message")
        self.assertEqual(len(self.ai._history), 4) # user, assistant, user, assistant
        self.assertEqual(self.ai._history[2].role, "user")
        self.assertEqual(self.ai._history[2].content[0].text, "Second message")
        self.assertEqual(self.ai._history[3].role, "assistant")
        # Mock response includes previous messages in its output
        self.assertIn("First message", response2.content)
        self.assertIn("Second message", response2.content)

        self.assertTrue(mock_log_tokens.called)

    def test_thinking_flag(self, mock_log_tokens):
        response = self.ai.message("Think about this", thinking=True, thinking_budget_tokens=500)
        self.assertIn("thinking: enabled", response.content)
        self.assertIn("thinking_budget_tokens: 500", response.content)
        self.assertIsNotNone(response.reasoning)
        self.assertIn("Mock reasoning", response.reasoning)
        self.assertTrue(mock_log_tokens.called)

    def test_temperature_max_tokens(self, mock_log_tokens):
         response = self.ai.message("Test params", temperature=0.5, max_tokens=100)
         self.assertIn("temperature: 0.5", response.content)
         self.assertIn("max_tokens: 100", response.content)
         self.assertTrue(mock_log_tokens.called)

    def test_debug_output(self, mock_log_tokens):
        # Capture print output
        with patch('builtins.print') as mock_print:
            self.ai.message("Debug test", debug=True)
            # Check if debug headers were printed
            printed_output = " ".join(call.args[0] for call in mock_print.call_args_list if call.args)
            self.assertIn("--MODEL: mock-", printed_output)
            self.assertIn("--SYSTEM PROMPT START--", printed_output)
            self.assertIn("--MESSAGES RECEIVED START--", printed_output)
            self.assertIn("--RESPONSE START--", printed_output)

        self.assertTrue(mock_log_tokens.called)

    # def test_tool_call_and_result_flow(self, mock_log_tokens):
    #     # NOTE: MockWrapper doesn't simulate tool *calls* back, just shows schema.
    #     #       A more advanced mock could potentially simulate this.
    #     #       We can test the *formatting* of tool results *sent* to the mock.
    #     tool_call = ToolCall(id="call123", name="simple_tool_mock", arguments={"param1": "value"})
    #     tool_result = ToolResult(name="simple_tool_mock", result="Mock Result", tool_call_id="call123")

    #     messages = [
    #         Message(role="user", content=[MessageContent(type="text", text="Initial request")]),
    #         Message(role="assistant", content=[MessageContent(type="tool_use", tool_call=tool_call)]),
    #         # Role 'tool' is not standard in our basic types, OpenAI uses it. Claude uses user/assistant.
    #         # Let's simulate sending a result back as a user message (common pattern)
    #         Message(role="user", content=[MessageContent(type="tool_result", tool_result=tool_result)])
    #     ]
    #     response = self.ai.messages(messages)

    #     self.assertIn("[tool_use: ToolCall(name='simple_tool_mock', arguments={'param1': 'value'}, id='call123')]", response.content)
    #     self.assertIn("[tool_result: ToolResult(name='simple_tool_mock', result='Mock Result', tool_call_id='call123', error=None)]", response.content)
    #     self.assertTrue(mock_log_tokens.called)


if __name__ == '__main__':
    unittest.main() 
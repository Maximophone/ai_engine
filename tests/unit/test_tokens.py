import unittest
from unittest.mock import patch, MagicMock
from ai_core.tokens import (
    n_tokens, n_tokens_images, count_tokens_input,
    count_tokens_output, log_token_use
)
from ai_core.types import Message, MessageContent

class TestTokenCounting(unittest.TestCase):

    def test_n_tokens_text(self):
        self.assertEqual(n_tokens("This is a test."), 3)
        self.assertEqual(n_tokens(""), 0)
        self.assertEqual(n_tokens("token"), 1) # Rough approximation

    @patch('ai_core.tokens.get_image_dimensions_from_base64')
    def test_n_tokens_images(self, mock_get_dims):
        mock_get_dims.side_effect = [(1024, 1024), (512, 512)]
        images = [
            {"data": "base64_encoded_string_1"},
            {"data": "base64_encoded_string_2"}
        ]
        # Calculation: (1024*1024 // 750) + (512*512 // 750)
        # = 1398 + 349 = 1747
        expected_tokens = (1024 * 1024 // 750) + (512 * 512 // 750)
        self.assertEqual(n_tokens_images(images), expected_tokens)
        self.assertEqual(mock_get_dims.call_count, 2)

    @patch('ai_core.tokens.get_image_dimensions_from_base64')
    def test_count_tokens_input(self, mock_get_dims):
        mock_get_dims.return_value = (100, 100) # 100*100 // 750 = 13 tokens
        system_prompt = "System prompt." # 3 tokens
        messages = [
            Message(role="user", content=[
                MessageContent(type="text", text="User message 1."), # 4 tokens
                MessageContent(type="image", image={"data": "img1"}) # 13 tokens
            ]),
            Message(role="assistant", content=[
                MessageContent(type="text", text="Assistant response.") # 3 tokens
            ]),
             Message(role="user", content=[
                MessageContent(type="text", text="User message 2.") # 4 tokens
            ]),
        ]
        # Total = 3 (sys) + 4 (u1) + 13 (img1) + 3 (a1) + 4 (u2) = 27
        expected_tokens = 3 + 4 + 13 + 3 + 4
        self.assertEqual(count_tokens_input(messages, system_prompt), expected_tokens)
        mock_get_dims.assert_called_once_with("img1")

    def test_count_tokens_output(self):
        self.assertEqual(count_tokens_output("This is a response."), 4)
        self.assertEqual(count_tokens_output(""), 0)

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('ai_core.tokens.dt')
    @patch('ai_core.tokens.sys')
    def test_log_token_use(self, mock_sys, mock_dt, mock_open):
        mock_now = MagicMock()
        mock_now.strftime.return_value = "2023-01-01 12:00:00"
        mock_dt.now.return_value = mock_now # Incorrect usage, should be just `dt.now()`
        mock_dt.now.return_value = "2023-01-01T12:00:00" # Corrected mock
        mock_sys.argv = ["test_script.py"]

        # Test input logging
        log_token_use("test-model", 100, input=True, fpath="dummy.csv")
        mock_open().assert_any_call("dummy.csv", "a+")
        mock_open().write.assert_called_with("test-model,input,100,2023-01-01T12:00:00,test_script.py\n")

        # Reset mock call args for next assert
        mock_open().write.reset_mock()

        # Test output logging
        log_token_use("test-model", 50, input=False, fpath="dummy.csv")
        mock_open().write.assert_called_with("test-model,output,50,2023-01-01T12:00:00,test_script.py\n")

        mock_open.assert_called_with("dummy.csv", "a+")


if __name__ == '__main__':
    unittest.main()

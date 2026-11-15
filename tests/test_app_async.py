import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import app
from utils.common import ModalType


class AppAsyncFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_handle_user_input_awaits_search_on_current_loop(self):
        session_state = SimpleNamespace(messages=[])
        user_message = Mock()
        placeholder = Mock()
        search = AsyncMock(return_value="answer")

        with (
            patch.object(app.st, "chat_input", return_value="question"),
            patch.object(
                app.st, "chat_message", side_effect=[user_message, nullcontext()]
            ),
            patch.object(app.st, "spinner", return_value=nullcontext()),
            patch.object(app.st, "empty", return_value=placeholder),
            patch.object(app.st, "session_state", session_state),
            patch.object(app, "search", search),
        ):
            await app.handle_user_input("hybrid", ModalType.TEXT_ONLY)

        search.assert_awaited_once_with(
            "hybrid",
            query="question",
            modal=ModalType.TEXT_ONLY,
            img_base64=None,
        )
        user_message.markdown.assert_called_once_with("question")
        placeholder.markdown.assert_called_once_with("answer")
        self.assertEqual(
            session_state.messages,
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        )


if __name__ == "__main__":
    unittest.main()

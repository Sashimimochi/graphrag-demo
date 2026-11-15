import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from utils.common import ModalType
from utils import rag as rag_module


class AsyncRagApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_make_index_uses_light_rag_async_insert(self):
        light_rag = SimpleNamespace(
            ainsert=AsyncMock(),
            finalize_storages=AsyncMock(),
            workspace="",
            graph_storage="Neo4JStorage",
        )
        rag_anything = SimpleNamespace(
            process_folder_complete=AsyncMock(),
        )

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "sample.txt").write_text("sample document", encoding="utf-8")
            with (
                patch.object(rag_module, "DATA_DIR", directory),
                patch.object(
                    rag_module, "initialize_rag", new=AsyncMock(return_value=light_rag)
                ),
                patch.object(
                    rag_module, "initialize_rag_anything", return_value=rag_anything
                ),
                patch.object(
                    rag_module.st,
                    "session_state",
                    SimpleNamespace(working_dir=directory),
                ),
                patch.object(rag_module.st, "info"),
                patch.object(rag_module.st, "success"),
                patch.object(rag_module.st, "error"),
                patch.object(rag_module, "finalize_pipeline_ingress", new=AsyncMock()) as finalize_ingress,
            ):
                await rag_module.make_index("sample")

        light_rag.ainsert.assert_awaited_once_with("sample document")
        rag_anything.process_folder_complete.assert_awaited_once()
        light_rag.finalize_storages.assert_awaited_once()
        finalize_ingress.assert_awaited_once_with(workspace="")

    async def test_text_only_search_surfaces_lightrag_failure_message(self):
        light_rag = SimpleNamespace(
            aquery_llm=AsyncMock(return_value={
                "status": "failure",
                "message": "embedding provider returned 401",
                "llm_response": {"content": None, "is_streaming": False},
            }),
            finalize_storages=AsyncMock(),
            workspace="test-workspace",
        )
        rag_anything = SimpleNamespace()

        with (
            patch.object(
                rag_module, "initialize_rag", new=AsyncMock(return_value=light_rag)
            ),
            patch.object(
                rag_module, "initialize_rag_anything", return_value=rag_anything
            ),
            patch.object(rag_module, "finalize_pipeline_ingress", new=AsyncMock()),
            patch.object(rag_module.st, "error") as show_error,
        ):
            result = await rag_module.search(
                mode="naive", query="test question", modal=ModalType.TEXT_ONLY
            )

        self.assertIn("embedding provider returned 401", result)
        show_error.assert_called_once_with(
            "Search failed: embedding provider returned 401"
        )

    async def _run_search(self, modal, img_base64=None):
        light_rag = SimpleNamespace(
            aquery_llm=AsyncMock(return_value={
                "status": "success",
                "llm_response": {"content": "text result", "is_streaming": False},
            }),
            finalize_storages=AsyncMock(),
            workspace="test-workspace",
        )
        rag_anything = SimpleNamespace(
            aquery=AsyncMock(return_value="multimodal result"),
            aquery_with_multimodal=AsyncMock(return_value="image result"),
        )

        with (
            patch.object(
                rag_module, "initialize_rag", new=AsyncMock(return_value=light_rag)
            ),
            patch.object(
                rag_module, "initialize_rag_anything", return_value=rag_anything
            ),
            patch.object(rag_module, "finalize_pipeline_ingress", new=AsyncMock()) as finalize_ingress,
        ):
            result = await rag_module.search(
                mode="hybrid",
                query="test question",
                modal=modal,
                img_base64=img_base64,
            )

        light_rag.finalize_storages.assert_awaited_once()
        finalize_ingress.assert_awaited_once_with(workspace="test-workspace")
        return result, light_rag, rag_anything

    async def test_text_only_search_uses_async_light_rag_query(self):
        result, light_rag, _ = await self._run_search(ModalType.TEXT_ONLY)

        self.assertEqual(result, "text result")
        light_rag.aquery_llm.assert_awaited_once()

    async def test_multimodal_search_uses_async_rag_anything_query(self):
        result, _, rag_anything = await self._run_search(ModalType.MULTIMODAL)

        self.assertEqual(result, "multimodal result")
        rag_anything.aquery.assert_awaited_once()

    async def test_image_input_uses_async_multimodal_query(self):
        result, _, rag_anything = await self._run_search(
            ModalType.MULTIMODAL_INPUT,
            img_base64="aW1hZ2U=",
        )

        self.assertEqual(result, "image result")
        rag_anything.aquery_with_multimodal.assert_awaited_once()

    async def test_both_mode_uses_both_async_query_apis(self):
        result, light_rag, rag_anything = await self._run_search(ModalType.BOTH)

        self.assertIn("text result", result)
        self.assertIn("multimodal result", result)
        light_rag.aquery_llm.assert_awaited_once()
        rag_anything.aquery.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest

import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from utils.common import ModalType
from utils import rag as rag_module
from lightrag.utils import EmbeddingFunc


class AsyncRagApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_embedding_passes_configured_dimension(self):
        embedding = EmbeddingFunc(
            embedding_dim=768,
            max_token_size=2048,
            send_dimensions=True,
            func=rag_module.embedding_func,
        )
        vectors = np.ones((2, 768), dtype=np.float32)
        with (
            patch.object(rag_module, "EMBEDDING_MODEL", "gemini-embedding-001"),
            patch.object(rag_module, "EMBEDDING_DIM", 768),
            patch.object(rag_module, "MAX_TOKENS", 2048),
            patch.object(
                rag_module, "openai_embed", new=AsyncMock(return_value=vectors)
            ) as openai_embed,
        ):
            result = await embedding(["first chunk", "second chunk"])

        self.assertEqual(result.shape, (2, 768))
        openai_embed.assert_awaited_once_with(
            ["first chunk", "second chunk"],
            model="gemini-embedding-001",
            api_key=rag_module.API_KEY,
            base_url=rag_module.BASE_URL,
            embedding_dim=768,
            max_token_size=2048,
        )

    async def test_make_index_uses_async_insert_and_checks_graphml(self):
        light_rag = SimpleNamespace(
            ainsert=AsyncMock(),
            finalize_storages=AsyncMock(),
            workspace="workspace-a",
            graph_storage="NetworkXStorage",
        )
        rag_anything = SimpleNamespace(process_folder_complete=AsyncMock())

        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory, "inputs")
            input_dir.mkdir()
            Path(input_dir, "sample.txt").write_text("sample document", encoding="utf-8")
            graph_dir = Path(directory, "workspace-a")

            async def insert_and_write_graph(_content):
                graph_dir.mkdir()
                (graph_dir / "graph_chunk_entity_relation.graphml").write_text("<graphml/>", encoding="utf-8")
                return "track-123"

            light_rag.ainsert.side_effect = insert_and_write_graph
            light_rag.working_dir = directory
            with (
                patch.object(rag_module, "DATA_DIR", str(input_dir)),
                patch.object(rag_module, "initialize_rag", new=AsyncMock(return_value=light_rag)),
                patch.object(rag_module, "initialize_rag_anything", return_value=rag_anything),
                patch.object(rag_module.st, "session_state", SimpleNamespace(working_dir=directory)),
                patch.object(rag_module.st, "info"),
                patch.object(rag_module.st, "success"),
                patch.object(rag_module.st, "error"),
                patch.object(rag_module, "finalize_pipeline_ingress", new=AsyncMock()) as finalize_ingress,
            ):
                await rag_module.make_index("sample")

        light_rag.ainsert.assert_awaited_once_with("sample document")
        rag_anything.process_folder_complete.assert_awaited_once()
        light_rag.finalize_storages.assert_awaited_once()
        finalize_ingress.assert_awaited_once_with(workspace="workspace-a")

    async def test_make_index_reports_missing_graphml(self):
        light_rag = SimpleNamespace(
            ainsert=AsyncMock(return_value="track-123"),
            finalize_storages=AsyncMock(),
            workspace="",
            graph_storage="NetworkXStorage",
            working_dir="unused",
        )
        rag_anything = SimpleNamespace(process_folder_complete=AsyncMock())

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "sample.txt").write_text("sample document", encoding="utf-8")
            with (
                patch.object(rag_module, "DATA_DIR", directory),
                patch.object(rag_module, "initialize_rag", new=AsyncMock(return_value=light_rag)),
                patch.object(rag_module, "initialize_rag_anything", return_value=rag_anything),
                patch.object(rag_module.st, "session_state", SimpleNamespace(working_dir=directory)),
                patch.object(rag_module.st, "info"),
                patch.object(rag_module.st, "success"),
                patch.object(rag_module.st, "error") as show_error,
                patch.object(rag_module, "finalize_pipeline_ingress", new=AsyncMock()),
            ):
                await rag_module.make_index("sample")

        self.assertIn("GraphMLが保存されていません", show_error.call_args.args[0])
        rag_anything.process_folder_complete.assert_not_awaited()

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

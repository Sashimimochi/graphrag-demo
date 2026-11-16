import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from graph_visualize import cluster_graph, graphml_to_json, resolve_graphml_path


class ResolveGraphMLPathTests(unittest.TestCase):
    def test_uses_workspace_subdirectory_when_configured(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_dir = Path(directory, "tenant-a")
            graph_dir.mkdir()
            graph_file = graph_dir / "graph_chunk_entity_relation.graphml"
            graph_file.write_text("<graphml/>", encoding="utf-8")

            with patch.dict(os.environ, {"WORKSPACE": "tenant-a"}):
                self.assertEqual(resolve_graphml_path(directory), str(graph_file))

    def test_falls_back_to_legacy_root_path(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_file = Path(directory, "graph_chunk_entity_relation.graphml")
            graph_file.write_text("<graphml/>", encoding="utf-8")

            with patch.dict(os.environ, {"WORKSPACE": "tenant-a"}):
                self.assertEqual(resolve_graphml_path(directory), str(graph_file))

    def test_uses_root_path_when_workspace_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_file = Path(directory, "graph_chunk_entity_relation.graphml")
            graph_file.write_text("<graphml/>", encoding="utf-8")

            with patch.dict(os.environ, {"WORKSPACE": ""}):
                self.assertEqual(resolve_graphml_path(directory), str(graph_file))

class GraphSerializationTests(unittest.TestCase):
    def test_node_link_json_uses_d3_links_key(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_file = Path(directory, "sample.graphml")
            graph = nx.Graph()
            graph.add_node("alpha", entity_type="person")
            graph.add_node("beta", entity_type="organization")
            graph.add_edge("alpha", "beta", weight=2.0)
            nx.write_graphml(graph, graph_file)

            data = json.loads(graphml_to_json(graph_file))

        self.assertEqual(len(data["nodes"]), 2)
        self.assertEqual(len(data["links"]), 1)
        self.assertEqual(data["links"][0]["source"], "alpha")
        self.assertEqual(data["links"][0]["target"], "beta")
        self.assertNotIn("edges", data)


class ClusterGraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        for nodes in (("a", "b", "c"), ("d", "e", "f")):
            self.graph.add_edges_from(
                (source, target, {"weight": 1.0})
                for index, source in enumerate(nodes)
                for target in nodes[index + 1 :]
            )
        self.graph.add_edge("c", "d", weight=0.01)
        self.graph.add_edge("outside-a", "outside-b", weight=1.0)

    def test_empty_graph_returns_no_communities(self):
        self.assertEqual(
            cluster_graph(nx.Graph(), {"max_cluster_size": 3}),
            [],
        )

    def test_hierarchy_shape_and_largest_component_filter(self):
        communities = cluster_graph(
            self.graph,
            {
                "type": "leiden",
                "max_cluster_size": 3,
                "use_lcc": True,
                "seed": 0xDEADBEEF,
                "levels": None,
            },
        )

        self.assertTrue(communities)
        self.assertTrue(all(len(row) == 4 for row in communities))
        member_ids = {
            node
            for level, community, parent, members in communities
            for node in members
        }
        self.assertEqual(member_ids, {"a", "b", "c", "d", "e", "f"})
        self.assertTrue(
            all(
                isinstance(level, int) and isinstance(community, int)
                for level, community, parent, members in communities
            )
        )

    def test_disabling_largest_component_filter_keeps_other_components(self):
        communities = cluster_graph(
            self.graph,
            {
                "type": "leiden",
                "max_cluster_size": 3,
                "use_lcc": False,
                "seed": 0xDEADBEEF,
                "levels": None,
            },
        )
        member_ids = {
            node
            for level, community, parent, members in communities
            for node in members
        }
        self.assertIn("outside-a", member_ids)
        self.assertIn("outside-b", member_ids)


if __name__ == "__main__":
    unittest.main()

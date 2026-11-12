import unittest

import networkx as nx

from graph_visualize import cluster_graph


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

import json
import os
import networkx as nx
import pandas as pd
from pyvis.network import Network
from graphrag.index.operations.cluster_graph import cluster_graph


def create_simple_html(dataset):
  # Load the GraphML file
  filepath = f'./{dataset}/graph_chunk_entity_relation.graphml'
  G = nx.read_graphml(filepath)

  # Create a Pyvis network
  net = Network(notebook=True)

  # Convert NetworkX graph to Pyvis network
  net.from_nx(G)

  # Save and display the network
  net.show(f'./visualize/knowledge_graph_{dataset}_simple.html')

from yfiles_jupyter_graphs import GraphWidget
import streamlit as st

def show_hierarchy_graph(dataset):
  # クラスタリングの設定
  strategy = {
    "type": "leiden",
    "max_cluster_size": 10,  # クラスタの最大サイズ
    "use_lcc": True,         # 最大全結合成分のみを使用
    "seed": 0xDEADBEEF,      # ランダムシード
    "levels": None,          # すべてのレベルを使用
    "verbose": True          # ログを表示
  }

  # Load the GraphML file
  G = nx.read_graphml(f'./{dataset}/graph_chunk_entity_relation.graphml')

  # クラスタリングを実行
  communities = cluster_graph(G, strategy)

  # 結果を表示
  print("Detected communities:")
  base_communities = pd.DataFrame(
    communities, columns=pd.Index(["level", "community", "parent", "title"])
  ).explode("title")
  base_communities["community"] = base_communities["community"].astype(int)

  return base_communities

# load GraphML file and transfer to JSON
def graphml_to_json(graphml_file):
    G = nx.read_graphml(graphml_file)
    data = nx.node_link_data(G)
    return json.dumps(data)

# create HTML file
def save_as_html(html_path, graph_json):
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }
        svg {
            width: 100%;
            height: 100%;
        }
        .links line {
            stroke: #999;
            stroke-opacity: 0.6;
        }
        .nodes circle {
            stroke: #fff;
            stroke-width: 1.5px;
        }
        .node-label {
            font-size: 12px;
            pointer-events: none;
        }
        .link-label {
            font-size: 10px;
            fill: #666;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .link:hover .link-label {
            opacity: 1;
        }
        .tooltip {
            position: absolute;
            text-align: left;
            padding: 10px;
            font: 12px sans-serif;
            background: lightsteelblue;
            border: 0px;
            border-radius: 8px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
            max-width: 300px;
        }
        .legend {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 5px;
        }
        .legend-item {
            margin: 5px 0;
        }
        .legend-color {
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            vertical-align: middle;
        }
    </style>
</head>
<body>
    <svg></svg>
    <div class="tooltip"></div>
    <div class="legend"></div>
    <script>
        [GRPH DATA];
        const graphData = graphJson;

        const svg = d3.select("svg"),
            width = window.innerWidth,
            height = window.innerHeight;

        svg.attr("viewBox", [0, 0, width, height]);

        const g = svg.append("g");

        const entityTypes = [...new Set(graphData.nodes.map(d => d.entity_type))];
        const color = d3.scaleOrdinal(d3.schemeCategory10).domain(entityTypes);

        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(150))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collide", d3.forceCollide().radius(30));

        const linkGroup = g.append("g")
            .attr("class", "links")
            .selectAll("g")
            .data(graphData.links)
            .enter().append("g")
            .attr("class", "link");

        const link = linkGroup.append("line")
            .attr("stroke-width", d => Math.sqrt(d.value));

        const linkLabel = linkGroup.append("text")
            .attr("class", "link-label")
            .text(d => d.description || "");

        const node = g.append("g")
            .attr("class", "nodes")
            .selectAll("circle")
            .data(graphData.nodes)
            .enter().append("circle")
            .attr("r", 5)
            .attr("fill", d => color(d.entity_type))
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        const nodeLabel = g.append("g")
            .attr("class", "node-labels")
            .selectAll("text")
            .data(graphData.nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .text(d => d.id);

        const tooltip = d3.select(".tooltip");

        node.on("mouseover", function(event, d) {
            tooltip.transition()
                .duration(200)
                .style("opacity", .9);
            tooltip.html(`<strong>${d.id}</strong><br>Entity Type: ${d.entity_type}<br>Description: ${d.description || "N/A"}`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function(d) {
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        });

        const legend = d3.select(".legend");
        entityTypes.forEach(type => {
            legend.append("div")
                .attr("class", "legend-item")
                .html(`<span class="legend-color" style="background-color: ${color(type)}"></span>${type}`);
        });

        simulation
            .nodes(graphData.nodes)
            .on("tick", ticked);

        simulation.force("link")
            .links(graphData.links);

        function ticked() {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            linkLabel
                .attr("x", d => (d.source.x + d.target.x) / 2)
                .attr("y", d => (d.source.y + d.target.y) / 2)
                .attr("text-anchor", "middle")
                .attr("dominant-baseline", "middle");

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            nodeLabel
                .attr("x", d => d.x + 8)
                .attr("y", d => d.y + 3);
        }

        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }

        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }

        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", zoomed);

        svg.call(zoom);

        function zoomed(event) {
            g.attr("transform", event.transform);
        }

    </script>
</body>
</html>
    '''

    html_content = html_content.replace("[GRPH DATA]", graph_json)

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def create_json(json_data):
    json_data = "var graphJson = " + json_data.replace('\\"', '').replace("'", "\\'").replace("\n", "")

    return json_data

# main function
def visualize_graphml(dataset, html_path):
    graphml_file = f'./{dataset}/graph_chunk_entity_relation.graphml'
    json_data = graphml_to_json(graphml_file)
    html_dir = os.path.dirname(html_path)
    if not os.path.exists(html_dir):
        os.makedirs(html_dir)

    save_as_html(html_path, create_json(json_data))

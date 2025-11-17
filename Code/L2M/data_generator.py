import networkx as nx
import numpy as np

def generate_single_er_graph(num_nodes, edge_prob, output_file):
    # Generate ER graph
    G = nx.erdos_renyi_graph(n=num_nodes, p=edge_prob, directed=False)
    
    # Add edge attributes to match the reading format
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]['eid'] = i  # Edge ID
        G[u][v]['weight'] = np.random.uniform(0.1, 10.0)  # Random weight
        G[u][v]['label'] = 0  # Default label (you can change this as needed)
    
    # Save as .gpickle
    nx.write_gpickle(G, output_file)
    
    print(f"Generated ER graph: {num_nodes} nodes, {G.number_of_edges()} edges")
    print(f"Saved to: {output_file}")
    
    return G

# Generate the graph
generate_single_er_graph(
    num_nodes=1000,
    edge_prob=0.15,
    output_file="./Datasets/er/er_25_35_0.15/test/a.gpickle"
)
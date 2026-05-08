import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data

def build_graph_dataset(df, feature_cols, target_col):
    graph_list = []
    
    # 1. Ensure the target column is encoded as integers (0, 1, 2...)
    # This avoids "RuntimeError: expected scalar type Long"
    df[target_col] = pd.Categorical(df[target_col]).codes
    
    for star_name, group in df.groupby('hostname'):
        # Node Features (X)
        x = torch.tensor(group[feature_cols].values, dtype=torch.float)
        
        # Target (y)
        y = torch.tensor(group[target_col].values, dtype=torch.long)
        
        num_nodes = x.shape[0]
        
        if num_nodes > 1:
            # Create all possible combinations (undirected)
            # This connects every planet in the system to every other planet
            adj = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)
            edge_index = adj.nonzero().t().contiguous()
        else:
            # For 1-planet systems, we create an empty edge_index of the correct shape
            # or a self-loop. Let's go with a self-loop [0, 0] to keep the graph valid.
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            
        # Create the Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        data.star_name = star_name 
        
        graph_list.append(data)
        
    return graph_list



def plot_stellar_graph(graph_list, star_name):
    # 1. Buscamos el grafo específico en nuestra lista
    target_graph = None
    for g in graph_list:
        if g.star_name == star_name:
            target_graph = g
            break
    
    if target_graph is None:
        print(f"Star system '{star_name}' not found.")
        return

    # 2. Convertimos el objeto Data de PyTorch Geometric a un grafo de NetworkX
    G = nx.Graph()
    
    # Añadimos los nodos (planetas)
    num_nodes = target_graph.num_nodes
    for i in range(num_nodes):
        # El label será el índice del planeta
        G.add_node(i)
        
    # Añadimos las aristas (conexiones)
    edges = target_graph.edge_index.t().tolist()
    G.add_edges_from(edges)

    # 3. Dibujamos el grafo
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42) # Layout para que se vea ordenado
    
    nx.draw(G, pos, 
            with_labels=True, 
            node_color='skyblue', 
            node_size=1200, 
            edge_color='gray', 
            font_weight='bold', 
            font_size=12)
    
    plt.title(f"Graph Topology for System: {star_name}")
    plt.show()
import os

import numpy as np
import pandas as pd
import networkx as nx

import torch
from torch_geometric.data import Data
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns



def build_graph_dataset(df, feature_cols, target_col):
    """
    Converts a planetary dataframe into a list of PyTorch Geometric graphs.

    Each graph represents a single stellar system (hostname). In these graphs, 
    planets are nodes and their gravitational/spatial relationships are edges.

    Args:
        df (pd.DataFrame): The input dataset containing planet features and system names.
        feature_cols (list): A list of column names to be used as node features (e.g., mass, period).
        target_col (str): The name of the column we want to predict (habitability).

    Returns:
        list: A list of torch_geometric.data.Data objects. Each object contains:
            - x: Node features (the physical data of the planets).
            - edge_index: Graph connectivity (how planets are linked).
            - y: The target labels for each planet.
            - star_name: The name of the host star for reference.
    """
    df = df.copy()
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
    """
    Visualizes the graph topology of a specific stellar system.

    This function converts a PyTorch Geometric Data object into a NetworkX graph 
    to create a 2D visualization of the planetary system's structure.

    Args:
        graph_list (list): A list of torch_geometric.data.Data objects created by build_graph_dataset.
        star_name (str): The name of the host star (hostname) to be searched and visualized.

    Returns:
        None: The function displays a Matplotlib plot showing the nodes (planets) 
              and edges (gravitational/spatial connections).
    """
    # Search for the specific graph in our list
    target_graph = None
    for g in graph_list:
        if g.star_name == star_name:
            target_graph = g
            break
    
    if target_graph is None:
        print(f"Star system '{star_name}' not found.")
        return

    # Convert the PyTorch Geometric Data object to a NetworkX graph
    G = nx.Graph()
    
    # Add nodes (representing planets)
    num_nodes = target_graph.num_nodes
    for i in range(num_nodes):
        # Use the planet index as the node label
        G.add_node(i)
        
    # Add edges (representing gravitational/spatial connections)
    edges = target_graph.edge_index.t().tolist()
    G.add_edges_from(edges)

    # Draw the graph using Matplotlib
    plt.figure(figsize=(8, 6))
    # Use spring_layout for an organized and aesthetic distribution of nodes
    pos = nx.spring_layout(G, seed=111)
    
    nx.draw(G, pos, 
            with_labels=True, 
            node_color='skyblue', 
            node_size=1200, 
            edge_color='gray', 
            font_weight='bold', 
            font_size=12)
    
    plt.title(f"Graph Topology for System: {star_name}")
    plt.show()
    
    



 

def plot_explainable_system(star_name, graph_list, model, device):
    """
    Creates an interpretability map for a specific stellar system.

    This function visualizes how the model "sees" a system by coloring planets 
    based on their predicted probability of habitability, moving from Red (Low) 
    to Green (High).

    Args:
        star_name (str): The name of the stellar system to visualize.
        graph_list (list): List of torch_geometric.data.Data objects.
        model (torch.nn.Module): The trained GNN model.
        device (str): The device for computation ("cpu" or "cuda").

    Returns:
        None: Displays a color-coded graph representing model confidence.
    """
    model.eval()
    
    # Search for the specific system
    target_graph = next((g for g in graph_list if g.star_name == star_name), None)
    if target_graph is None:
        print(f"Sistema {star_name} no encontrado.")
        return

    # Get model predictions and probabilities for this system
    with torch.no_grad():
        data = target_graph.to(device)
        logits = model(data.x, data.edge_index)
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        # Extract probability for Class 1 (Habitable)
        habitable_probs = probs[:, 1].cpu().numpy()
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    # Create a NetworkX graph structure
    G = nx.Graph()
    edges = target_graph.edge_index.t().tolist()
    G.add_edges_from(edges)

    # Configure visual elements (colors and labels)
    # Node colors are mapped to the probability values
    node_colors = habitable_probs 
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=111)
    
    # Draw graph connections (edges) with transparency
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', width=1.5)
    
    # Draw nodes with a Red-Yellow-Green colormap (RdYlGn)
    nodes = nx.draw_networkx_nodes(G, pos, 
                                   node_color=node_colors, 
                                   cmap=plt.cm.RdYlGn, # Red-Yellow-Green
                                   node_size=1500,
                                   edgecolors='black')

    # Add labels showing Planet index (P) and the habitability percentage
    labels = {i: f"P{i}\n{habitable_probs[i]:.2%}" for i in range(len(habitable_probs))}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')

    # Add a color bar legend to translate colors into probability values
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=1))
    plt.colorbar(sm, label="Habitability Probability", ax=plt.gca())

    plt.title(f"Interpretability Map: {star_name}\n(Model's Vision of the System)", fontsize=14)
    plt.axis('off')
    plt.show()





def check_connectivity(graph_list):
    """
    Validates the edge structure for all systems in the dataset.

    This function ensures that multi-planet systems are modeled as "Complete Graphs," 
    meaning every planet is connected to every other planet in its system.

    Args:
        graph_list (list): A list of torch_geometric.data.Data objects.

    Returns:
        None: The function prints a success message or a warning with the error count.
    """
    errors = 0
    for g in graph_list:
        n = g.num_nodes
        # In a fully connected graph (without self-loops): edges = n * (n - 1)
        # For example: 3 nodes should have 3 * 2 = 6 directed edges
        if n > 1:
            expected_edges = n * (n - 1)
            if g.edge_index.shape[1] != expected_edges:
                errors += 1
    
    if errors == 0:
        print("Perfect Connectivity. All multi-planet systems are complete graphs.")
    else:
        print(f"Warning: {errors} systems have unexpected connectivity.")
        
        
        
        
        
        
        
        
        
        
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device="cpu", epochs=200, patience=10, save_path="best_gnn_model.pth"):
    """
    Trains the GNN model using a training and validation loop with Early Stopping.

    This function optimizes the model weights, tracks performance metrics across epochs,
    and saves the best version of the model to prevent overfitting.

    Args:
        model (torch.nn.Module): The GNN model to be trained.
        train_loader (DataLoader): DataLoader containing the training graph dataset.
        val_loader (DataLoader): DataLoader containing the validation graph dataset.
        criterion (loss function): The objective function (e.g., CrossEntropyLoss).
        optimizer (optimizer): The optimization algorithm (e.g., Adam).
        device (str): The device to run the training on ("cpu" or "cuda").
        epochs (int): Maximum number of passes through the complete dataset.
        patience (int): Number of epochs to wait for improvement before stopping.
        save_path (str): File path to save the best model weights.

    Returns:
        tuple: (trained_model, history) where history is a dictionary of metrics.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            # Forward: pass node features and edge index
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.y.size(0)
            preds = out.argmax(dim=1)
            correct_train += (preds == data.y).sum().item()
            total_train += data.y.size(0)

        train_loss /= total_train
        train_acc = correct_train / total_train

        # ---- Validation ----
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index)
                
                loss = criterion(out, data.y)
                val_loss += loss.item() * data.y.size(0)
                preds = out.argmax(dim=1)
                correct_val += (preds == data.y).sum().item()
                total_val += data.y.size(0)

        val_loss /= total_val
        val_acc = correct_val / total_val
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # ---- Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    model.load_state_dict(torch.load(save_path))
    return model, history





def evaluate_model(model, test_loader, device, class_names=['Not Habitable', 'Habitable']):
    """
    Evaluates the GNN performance and generates visual metrics.

    This function tests the model on unseen data, calculates the accuracy of predictions, 
    and plots both a Normalized Confusion Matrix and a comparison bar chart.

    Args:
        model (torch.nn.Module): The trained GNN model.
        test_loader (DataLoader): DataLoader containing the test graph dataset.
        device (str): The device to run evaluation on ("cpu" or "cuda").
        class_names (list): Labels for the categories (e.g., 'Not Habitable', 'Habitable').

    Returns:
        tuple: (all_labels, all_preds) containing the true labels and model predictions.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            # Forward pass: get model scores
            out = model(data.x, data.edge_index)
            # Pick the class with the highest score as the prediction
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Normalized Confusion Matrix')
    plt.show()

    # --- Classification Report ---
    print("\nClassification Report:\n", 
          classification_report(all_labels, all_preds, target_names=class_names))

    # --- Actual vs Predicted Plot ---
    actual_counts = np.bincount(all_labels, minlength=2)
    pred_counts = np.bincount(all_preds, minlength=2)

    plt.figure(figsize=(8,5))
    x = np.arange(len(class_names))
    plt.bar(x - 0.2, actual_counts, width=0.4, label='Actual', color='gray')
    plt.bar(x + 0.2, pred_counts, width=0.4, label='Predicted', color='green')
    plt.xticks(x, class_names)
    plt.ylabel('Number of Planets')
    plt.title('Actual vs Predicted Class Counts')
    plt.legend()
    plt.show()

    return all_labels, all_preds




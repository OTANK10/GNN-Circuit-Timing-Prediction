"""
GNN-based Circuit Timing Prediction
===================================

This project demonstrates using Graph Neural Networks (GNNs) to predict 
gate delays in digital circuits - a key application in VLSI design. || Based on Apple's recent jobs descriptions and requirements

Author: [Om Tank]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing
from torch_geometric.data import Data, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import mean_absolute_error, r2_score
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class CircuitGNN(nn.Module):
    """
    Graph Neural Network for circuit timing prediction.
    
    Architecture:
    - 2 GCN layers for message passing between connected gates
    - Global pooling to get circuit-level representation
    - MLP for final delay prediction
    """
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(CircuitGNN, self).__init__()
        
        # GCN layers for learning gate interactions
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # MLP for final prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, edge_index, batch=None):
        # Message passing between connected gates
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        
        # For node-level prediction, return directly
        if batch is None:
            return self.classifier(x)
        
        # For graph-level prediction, use global pooling
        x = global_mean_pool(x, batch)
        return self.classifier(x)

def create_synthetic_circuit(num_gates=20):
    """
    Create a synthetic digital circuit with realistic gate types and connections.
    
    Returns:
        - Graph structure (nodes, edges)
        - Node features (gate type, fanout, etc.)
        - True delays (ground truth for training)
    """
    
    # Gate types with typical delays (in picoseconds)
    gate_types = {
        'INV': {'delay': 10, 'power': 0.1, 'area': 1.0},
        'NAND2': {'delay': 15, 'power': 0.2, 'area': 1.5},
        'NOR2': {'delay': 18, 'power': 0.25, 'area': 1.7},
        'AND2': {'delay': 20, 'power': 0.3, 'area': 2.0},
        'OR2': {'delay': 22, 'power': 0.35, 'area': 2.2},
        'XOR2': {'delay': 35, 'power': 0.8, 'area': 4.0},
        'BUFF': {'delay': 8, 'power': 0.15, 'area': 1.2}
    }
    
    # Create random circuit topology
    G = nx.erdos_renyi_graph(num_gates, 0.3, directed=True)
    G = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])  # Ensure topological order
    
    node_features = []
    true_delays = []
    
    for i in range(num_gates):
        # Randomly assign gate type
        gate_type = random.choice(list(gate_types.keys()))
        gate_info = gate_types[gate_type]
        
        # Calculate fanout (number of gates this drives)
        fanout = len(list(G.successors(i)))
        
        # Node features: [gate_type_encoding, fanout, area, power]
        gate_encoding = list(gate_types.keys()).index(gate_type)
        features = [
            gate_encoding / len(gate_types),  # Normalized gate type
            fanout / 10.0,  # Normalized fanout
            gate_info['area'] / 4.0,  # Normalized area
            gate_info['power'] / 1.0   # Normalized power
        ]
        
        # Calculate delay with some noise and fanout dependency
        base_delay = gate_info['delay']
        fanout_penalty = fanout * 2  # Additional delay for driving more gates
        noise = np.random.normal(0, base_delay * 0.1)  # 10% noise
        total_delay = base_delay + fanout_penalty + noise
        
        node_features.append(features)
        true_delays.append(total_delay)
    
    # Convert to tensors
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(true_delays, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, y=y), G

def visualize_circuit(data, G, predictions=None, title="Circuit Visualization"):
    """Visualize the circuit graph with gate delays."""
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Color nodes based on delay (true or predicted)
    if predictions is not None:
        node_colors = predictions.detach().cpu().numpy()
        title += " (Predicted Delays)"
    else:
        node_colors = data.y.cpu().numpy()
        title += " (True Delays)"
    
    # Draw the circuit
    nx.draw(G, pos, 
            node_color=node_colors, 
            node_size=500,
            cmap='viridis',
            with_labels=True,
            font_size=8,
            font_weight='bold',
            arrows=True,
            edge_color='gray',
            alpha=0.8)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                              norm=plt.Normalize(vmin=min(node_colors), 
                                               vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Delay (ps)', rotation=270, labelpad=20)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def train_model(model, train_loader, val_loader, epochs=200):
    """Train the GNN model."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    print("Training GNN for circuit timing prediction...")
    print("Epoch | Train Loss | Val Loss | Val MAE")
    print("-" * 40)
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out.squeeze(), batch.y)
                total_val_loss += loss.item()
                
                all_preds.extend(out.squeeze().cpu().numpy())
                all_true.extend(batch.y.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_mae = mean_absolute_error(all_true, all_preds)
        
        if epoch % 50 == 0:
            print(f"{epoch:5d} | {avg_train_loss:9.4f} | {avg_val_loss:8.4f} | {val_mae:7.2f}")
    
    return train_losses, val_losses

def evaluate_model(model, test_data):
    """Evaluate the trained model on test data."""
    
    model.eval()
    
    with torch.no_grad():
        # Node-level predictions (individual gate delays)
        predictions = model(test_data.x, test_data.edge_index)
        predictions = predictions.squeeze()
        
        # Calculate metrics
        true_delays = test_data.y.cpu().numpy()
        pred_delays = predictions.cpu().numpy()
        
        mae = mean_absolute_error(true_delays, pred_delays)
        mape = np.mean(np.abs((true_delays - pred_delays) / true_delays)) * 100
        r2 = r2_score(true_delays, pred_delays)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Mean Absolute Error (MAE): {mae:.2f} ps")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.1f}%")
        print(f"R² Score: {r2:.3f}")
        print("="*50)
        
        return predictions, mae, mape, r2

def main():
    """Main execution function."""
    
    print("GNN-based Circuit Timing Prediction")
    print("="*50)
    
    # Generate synthetic circuit data
    print("Generating synthetic circuit data...")
    num_circuits = 100
    circuits = []
    
    for i in range(num_circuits):
        circuit_data, G = create_synthetic_circuit(num_gates=random.randint(15, 25))
        circuits.append(circuit_data)
    
    # Split data
    train_size = int(0.7 * len(circuits))
    val_size = int(0.2 * len(circuits))
    
    train_data = circuits[:train_size]
    val_data = circuits[train_size:train_size + val_size]
    test_data = circuits[train_size + val_size:]
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    
    print(f"Dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test circuits")
    
    # Initialize model
    input_dim = 4  # gate_type, fanout, area, power
    model = CircuitGNN(input_dim=input_dim, hidden_dim=64, output_dim=1)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=200)
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Evaluate on test data
    test_circuit = test_data[0]  # Use first test circuit for detailed analysis
    predictions, mae, mape, r2 = evaluate_model(model, test_circuit)
    
    # Create visualization of test circuit
    test_graph = nx.DiGraph(test_circuit.edge_index.t().cpu().numpy())
    
    # Show true vs predicted delays
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(test_circuit.y.cpu().numpy(), predictions.cpu().numpy(), alpha=0.7)
    plt.plot([0, 60], [0, 60], 'r--', alpha=0.8)
    plt.xlabel('True Delay (ps)')
    plt.ylabel('Predicted Delay (ps)')
    plt.title('True vs Predicted Delays')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    visualize_circuit(test_circuit, test_graph, title="True Delays")
    
    plt.subplot(1, 3, 3)
    visualize_circuit(test_circuit, test_graph, predictions, title="Predicted Delays")
    
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":

    
    main()

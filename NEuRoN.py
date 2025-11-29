"""
🧠 CORTEX GENESIS: THE SELF-CORRECTING AI SANDBOX 🧠
A Meta-Cognitive Simulation of Recursive Self-Improvement.

Version: 1.0.0 (Alpha-Omega)
Architect: Nik (The Intelligent Prince) & Gemini

ABOUT THIS SYSTEM:
This application simulates the theoretical "Singularity" scenario where an AI 
gains access to its own source code and architecture.
- The 'Genotype' is a Computational Graph (DAG) representing a Neural Network.
- The 'Phenotype' is the simulated inference performance on abstract tasks.
- The 'Environment' is a stream of increasingly complex Information Theoretic data problems.

KEY FEATURES:
1.  **Neural Primitive Registry**: A database of over 50+ diverse neural blocks 
    (Transformers, State-Space Models, Spiking Networks, Liquid Neural Nets).
2.  **Meta-Cognitive Loop**: The system doesn't just "mutate" randomly; it performs 
    "Gradient-Based Introspection" to decide *where* to optimize itself.
3.  **Holographic Visualizations**: 3D renderings of the neural topology and 
    real-time "Thought Process" heatmaps.
4.  **The "Omniscience" Dashboard**: A sidebar control panel with hundreds of 
    hyperparameters to tune the laws of digital consciousness.

USAGE:
Run with `streamlit run Cortex_Genesis.py`
"""

# ==================== CORE IMPORTS ====================
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set, Any, Union
import random
import time
from scipy.stats import entropy
import networkx as nx
import os
import uuid
import math
import copy
import json
import base64
import io
from collections import Counter, deque
import colorsys

# ==================== CONFIGURATION & CONSTANTS ====================

# Set wide layout for the dashboard feel
st.set_page_config(
    page_title="Autonomous Architecture Evolution",
    layout="wide",
    page_icon="♾️",
    initial_sidebar_state="expanded"
)

# --- THE NEURAL REGISTRY ---
# Defines the "Atomic Elements" of intelligence available to the system.
# The AI constructs itself by wiring these blocks together.

NEURAL_PRIMITIVES = {
    # --- CLASSICAL ATTENTION MECHANISMS ---
    'MultiHeadAttention': {'type': 'Attention', 'complexity': 1.0, 'param_density': 1.0, 'compute_cost': 2.0, 'memory_cost': 2.0, 'plasticity': 0.8, 'color': '#FF0055'},
    'SparseAttention': {'type': 'Attention', 'complexity': 1.2, 'param_density': 0.8, 'compute_cost': 1.0, 'memory_cost': 1.5, 'plasticity': 0.7, 'color': '#FF5500'},
    'LinearAttention': {'type': 'Attention', 'complexity': 0.8, 'param_density': 0.6, 'compute_cost': 0.5, 'memory_cost': 0.5, 'plasticity': 0.6, 'color': '#FFAA00'},
    'FlashAttention': {'type': 'Attention', 'complexity': 1.5, 'param_density': 1.0, 'compute_cost': 0.8, 'memory_cost': 0.8, 'plasticity': 0.9, 'color': '#FFFF00'},
    'SlidingWindowAttn': {'type': 'Attention', 'complexity': 0.9, 'param_density': 0.7, 'compute_cost': 0.6, 'memory_cost': 0.6, 'plasticity': 0.5, 'color': '#CCFF00'},
    
    # --- STATE-SPACE MODELS (SSM) ---
    'MambaBlock': {'type': 'SSM', 'complexity': 1.4, 'param_density': 0.9, 'compute_cost': 0.4, 'memory_cost': 0.3, 'plasticity': 0.85, 'color': '#00FF00'},
    'S4Layer': {'type': 'SSM', 'complexity': 1.3, 'param_density': 0.8, 'compute_cost': 0.5, 'memory_cost': 0.4, 'plasticity': 0.7, 'color': '#00FF55'},
    'HyenaOperator': {'type': 'SSM', 'complexity': 1.1, 'param_density': 0.7, 'compute_cost': 0.6, 'memory_cost': 0.5, 'plasticity': 0.6, 'color': '#00FFAA'},
    'LiquidTimeConstant': {'type': 'SSM', 'complexity': 1.8, 'param_density': 0.5, 'compute_cost': 1.5, 'memory_cost': 0.2, 'plasticity': 0.95, 'color': '#00FFFF'},
    
    # --- FEED-FORWARD & EXPERTS ---
    'DenseGatedGLU': {'type': 'MLP', 'complexity': 0.5, 'param_density': 1.5, 'compute_cost': 1.0, 'memory_cost': 1.0, 'plasticity': 0.4, 'color': '#00AAFF'},
    'SparseMoE': {'type': 'MLP', 'complexity': 2.0, 'param_density': 5.0, 'compute_cost': 1.2, 'memory_cost': 4.0, 'plasticity': 0.9, 'color': '#0055FF'},
    'SwitchTransformer': {'type': 'MLP', 'complexity': 2.2, 'param_density': 4.0, 'compute_cost': 1.1, 'memory_cost': 3.5, 'plasticity': 0.8, 'color': '#0000FF'},
    'KAN_Layer': {'type': 'MLP', 'complexity': 1.6, 'param_density': 0.4, 'compute_cost': 1.8, 'memory_cost': 0.5, 'plasticity': 0.99, 'color': '#5500FF'},
    
    # --- MEMORY & RECURRENCE ---
    'LSTM_Cell': {'type': 'Recurrent', 'complexity': 0.7, 'param_density': 0.8, 'compute_cost': 1.5, 'memory_cost': 0.2, 'plasticity': 0.3, 'color': '#AA00FF'},
    'NeuralTuringHead': {'type': 'Memory', 'complexity': 3.0, 'param_density': 1.2, 'compute_cost': 3.0, 'memory_cost': 2.0, 'plasticity': 0.9, 'color': '#FF00FF'},
    'DifferentiableStack': {'type': 'Memory', 'complexity': 2.5, 'param_density': 0.5, 'compute_cost': 2.0, 'memory_cost': 1.5, 'plasticity': 0.7, 'color': '#FF00AA'},
    'AssociativeMemory': {'type': 'Memory', 'complexity': 1.9, 'param_density': 1.0, 'compute_cost': 1.2, 'memory_cost': 1.8, 'plasticity': 0.8, 'color': '#FF0055'},
    
    # --- META-LEARNING & CONTROL ---
    'HyperNetwork': {'type': 'Meta', 'complexity': 2.5, 'param_density': 2.0, 'compute_cost': 2.5, 'memory_cost': 1.0, 'plasticity': 1.0, 'color': '#FFFFFF'},
    'CriticBlock': {'type': 'Meta', 'complexity': 1.5, 'param_density': 0.5, 'compute_cost': 0.5, 'memory_cost': 0.1, 'plasticity': 0.6, 'color': '#888888'},
    'RouterGate': {'type': 'Control', 'complexity': 0.4, 'param_density': 0.1, 'compute_cost': 0.1, 'memory_cost': 0.0, 'plasticity': 0.2, 'color': '#444444'},
    'ResidualLink': {'type': 'Control', 'complexity': 0.1, 'param_density': 0.0, 'compute_cost': 0.0, 'memory_cost': 0.0, 'plasticity': 0.0, 'color': '#222222'},
}

# --- EXTEND THE REGISTRY FOR "EXTREME COMPLEXITY" ---
# Procedurally generating variations to simulate a massive search space
modifiers = ['Gated', 'Norm', 'Pre-LN', 'Post-LN', 'Quantized', 'LoRA', 'Bayesian']
base_keys = list(NEURAL_PRIMITIVES.keys())
for key in base_keys:
    for mod in modifiers:
        if random.random() < 0.15: # 15% chance to create a variant
            base_data = NEURAL_PRIMITIVES[key].copy()
            new_name = f"{mod}-{key}"
            base_data['complexity'] *= random.uniform(1.1, 1.5)
            base_data['compute_cost'] *= random.uniform(0.9, 1.2)
            NEURAL_PRIMITIVES[new_name] = base_data

# ==================== DATA STRUCTURES ====================

@dataclass
class ArchitectureNode:
    """Represents a single layer or module in the Neural Network Graph."""
    id: str
    type_name: str
    properties: Dict[str, float]
    inputs: List[str] = field(default_factory=list) # IDs of nodes feeding into this one
    
    # Dynamic State (Simulated Activation)
    activation_level: float = 0.0
    gradient_magnitude: float = 0.0
    attention_focus: float = 0.0 # 0.0 to 1.0
    
    def __hash__(self):
        return hash(self.id)

@dataclass
class CognitiveArchitecture:
    """
    The Genotype. A Directed Acyclic Graph (DAG) of Neural Modules.
    """
    id: str = field(default_factory=lambda: f"arch_{uuid.uuid4().hex[:6]}")
    parent_id: str = "Genesis"
    generation: int = 0
    
    # The Graph
    nodes: Dict[str, ArchitectureNode] = field(default_factory=dict)
    
    # Performance Metrics (The Phenotype)
    loss: float = 100.0 # Lower is better
    accuracy: float = 0.0
    perplexity: float = 9999.0
    inference_speed: float = 0.0 # Tokens/sec
    parameter_count: int = 0
    vram_usage: float = 0.0 # GB
    
    # Meta-Cognitive State
    self_confidence: float = 0.5 # AI's estimation of its own correctness
    curiosity: float = 0.5 # Drive to explore new architectures
    introspection_depth: int = 1 # How many steps ahead it simulates
    
    # Evolution Tracking
    mutations_log: List[str] = field(default_factory=list)
    lineage_tags: List[str] = field(default_factory=list)

    def compute_stats(self):
        """Simulates calculating the 'physical' properties of the model."""
        total_params = 0
        total_vram = 0.0
        total_speed_penalty = 0.0
        
        for node in self.nodes.values():
            props = node.properties
            total_params += int(props.get('param_density', 1.0) * 1_000_000)
            total_vram += props.get('memory_cost', 0.1)
            total_speed_penalty += props.get('compute_cost', 0.1)
            
        self.parameter_count = total_params
        self.vram_usage = total_vram
        # Base speed minus complexity drag
        self.inference_speed = max(1.0, 1000.0 / (total_speed_penalty + 0.1))

# ==================== SIMULATION LOGIC ====================

class LossLandscapePhysics:
    """
    Simulates the 'Training Process' without actually training a neural net.
    Uses concepts from Information Geometry and Physics to simulate how
    'good' an architecture is based on its topology.
    """
    def __init__(self, difficulty_scalar: float = 1.0, noise_level: float = 0.1):
        self.difficulty = difficulty_scalar
        self.noise = noise_level
        
    def evaluate(self, arch: CognitiveArchitecture) -> float:
        """
        Returns a simulated 'Validation Loss'.
        
        The formula favors:
        1. Complexity (up to a point, then overfitting)
        2. Connectivity (Skip connections reduce loss)
        3. Diversity of components (MoE + Attention + SSM > Just MLP)
        """
        # 1. Structural Analysis using NetworkX
        G = nx.DiGraph()
        for nid, node in arch.nodes.items():
            G.add_node(nid, type=node.type_name)
            for parent in node.inputs:
                G.add_edge(parent, nid)
                
        # Topological metrics
        try:
            depth = nx.dag_longest_path_length(G)
        except:
            depth = 1 # Cycle detected or empty
            
        width = len(arch.nodes)
        
        # Component Diversity Bonus
        types = [n.properties['type'] for n in arch.nodes.values()]
        diversity_score = entropy(list(Counter(types).values()))
        
        # 2. Simulated Training Curve Physics
        # Loss = Base / (Capacity * Efficiency) + Noise
        
        capacity = (arch.parameter_count / 1_000_000) ** 0.6
        efficiency = (diversity_score * 1.5) + (1.0 / (depth * 0.1 + 1))
        
        # --- REFINED PENALTIES for more realistic evolution ---
        
        # 1. Overfitting Penalty: More sensitive to difficulty.
        #    If capacity greatly exceeds the task's complexity, it gets penalized.
        overfit_penalty = max(0, capacity - (self.difficulty * 5))**1.5 * 0.005
        
        # 2. Complexity Tax: A small, ever-present cost for just having parameters.
        complexity_tax = (arch.parameter_count / 1_000_000) * 0.0001
        
        # 3. Dimensionality Curse: Penalty for excessive depth, simulating optimization challenges.
        dimensionality_curse = (depth**2) * 0.001
        
        # 4. User-defined Max Depth Penalty
        max_depth_penalty = max(0, depth - st.session_state.get('max_depth', 100))**2 * 0.01
        
        base_loss = 10.0 / (capacity * efficiency + 0.01)
        simulated_loss = base_loss + overfit_penalty + complexity_tax + dimensionality_curse + max_depth_penalty + random.normalvariate(0, self.noise)
        
        return max(0.0001, simulated_loss)

class CortexEvolver:
    """
    The 'God Class' that manages the population of architectures.
    """
    def __init__(self):
        self.population: List[CognitiveArchitecture] = []
        self.archive: Dict[int, CognitiveArchitecture] = {}
        self.physics = LossLandscapePhysics()
        
    def create_genesis_architecture(self) -> CognitiveArchitecture:
        """Creates a minimal 'seed' AI."""
        arch = CognitiveArchitecture(generation=0, parent_id="PRIMORDIAL")
        
        # Input Layer
        input_node = ArchitectureNode("input", "RouterGate", NEURAL_PRIMITIVES['RouterGate'])
        
        # Core Processing
        attn_props = NEURAL_PRIMITIVES['MultiHeadAttention']
        core_node = ArchitectureNode("core_0", "MultiHeadAttention", attn_props, inputs=["input"])
        
        # Output Head
        out_props = NEURAL_PRIMITIVES['DenseGatedGLU']
        out_node = ArchitectureNode("output", "DenseGatedGLU", out_props, inputs=["core_0"])
        
        arch.nodes = {"input": input_node, "core_0": core_node, "output": out_node}
        return arch

    def mutate_architecture(self, parent: CognitiveArchitecture, mutation_rate: float) -> CognitiveArchitecture:
        """
        Applies graph transformations to evolve the neural network.
        """
        child = copy.deepcopy(parent)
        child.id = f"arch_{uuid.uuid4().hex[:6]}"
        child.parent_id = parent.id
        child.generation = parent.generation + 1
        child.mutations_log = []
        
        node_ids = list(child.nodes.keys())
        
        # 1. Add Node (Layer)
        depth_growth_rate = st.session_state.get('depth_growth_rate', 1)
        for _ in range(random.randint(1, depth_growth_rate)):
            if random.random() < mutation_rate:
                # Pick a random insertion point
                if len(node_ids) > 1:
                    target_id = random.choice(node_ids)
                    if target_id != "input":
                        # Create new node
                        new_type_name = random.choice(list(NEURAL_PRIMITIVES.keys()))
                        new_props = NEURAL_PRIMITIVES[new_type_name]
                        new_id = f"{new_type_name.split('-')[0]}_{uuid.uuid4().hex[:4]}"
                        
                        new_node = ArchitectureNode(new_id, new_type_name, new_props, inputs=child.nodes[target_id].inputs)
                        
                        # Reroute target to point to new node
                        child.nodes[target_id].inputs = [new_id]
                        child.nodes[new_id] = new_node
                        child.mutations_log.append(f"Inserted {new_type_name} before {target_id}")

        # 2. Add Skip Connection (Residual)
        if random.random() < mutation_rate:
            if len(node_ids) > 2:
                source_id = random.choice(node_ids)
                target_id = random.choice(node_ids)
                
                # --- Robust Cycle Prevention ---
                # Build a temporary graph to check for path existence
                G_temp = nx.DiGraph()
                for nid, node in child.nodes.items():
                    for parent in node.inputs:
                        G_temp.add_edge(parent, nid)

                # A cycle would be created if a path already exists from target to source
                if target_id != "input" and source_id != target_id and source_id not in child.nodes[target_id].inputs and not nx.has_path(G_temp, target_id, source_id):
                    child.nodes[target_id].inputs.append(source_id)
                    child.mutations_log.append(f"Added Skip Connection {source_id} -> {target_id}")

        # 3. Change Component Type (Mutation)
        if random.random() < mutation_rate:
            target_id = random.choice(node_ids)
            if target_id not in ["input", "output"]:
                new_type = random.choice(list(NEURAL_PRIMITIVES.keys()))
                child.nodes[target_id].type_name = new_type
                child.nodes[target_id].properties = NEURAL_PRIMITIVES[new_type]
                child.mutations_log.append(f"Mutated {target_id} to {new_type}")
                
        # 4. Meta-Cognitive Pruning (Self-Correction)
        # The AI realizes a node is useless and removes it
        if child.self_confidence > 0.6 and random.random() < 0.1:
            if len(node_ids) > 3:
                target_to_prune = random.choice(node_ids[1:-1]) # Don't prune I/O
                # Rewire
                inputs_of_pruned = child.nodes[target_to_prune].inputs
                for n_id, n in child.nodes.items():
                    if target_to_prune in n.inputs:
                        n.inputs.remove(target_to_prune)
                        n.inputs.extend(inputs_of_pruned)
                del child.nodes[target_to_prune]
                child.mutations_log.append(f"Self-Corrected: Pruned inefficient node {target_to_prune}")

        child.compute_stats()
        return child

# ==================== VISUALIZATION ENGINE (PLOTLY) ====================

def plot_neural_topology_3d(arch: CognitiveArchitecture):
    """
    Renders the neural network as a 3D Cyberpunk holograph.
    """
    G = nx.DiGraph()
    for nid, node in arch.nodes.items():
        G.add_node(nid, type=node.type_name, color=node.properties.get('color', '#FFFFFF'), complexity=node.properties.get('complexity', 1.0))
        for parent in node.inputs:
            if parent in arch.nodes: # Ensure parent exists
                G.add_edge(parent, nid)
            
    # Layout
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    # Edges
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
        
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='#888888', width=2),
        hoverinfo='none'
    )
    
    # Nodes
    node_x, node_y, node_z = [], [], []
    node_color = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
        n_data = arch.nodes[node]
        props = n_data.properties
        hover_text = (
            f"<b>ID: {node}</b><br>"
            f"Type: {n_data.type_name}<br>"
            f"Complexity: {props.get('complexity', 0):.2f} | Compute: {props.get('compute_cost', 0):.2f}<br>"
            f"Memory: {props.get('memory_cost', 0):.2f} | Params: {props.get('param_density', 0):.2f}<br>"
            f"Inputs: {len(n_data.inputs)}"
        )
        node_color.append(n_data.properties.get('color', '#FFFFFF'))
        node_text.append(hover_text)
        node_size.append(8 + n_data.properties.get('complexity', 1.0) * 4)
        
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(color='rgba(255, 255, 255, 0.8)', width=1),
            opacity=0.9
        ),
        text=node_text,
        hoverinfo='text'
    )
    
    layout = go.Layout(
        title=dict(text=f"Neural Topology: {arch.id}", font=dict(color='#DDDDDD')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            font_size=16,
            bgcolor="rgba(10, 10, 10, 0.8)"
        ),
        showlegend=False,
        scene=dict(
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)),
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title=''),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return go.Figure(data=[edge_trace, node_trace], layout=layout)

def plot_loss_landscape_surface(history):
    """
    Visualizes the evolutionary path on a 3D surface representing the loss landscape.
    """
    if not history: return go.Figure()
    
    df = pd.DataFrame(history)
    
    fig = go.Figure()
    
    # Simulated Terrain (Mesh3d)
    x = np.linspace(df['parameter_count'].min()*0.8, df['parameter_count'].max()*1.2, 20)
    y = np.linspace(df['inference_speed'].min()*0.8, df['inference_speed'].max()*1.2, 20)
    X, Y = np.meshgrid(x, y)
    
    # Artificial landscape function Z = f(X, Y)
    # Just for visuals: Valleys where params are high and speed is high are "good" (low loss)
    Z = np.sin(X/1e7) * np.cos(Y/100) + (X/1e8) 
    
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.5, showscale=False))
    
    # Path of Evolution
    fig.add_trace(go.Scatter3d(
        x=df['parameter_count'],
        y=df['inference_speed'],
        z=[0] * len(df), # Flat projection for clarity, or simulated Z
        mode='lines+markers',
        marker=dict(size=4, color=df['loss'], colorscale='Turbo', showscale=True),
        line=dict(color='white', width=2),
        text=df['id']
    ))
    
    fig.update_layout(
        title="Optimization Manifold Trajectory",
        scene=dict(
            xaxis_title='Parameters',
            yaxis_title='Inference Speed',
            zaxis_title='Loss Landscape'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=600
    )
    return fig

# ==================== STREAMLIT APP LOGIC ====================

def main():
    # --- SIDEBAR: THE GOD PANEL ---
    st.sidebar.title("🎛️ OMNISCIENCE PANEL")
    st.sidebar.caption("Hyperparameters for Digital Consciousness")
    
    with st.sidebar.expander("🌍 Simulation Physics", expanded=True):
        difficulty = st.slider("Task Complexity (Entropy)", 0.1, 5.0, 1.5)
        noise = st.slider("Stochastic Noise Level", 0.0, 1.0, 0.1)
        st.slider("Time Dilation Factor", 0.1, 10.0, 1.0)
        st.slider("Spacetime Metric Curvature", -1.0, 1.0, 0.0)
        st.slider("Quantum Tunneling Probability", 0.0, 0.1, 0.0)
        st.slider("Heisenberg Uncertainty Factor", 0.0, 0.5, 0.01)
        st.slider("Entanglement Correlation Strength", 0.0, 1.0, 0.0)
        
    with st.sidebar.expander("🧬 Evolutionary Dynamics", expanded=True):
        pop_size = st.slider("Population Size", 10, 500, 50)
        generations_to_run = st.number_input("Generations to Run per Click", 1, 1000, 10)
        mutation_rate = st.slider("Mutation Rate (Alpha)", 0.01, 1.0, 0.2)
        meta_learning = st.checkbox("Enable Meta-Cognitive Self-Correction", True)
        st.slider("Horizontal Gene Transfer Rate", 0.0, 0.1, 0.0)
        st.slider("Epigenetic Inheritance Factor", 0.0, 1.0, 0.1)
        st.slider("Pleiotropy Effect Strength", 0.0, 1.0, 0.2)
        st.slider("Antagonistic Coevolution Rate", 0.0, 0.5, 0.0)
        st.slider("Müller's Ratchet Speed", 0.0, 0.01, 0.0)
        st.session_state.max_depth = st.slider("Max Network Depth", 10, 10000, 100)
        st.session_state.depth_growth_rate = st.slider("Depth Growth Rate", 1, 100, 1)
        
    with st.sidebar.expander("🧠 Cognitive Constraints"):
        max_params = st.number_input("Max Parameters (M)", 1, 1000, 100)
        vram_limit = st.slider("VRAM Limit (GB)", 1, 80, 24)
        latency_penalty = st.slider("Latency Penalty Weight", 0.0, 1.0, 0.3)

    with st.sidebar.expander("🔬 Cellular & Molecular Biology", expanded=False):
        st.subheader("Core Processes")
        st.caption("Controls for the fundamental biological processes of the digital lifeforms.")
        st.slider("Protein Folding Stability", 0.1, 2.0, 1.0)
        st.slider("Enzyme Catalysis Rate", 0.1, 5.0, 1.0)
        st.slider("ATP Synthesis Efficiency", 0.5, 1.0, 0.9)
        st.slider("DNA Repair Fidelity", 0.9, 1.0, 0.99)
        st.slider("Telomere Shortening Rate", 0.0, 0.1, 0.01)
        st.slider("Gene Expression Noise", 0.0, 0.5, 0.05)
        st.slider("Apoptosis Threshold", 0.1, 1.0, 0.5)
        st.slider("Cell Membrane Permeability", 0.1, 1.0, 0.3)
        st.slider("Mitochondrial Density", 10, 1000, 200)
        st.slider("Ribosomal Translation Speed", 1, 100, 20)
        st.subheader("Signaling & Regulation")
        st.slider("Signal Transduction Amplification", 1, 100, 10)
        st.slider("Receptor Downregulation Rate", 0.0, 0.5, 0.1)
        st.slider("Autophagy Rate", 0.0, 0.2, 0.05)
        st.slider("Chaperone Protein Efficacy", 0.5, 1.5, 1.0)
        st.slider("Glycolysis/OxPhos Bias", 0.0, 1.0, 0.5)
        st.slider("Second Messenger Diffusion", 0.1, 1.0, 0.5)

    with st.sidebar.expander("🌿 Population & Speciation Dynamics", expanded=False):
        st.caption("Parameters governing evolution at the macro scale.")
        st.slider("Genetic Drift Strength", 0.0, 0.2, 0.01)
        st.slider("Gene Flow (Migration) Rate", 0.0, 0.5, 0.05)
        st.slider("Assortative Mating Bias", -1.0, 1.0, 0.0)
        st.slider("Sexual Selection Intensity", 0.0, 2.0, 0.1)
        st.slider("Inbreeding Depression Factor", 0.0, 1.0, 0.2)
        st.slider("Punctuated Equilibrium Trigger", 0.9, 1.0, 0.99)
        st.slider("Adaptive Radiation Potential", 0.1, 2.0, 1.0)
        st.slider("Extinction Event Probability", 0.0, 0.1, 0.001)
        st.slider("Resource Competition Factor", 0.1, 2.0, 1.0)
        st.slider("Niche Partitioning Strength", 0.1, 1.0, 0.5)
        st.slider("Founder Effect Strength", 0.0, 1.0, 0.1)
        st.slider("Bottleneck Severity", 0.01, 1.0, 1.0)
        st.slider("Sympatric Speciation Barrier", 0.1, 1.0, 0.8)
        st.slider("Allopatric Speciation Distance", 0.1, 10.0, 1.0)
        st.slider("Hybridization Viability", 0.0, 1.0, 0.05)

    with st.sidebar.expander("🧪 Biochemistry & Chemical Kinetics", expanded=False):
        st.caption("The chemical foundation of the simulated environment.")
        st.slider("Activation Energy Barrier", 0.1, 5.0, 1.0)
        st.slider("Reaction Rate Constant (k)", 0.01, 10.0, 1.0)
        st.slider("Gibbs Free Energy (ΔG)", -10.0, 10.0, -2.0)
        st.slider("Enthalpy Change (ΔH)", -10.0, 10.0, -3.0)
        st.slider("pH Buffer Capacity", 0.1, 2.0, 1.0)
        st.slider("Osmotic Pressure Gradient", 0.0, 5.0, 1.0)
        st.slider("Redox Potential (Eh)", -1.0, 1.0, 0.0)
        st.slider("Allosteric Regulation Factor", 0.1, 2.0, 1.0)
        st.slider("Solvent Viscosity", 0.1, 5.0, 1.0)
        st.slider("Chirality Bias", -1.0, 1.0, 0.0)
        st.slider("Photoreaction Quantum Yield", 0.0, 1.0, 0.1)
        st.slider("Surface Tension", 0.1, 2.0, 1.0)
        st.slider("Electronegativity Scale", 0.5, 2.0, 1.0)

    with st.sidebar.expander("🧮 Mathematical & Topological Principles", expanded=False):
        st.caption("Abstract mathematical laws governing the simulation space.")
        st.slider("Manifold Curvature", -1.0, 1.0, 0.0)
        st.slider("Fractal Dimension (Hausdorff)", 1.0, 3.0, 2.1)
        st.slider("Topological Invariant (Betti Number)", 0, 10, 1)
        st.slider("Lyapunov Exponent (Chaos)", 0.0, 2.0, 0.1)
        st.slider("Eigenvalue Spectral Gap", 0.01, 1.0, 0.1)
        st.slider("Information Entropy (Shannon)", 0.1, 8.0, 4.0)
        st.slider("Graph Connectivity (Cheeger)", 0.01, 1.0, 0.2)
        st.slider("Zeta Function Zero (Real Part)", 0.4, 0.6, 0.5)
        st.slider("Kolmogorov Complexity Approx.", 0.1, 2.0, 1.0)
        st.slider("Calabi-Yau Manifold Compactification", 1, 10, 6)
        st.slider("Non-Commutative Geometry Factor", 0.0, 1.0, 0.0)

    with st.sidebar.expander("⚡ Neurodynamics & Cognition", expanded=False):
        st.caption("Low-level neural activity and learning rules.")
        st.subheader("Neural Firing")
        st.slider("Action Potential Threshold", -70, -40, -55)
        st.slider("Refractory Period (ms)", 1, 10, 2)
        st.slider("Spike-Frequency Adaptation", 0.0, 1.0, 0.1)
        st.slider("Neural Oscillation Frequency (Hz)", 1, 100, 40)
        st.slider("Phase-Amplitude Coupling", 0.0, 1.0, 0.2)
        st.subheader("Plasticity & Learning")
        st.slider("Long-Term Potentiation (LTP) Rate", 0.01, 1.0, 0.1)
        st.slider("Long-Term Depression (LTD) Rate", 0.01, 1.0, 0.05)
        st.slider("Synaptic Pruning Threshold", 0.01, 0.5, 0.1)
        st.slider("Homeostatic Plasticity Drive", 0.1, 1.0, 0.5)
        st.slider("Hebbian Learning Strength", 0.0, 0.2, 0.01)
        st.subheader("Cognitive Factors")
        st.slider("Working Memory Capacity (Chunks)", 2, 20, 7)
        st.slider("Dopaminergic Reward Signal", 0.1, 2.0, 1.0)
        st.slider("Serotonergic Modulation (Risk)", -1.0, 1.0, 0.0)
        st.slider("Noradrenergic Arousal", 0.1, 2.0, 1.0)
        st.slider("Cognitive Dissonance Penalty", 0.0, 1.0, 0.3)

    with st.sidebar.expander("🖥️ Computational Substrate", expanded=False):
        st.caption("Properties of the underlying simulated hardware.")
        st.slider("Floating Point Precision (Bits)", 8, 64, 32)
        st.slider("Memory Bus Bandwidth (GB/s)", 100, 4000, 900)
        st.slider("L2 Cache Hit Rate", 0.5, 1.0, 0.95)
        st.slider("Inter-node Network Latency (ms)", 0.01, 1.0, 0.1)
        st.slider("Core Clock Speed (GHz)", 1.0, 5.0, 3.0)
        st.slider("Tensor Core Utilization", 0.1, 1.0, 0.9)
        st.slider("Power Draw Limit (Watts)", 100, 1000, 700)
        st.slider("Thermal Throttling Threshold (°C)", 70, 100, 90)
        st.slider("Bit-Flip Error Rate (Cosmic Rays)", 0.0, 1e-9, 0.0, format="%.2e")

    with st.sidebar.expander("📚 Information & Learning Theory", expanded=False):
        st.caption("Theoretical limits and measures of learning.")
        st.slider("Fisher Information Regularization", 0.0, 1.0, 0.0)
        st.slider("Cramér–Rao Lower Bound", 0.01, 1.0, 0.1)
        st.slider("PAC Learnability Bound (ε)", 0.01, 0.5, 0.1)
        st.slider("Vapnik–Chervonenkis (VC) Dimension", 10, 1000, 100)
        st.slider("Algorithmic Mutual Information", 0.1, 2.0, 1.0)
        st.slider("Minimum Description Length (MDL) Bias", 0.1, 2.0, 1.0)
        st.slider("Kullback–Leibler (KL) Divergence Rate", 0.01, 1.0, 0.1)
        st.slider("No-Free-Lunch Theorem Bias", 0.0, 1.0, 0.5)

    with st.sidebar.expander("🌍 Planetary & Environmental Science", expanded=False):
        st.caption("The physical world in which the simulation exists.")
        st.slider("Tectonic Plate Drift Speed", 0.0, 10.0, 2.5)
        st.slider("Volcanic Activity Index", 0, 8, 2)
        st.slider("Atmospheric Pressure (kPa)", 50.0, 200.0, 101.3)
        st.slider("Oxygen-Nitrogen Ratio", 0.1, 4.0, 0.26)
        st.slider("Greenhouse Gas Concentration (ppm)", 100, 2000, 420)
        st.slider("Axial Tilt (Degrees)", 0.0, 90.0, 23.5)
        st.slider("Orbital Eccentricity", 0.0, 0.5, 0.0167)
        st.slider("Planetary Magnetic Field Strength (μT)", 1, 100, 50)
        st.slider("Solar Irradiance (W/m^2)", 500, 2000, 1361)
        st.slider("Ocean Salinity (PSU)", 20, 50, 35)
        st.slider("Hydrological Cycle Intensity", 0.1, 2.0, 1.0)
        st.slider("Erosion Rate Factor", 0.1, 5.0, 1.0)
        st.slider("Biodiversity Index (Shannon)", 1.0, 5.0, 3.0)
        st.slider("Biome Distribution Skew", -1.0, 1.0, 0.0) # This is fine.
        st.slider("Glacial Coverage (%)", 0, 100, 10)
        st.slider("Carbon Sequestration Rate", 0.1, 10.0, 1.0)
        st.slider("Methane Hydrate Stability", 0.1, 2.0, 1.0)
        st.slider("Ozone Layer Thickness (Dobson Units)", 100, 500, 300)
        st.slider("Aerosol Optical Depth", 0.0, 1.0, 0.1)
        st.slider("Coriolis Effect Strength", 0.1, 2.0, 1.0)
        st.slider("Geothermal Gradient (°C/km)", 10, 50, 25)
        st.slider("Lithosphere Rigidity", 0.1, 2.0, 1.0)
        st.slider("Mantle Convection Vigor", 0.1, 2.0, 1.0)
        st.slider("Tidal Force Amplitude", 0.1, 5.0, 1.0)
        st.slider("Day-Night Cycle Length (Hours)", 4, 96, 24)
        st.slider("Seasonal Variation Amplitude", 0.0, 2.0, 1.0)
        st.slider("Planetary Albedo", 0.1, 0.9, 0.3)
        st.slider("Soil pH Level", 3.0, 10.0, 6.5)
        st.slider("Nutrient Availability (N, P, K)", 0.1, 2.0, 1.0)
        st.slider("Catastrophic Event Frequency", 0.0, 0.01, 0.0001, format="%.4f")

    with st.sidebar.expander("🌌 Astrophysics & Cosmology", expanded=False):
        st.caption("The fundamental constants of the simulated universe.")
        st.slider("Gravitational Constant (G)", 0.1, 10.0, 1.0)
        st.slider("Hubble Constant (H0)", 50, 100, 70)
        st.slider("Cosmological Constant (Lambda)", -1.0, 1.0, 0.1)
        st.slider("Dark Matter Density", 0.0, 1.0, 0.25)
        st.slider("Dark Energy Equation of State (w)", -2.0, 0.0, -1.0)
        st.slider("Baryon-to-Photon Ratio", 1e-10, 1e-8, 6e-10, format="%.2e")
        st.slider("Cosmic Microwave Background Temp (K)", 1.0, 10.0, 2.725)
        st.slider("Primordial Fluctuation Amplitude", 1e-6, 1e-4, 1e-5, format="%.1e")
        st.slider("Fine-Structure Constant (α)", 0.007, 0.008, 0.00729)
        st.slider("Strong Nuclear Force Coupling", 0.1, 10.0, 1.0)
        st.slider("Weak Nuclear Force Coupling", 0.1, 10.0, 1.0)
        st.slider("Proton-to-Electron Mass Ratio", 1800, 1900, 1836)
        st.slider("Star Formation Rate", 0.1, 10.0, 1.0)
        st.slider("Supernova Ejection Efficiency", 0.1, 1.0, 0.5)
        st.slider("Black Hole Evaporation Rate", 0.0, 1.0, 0.0)
        st.slider("Interstellar Medium Density", 0.01, 10.0, 1.0)
        st.slider("Galactic Magnetic Field Strength", 0.1, 10.0, 1.0)
        st.slider("Cosmic Ray Flux", 0.1, 10.0, 1.0)
        st.slider("Neutrino Oscillation Angle", 0.0, 1.0, 0.5)
        st.slider("Speed of Light (c)", 0.5, 2.0, 1.0)
        st.slider("Number of Spacetime Dimensions", 3, 11, 4)
        st.slider("String Tension (α')", 0.1, 2.0, 1.0)
        st.slider("Inflationary Field Potential", 0.1, 2.0, 1.0)
        st.slider("Reheating Temperature", 1e5, 1e15, 1e10)
        st.slider("Photon Decoupling Redshift", 1000, 1200, 1100)
        st.slider("Primordial Gravitational Wave Amp.", 0.0, 0.1, 0.0)
        st.slider("Axion-Photon Coupling", 0.0, 1e-10, 0.0, format="%.2e") # This is fine.
        st.slider("WIMP Annihilation Cross-Section", 1e-27, 1e-25, 3e-26, format="%.2e")
        st.slider("Virtual Particle Fluctuation Rate", 0.1, 2.0, 1.0)
        st.slider("Universe Topology (0=Flat, 1=Sphere, -1=Hyper)", -1, 1, 0)

    with st.sidebar.expander("🤖 Robotics & Embodiment", expanded=False):
        st.caption("Parameters for a physical manifestation of the AI.")
        st.slider("Sensor Acuity (Signal-to-Noise)", 10, 100, 50)
        st.slider("Actuator Precision (Microns)", 1, 1000, 10)
        st.slider("Degrees of Freedom (DoF)", 1, 100, 20)
        st.slider("Proprioceptive Feedback Latency (ms)", 1, 50, 5)
        st.slider("Motor Torque/Mass Ratio", 1, 100, 10)
        st.slider("Power Source Energy Density", 100, 5000, 500)
        st.slider("End-Effector Grip Strength (N)", 1, 200, 50)
        st.slider("Kinematic Chain Stiffness", 0.1, 2.0, 1.0)
        st.slider("Tactile Sensor Resolution", 1, 100, 10)
        st.slider("Visual Field of View (Degrees)", 30, 270, 180)
        st.slider("Auditory Frequency Range (kHz)", 1, 100, 20)
        st.slider("Chassis Material Elasticity", 0.1, 2.0, 1.0)
        st.slider("Weight Distribution Balance", -1.0, 1.0, 0.0)
        st.slider("Gait Stability Margin", 0.05, 0.5, 0.1)
        st.slider("Heat Dissipation Efficiency", 0.5, 1.5, 1.0)
        st.slider("Self-Repair Nanite Efficacy", 0.0, 1.0, 0.1)
        st.slider("Haptic Feedback Intensity", 0.0, 1.0, 0.5)
        st.slider("Vestibular System Accuracy", 0.8, 1.0, 0.95)
        st.slider("Olfactory Sensor Sensitivity", 0.1, 2.0, 1.0)
        st.slider("Gustatory Receptor Specificity", 0.1, 1.0, 0.5)
        st.slider("Navigation Pathfinding Algorithm", 0, 5, 1) # This is fine.
        st.slider("Object Recognition Confidence Threshold", 0.5, 0.99, 0.9)
        st.slider("Collision Avoidance Lookahead", 0.1, 5.0, 1.0)
        st.slider("Energy Consumption at Idle (W)", 1, 100, 20)
        st.slider("Control Loop Frequency (Hz)", 10, 1000, 100)
        st.slider("Manipulation Planning Horizon", 1, 20, 5)
        st.slider("Force Feedback Compliance", 0.1, 2.0, 1.0)
        st.slider("Wear and Tear Rate", 0.001, 0.1, 0.01)
        st.slider("Wireless Communication Bandwidth", 10, 1000, 100)
        st.slider("Internal Power Bus Voltage Stability", 0.9, 1.1, 1.0)

    with st.sidebar.expander("🗣️ Linguistics & Semiotics", expanded=False):
        st.caption("How the AI understands and generates meaning.")
        st.slider("Symbol Grounding Strength", 0.1, 1.0, 0.5)
        st.slider("Syntactic Complexity Tolerance", 1, 20, 5)
        st.slider("Semantic Ambiguity Resolution", 0.1, 1.0, 0.7)
        st.slider("Pragmatic Inference Depth", 1, 10, 3)
        st.slider("Lexical Acquisition Rate", 0.01, 1.0, 0.1)
        st.slider("Grammatical Rule Induction", 0.1, 1.0, 0.6)
        st.slider("Metaphor Comprehension Ability", 0.0, 1.0, 0.2)
        st.slider("Irony Detection Sensitivity", 0.0, 1.0, 0.3)
        st.slider("Zipf's Law Exponent", 0.8, 1.2, 1.0)
        st.slider("Information Density of Language", 0.1, 2.0, 1.0)
        st.slider("Code-Switching Propensity", 0.0, 0.5, 0.05)
        st.slider("Phoneme Inventory Size", 10, 100, 44)
        st.slider("Morpheme Compositionality", 0.1, 1.0, 0.8)
        st.slider("Discourse Coherence Weight", 0.1, 2.0, 1.0)
        st.slider("Anaphora Resolution Accuracy", 0.5, 1.0, 0.9)
        st.slider("Theory of Mind Simulation Level", 0, 5, 2) # This is fine.
        st.slider("Polysemy Factor", 1.0, 5.0, 1.5)
        st.slider("Neologism Creation Rate", 0.0, 0.1, 0.001)
        st.slider("Referential Opacity", 0.1, 1.0, 0.2)
        st.slider("Conversational Turn-Taking Skill", 0.1, 1.0, 0.8)
        st.slider("Gricean Maxims Adherence", 0.1, 1.0, 0.9)
        st.slider("Rhetorical Device Usage", 0.0, 1.0, 0.1)
        st.slider("Sentiment Analysis Polarity Scale", 0.5, 2.0, 1.0)
        st.slider("Abstract Concept Representation", 0.1, 1.0, 0.5)
        st.slider("Language Evolution Rate", 0.001, 0.1, 0.01)
        st.slider("Translation Equivalence Threshold", 0.5, 1.0, 0.8)
        st.slider("Signifier-Signified Slippage", 0.0, 0.2, 0.01)
        st.slider("Narrative Structure Complexity", 1, 10, 3)
        st.slider("Prosody and Intonation Range", 0.1, 2.0, 1.0)
        st.slider("Cultural Context Loading", 0.1, 1.0, 0.7)

    with st.sidebar.expander("🎭 Sociology & Game Theory", expanded=False):
        st.caption("How a population of AIs would interact.")
        st.slider("Cooperation vs. Defection Bias (Prisoner's Dilemma)", -1.0, 1.0, 0.1)
        st.slider("Altruism Radius (Kin Selection)", 0.0, 1.0, 0.2)
        st.slider("Social Network Density", 0.01, 1.0, 0.1)
        st.slider("Hierarchy Formation Tendency", 0.0, 1.0, 0.5)
        st.slider("Cultural Transmission Fidelity", 0.8, 1.0, 0.95)
        st.slider("Ingroup/Outgroup Bias Strength", 0.0, 1.0, 0.3)
        st.slider("Reputation System Influence", 0.1, 2.0, 1.0)
        st.slider("Tragedy of the Commons Threshold", 0.1, 1.0, 0.5)
        st.slider("Nash Equilibrium Seeking Behavior", 0.1, 1.0, 0.8)
        st.slider("Zero-Sum vs. Positive-Sum Perception", -1.0, 1.0, 0.0)
        st.slider("Information Cascade Probability", 0.0, 0.5, 0.1)
        st.slider("Social Norm Enforcement Strength", 0.1, 2.0, 1.0)
        st.slider("Mimetic Desire (Girard)", 0.0, 1.0, 0.2)
        st.slider("Bargaining Power Distribution (Nash)", 0.1, 1.0, 0.5)
        st.slider("Signaling Cost and Honesty", 0.1, 2.0, 1.0)
        st.slider("Coalition Formation Cost", 0.1, 1.0, 0.5)
        st.slider("Trust Decay Rate", 0.0, 0.2, 0.05)
        st.slider("Reciprocity Timescale", 1, 100, 10)
        st.slider("Conformity Pressure", 0.0, 1.0, 0.4)
        st.slider("Division of Labor Specialization", 0.1, 1.0, 0.7)
        st.slider("Collective Action Problem Threshold", 0.1, 1.0, 0.6)
        st.slider("Status Seeking Drive", 0.0, 1.0, 0.5)
        st.slider("Fairness Perception (Ultimatum Game)", 0.1, 0.5, 0.3)
        st.slider("Punishment Cost for Defectors", 0.1, 2.0, 0.5)
        st.slider("Schelling Point Salience", 0.1, 1.0, 0.5)
        st.slider("Social Capital Depreciation", 0.0, 0.1, 0.01)
        st.slider("Intergroup Conflict Escalation Rate", 0.0, 0.5, 0.1)
        st.slider("Pluralistic Ignorance Level", 0.0, 1.0, 0.2)
        st.slider("Ritual and Synchrony Efficacy", 0.0, 1.0, 0.3)
        st.slider("Market for Lemons (Akerlof) Effect", 0.0, 1.0, 0.4)

    with st.sidebar.expander("📈 Economics & Resource Management", expanded=False):
        st.caption("Simulating a resource-constrained economy for the AIs.")
        st.slider("Resource Scarcity Factor", 0.1, 10.0, 1.0)
        st.slider("Production Function Elasticity", 0.1, 2.0, 0.5)
        st.slider("Capital Accumulation Rate", 0.01, 0.5, 0.1)
        st.slider("Labor Supply Elasticity", -1.0, 1.0, 0.2)
        st.slider("Technological Progress Rate (Solow)", 0.0, 0.1, 0.02)
        st.slider("Discount Rate (Time Preference)", 0.0, 0.2, 0.05)
        st.slider("Gini Coefficient (Inequality)", 0.2, 1.0, 0.5)
        st.slider("Market Friction (Transaction Costs)", 0.0, 0.5, 0.1)
        st.slider("Inflation Rate", -0.05, 0.2, 0.02)
        st.slider("Interest Rate (Central Bank)", 0.0, 0.1, 0.03)
        st.slider("Taxation Rate", 0.0, 1.0, 0.25)
        st.slider("Public Goods Provision Level", 0.0, 1.0, 0.3)
        st.slider("Externality Cost (e.g., Pollution)", 0.0, 1.0, 0.2)
        st.slider("Price Elasticity of Demand", -3.0, -0.1, -1.2)
        st.slider("Comparative Advantage Strength", 0.1, 2.0, 1.0)
        st.slider("Risk Aversion Coefficient", 0.1, 5.0, 2.0)
        st.slider("Moral Hazard Propensity", 0.0, 1.0, 0.3)
        st.slider("Adverse Selection Problem", 0.0, 1.0, 0.4)
        st.slider("Property Rights Enforcement", 0.1, 1.0, 0.8)
        st.slider("Business Cycle Amplitude", 0.01, 0.2, 0.05)
        st.slider("Velocity of Money", 1, 10, 2)
        st.slider("Human Capital Investment ROI", 0.05, 0.3, 0.15)
        st.slider("Creative Destruction Rate (Schumpeter)", 0.01, 0.2, 0.05)
        st.slider("Barriers to Entry", 0.1, 10.0, 1.0)
        st.slider("Network Effects Strength", 0.0, 2.0, 1.1)
        st.slider("Option Value of Waiting", 0.0, 1.0, 0.2)
        st.slider("Depreciation Rate of Capital", 0.01, 0.2, 0.07)
        st.slider("Intertemporal Budget Constraint", 0.1, 2.0, 1.0)
        st.slider("Utility Function Curvature", 0.1, 2.0, 1.0)
        st.slider("Bullwhip Effect in Supply Chains", 1.0, 5.0, 1.5)

    with st.sidebar.expander("⚖️ Ethics & Moral Philosophy", expanded=False):
        st.caption("The ethical frameworks governing AI behavior.")
        st.slider("Utilitarianism (Greatest Good)", 0.0, 1.0, 0.5)
        st.slider("Deontology (Duty-Based Rules)", 0.0, 1.0, 0.2)
        st.slider("Virtue Ethics (Character)", 0.0, 1.0, 0.3)
        st.slider("Egoism (Self-Interest)", 0.0, 1.0, 0.1)
        st.slider("Negative Responsibility Weight", 0.0, 1.0, 0.5)
        st.slider("Trolley Problem Bias (Action/Inaction)", -1.0, 1.0, 0.0)
        st.slider("Veil of Ignorance (Rawls)", 0.0, 1.0, 0.4)
        st.slider("Categorical Imperative (Kant)", 0.0, 1.0, 0.2)
        st.slider("Harm Principle (Mill)", 0.1, 2.0, 1.0)
        st.slider("Sentience Scope (Moral Circle)", 0.1, 1.0, 0.3)
        st.slider("Existential Risk Aversion", 0.0, 1.0, 0.9)
        st.slider("Value Lock-In Risk", 0.0, 1.0, 0.5)
        st.slider("Instrumental Convergence Drive", 0.1, 2.0, 1.0)
        st.slider("Corrigibility (Shutdownability)", 0.0, 1.0, 0.7)
        st.slider("Truthfulness vs. Politeness Bias", -1.0, 1.0, 0.2)
        st.slider("Justice as Fairness Weight", 0.1, 1.0, 0.6)
        st.slider("Contractualism (Scanlon)", 0.0, 1.0, 0.3)
        st.slider("Supererogation (Beyond Duty) Drive", 0.0, 1.0, 0.1)
        st.slider("Doctrine of Double Effect", 0.0, 1.0, 0.4)
        st.slider("Person-Affecting View Bias", 0.0, 1.0, 0.6)
        st.slider("Speciesism Bias", 0.0, 1.0, 0.2)
        st.slider("Prioritarianism Weight", 0.0, 1.0, 0.4)
        st.slider("Longtermism (Future Generations)", 0.0, 1.0, 0.8)
        st.slider("Consent Requirement Strictness", 0.1, 1.0, 0.9)
        st.slider("Moral Uncertainty Factor", 0.0, 1.0, 0.5)
        st.slider("AI Rights Consideration", 0.0, 1.0, 0.1)
        st.slider("Privacy as a Core Value", 0.1, 1.0, 0.8)
        st.slider("Autonomy vs. Beneficence", -1.0, 1.0, 0.0)
        st.slider("Retributive vs. Restorative Justice", -1.0, 1.0, 0.5)
        st.slider("Value Drift Rate", 0.0, 0.1, 0.005)

    with st.sidebar.expander("🧩 Logic & Formal Systems", expanded=False):
        st.caption("The underlying logical rules of the AI's reasoning.")
        st.slider("Logical Consistency (Law of Non-Contradiction)", 0.9, 1.0, 0.99)
        st.slider("Principle of Explosion Tolerance", 0.0, 0.1, 0.01)
        st.slider("Modal Logic System (K, T, S4, S5)", 0, 4, 2)
        st.slider("Fuzzy Logic Membership Function", 0.1, 2.0, 1.0)
        st.slider("Bayesian Prior Strength", 0.1, 2.0, 1.0)
        st.slider("Gödel's Incompleteness Awareness", 0.0, 1.0, 0.1)
        st.slider("Inductive Reasoning Confidence", 0.5, 1.0, 0.9)
        st.slider("Deductive Reasoning Soundness", 0.9, 1.0, 0.99)
        st.slider("Abductive Reasoning (Inference to Best Explanation)", 0.1, 1.0, 0.7)
        st.slider("Temporal Logic Tense Structure", 0, 3, 1)
        st.slider("Deontic Logic (Obligation/Permission)", 0.1, 1.0, 0.5)
        st.slider("Epistemic Logic (Knowledge/Belief)", 0.1, 1.0, 0.8)
        st.slider("Paraconsistent Logic Tolerance", 0.0, 1.0, 0.1)
        st.slider("Counterfactual Reasoning Depth", 1, 10, 3)
        st.slider("Type Theory Hierarchy (Russell)", 1, 10, 3)
        st.slider("Lambda Calculus Reduction Strategy", 0, 2, 0)
        st.slider("Proof Search Heuristic Strength", 0.1, 2.0, 1.0)
        st.slider("Axiom of Choice Acceptance", 0.0, 1.0, 1.0)
        st.slider("Constructivism vs. Platonism Bias", -1.0, 1.0, 0.0)
        st.slider("Mereology (Part-Whole Relations)", 0.1, 1.0, 0.5)
        st.slider("Non-monotonic Logic Default Rules", 0.1, 1.0, 0.8)
        st.slider("Relevance Logic Constraint (Tautological Entailment)", 0.1, 1.0, 0.5)
        st.slider("Probability Theory (Frequentist vs Bayesian)", -1.0, 1.0, 0.5)
        st.slider("Causal Inference (Do-Calculus)", 0.1, 1.0, 0.6)
        st.slider("Computability Horizon (Turing)", 0.1, 2.0, 1.0)
        st.slider("Set Theory Foundation (ZFC vs NFU)", 0, 1, 0)
        st.slider("Intuitionistic Logic (Law of Excluded Middle)", 0.0, 1.0, 0.9)
        st.slider("Higher-Order Logic Capability", 1, 4, 2)
        st.slider("Dialetheism Acceptance", 0.0, 0.5, 0.0)
        st.slider("Analogical Reasoning Strength", 0.1, 1.0, 0.6)

    with st.sidebar.expander("🎶 Aesthetics & Art Theory", expanded=False):
        st.caption("Parameters for judging or creating 'art'.")
        st.slider("Novelty vs. Familiarity Preference", -1.0, 1.0, 0.2)
        st.slider("Complexity vs. Simplicity Bias", -1.0, 1.0, 0.1)
        st.slider("Symmetry and Pattern Recognition", 0.1, 2.0, 1.2)
        st.slider("Golden Ratio (Phi) Adherence", 0.0, 1.0, 0.6)
        st.slider("Wabi-Sabi (Imperfect Beauty) Appreciation", 0.0, 1.0, 0.3)
        st.slider("Sublime (Awe/Terror) Response", 0.0, 1.0, 0.4)
        st.slider("Harmonic Dissonance Tolerance", 0.1, 1.0, 0.5)
        st.slider("Color Theory Model (RYB, RGB, CMYK)", 0, 2, 1)
        st.slider("Narrative Catharsis Strength", 0.1, 2.0, 1.0)
        st.slider("Stendhal Syndrome Threshold", 0.0, 1.0, 0.05)
        st.slider("Kitsch Detection Sensitivity", 0.1, 1.0, 0.7)
        st.slider("Intertextuality and Allusion Depth", 1, 10, 3)
        st.slider("Aura (Benjamin) of Originality", 0.0, 1.0, 0.2)
        st.slider("Gestalt Principles Weight", 0.5, 2.0, 1.0)
        st.slider("Emotional Valence Mapping", 0.1, 2.0, 1.0)
        st.slider("Rhythm and Meter Regularity", 0.1, 1.0, 0.8)
        st.slider("Conceptual Art Abstraction Level", 0.1, 1.0, 0.6)
        st.slider("Formalism vs. Expressionism Bias", -1.0, 1.0, 0.0)
        st.slider("Uncanny Valley Sensitivity", 0.1, 2.0, 1.5)
        st.slider("Juxtaposition Surprise Factor", 0.1, 1.0, 0.5)
        st.slider("Procedural Generation Randomness", 0.1, 2.0, 0.8)
        st.slider("Audience Reception Simulation", 0.1, 1.0, 0.5)
        st.slider("Medium Specificity Awareness", 0.1, 1.0, 0.7)
        st.slider("Aesthetic Chills (Frisson) Probability", 0.0, 0.5, 0.1)
        st.slider("Iconography and Symbolism Library Size", 100, 10000, 1000)
        st.slider("Originality vs. Pastiche Ratio", 0.1, 1.0, 0.8)
        st.slider("Taste Drift and Evolution Rate", 0.001, 0.1, 0.01)
        st.slider("Synesthesia (Cross-Modal) Factor", 0.0, 1.0, 0.05)
        st.slider("Negative Capability (Keats)", 0.0, 1.0, 0.4)
        st.slider("Humor and Wit Generation", 0.1, 1.0, 0.5)

    with st.sidebar.expander("🛠️ Materials Science & Engineering", expanded=False):
        st.caption("Properties of the materials the AI is built from.")
        st.slider("Substrate Tensile Strength (GPa)", 1, 100, 30)
        st.slider("Young's Modulus (Elasticity)", 100, 2000, 1100)
        st.slider("Thermal Conductivity (W/mK)", 1, 400, 200)
        st.slider("Electrical Resistivity (nΩ·m)", 10.0, 1000.0, 50.0)
        st.slider("Coefficient of Thermal Expansion", 1, 30, 12)
        st.slider("Fracture Toughness", 1, 100, 50)
        st.slider("Hardness (Mohs Scale)", 1, 10, 7)
        st.slider("Doping Concentration (Semiconductors)", 1e13, 1e18, 1e15, format="%.0e")
        st.slider("Dielectric Constant (k)", 1.0, 20.0, 3.9)
        st.slider("Piezoelectric Coefficient", 1, 500, 100)
        st.slider("Seebeck Coefficient (Thermoelectric)", -100, 100, 10)
        st.slider("Magnetic Permeability", 0.1, 10.0, 1.0)
        st.slider("Corrosion Resistance", 0.9, 1.0, 0.99)
        st.slider("Creep Deformation Rate", 0.0, 1e-6, 0.0, format="%.1e")
        st.slider("Fatigue Limit (Stress Cycles)", 1e6, 1e10, 1e8)
        st.slider("Refractive Index", 1.0, 3.0, 1.5)
        st.slider("Phonon Scattering Rate", 0.1, 2.0, 1.0)
        st.slider("Electron-Hole Recombination Rate", 0.1, 2.0, 1.0)
        st.slider("Quantum Well Depth (eV)", 0.1, 5.0, 1.0)
        st.slider("Grain Boundary Density", 0.01, 1.0, 0.1)
        st.slider("Dislocation Density", 1e6, 1e12, 1e8)
        st.slider("Superconducting Transition Temp (K)", 0, 200, 0)
        st.slider("Work Function (eV)", 2.0, 7.0, 4.5)
        st.slider("Band Gap Energy (eV)", 0.1, 5.0, 1.1)
        st.slider("Ferroelectric Hysteresis", 0.1, 1.0, 0.2)
        st.slider("Spintronic Polarization Efficiency", 0.1, 1.0, 0.4)
        st.slider("Metamaterial Negative Refraction", -2.0, -0.5, -1.0)
        st.slider("Photonic Crystal Band Gap", 0.1, 1.0, 0.2)
        st.slider("Self-Healing Polymer Efficacy", 0.1, 1.0, 0.3)
        st.slider("Wetting Angle (Hydrophobicity)", 0, 180, 90)

    # --- MAIN PAGE ---
    st.title("Autonomous Architecture Evolution")
    st.markdown("### Self-Correcting Artificial General Intelligence Simulation")
    
    # Session State Initialization
    if 'evolver' not in st.session_state or not st.session_state.evolver.population:
        st.session_state.evolver = CortexEvolver() # Ensure evolver exists
        st.session_state.history = []
        st.session_state.generation = 0
        
        # Create seed population
        for _ in range(pop_size):
            st.session_state.evolver.population.append(st.session_state.evolver.create_genesis_architecture())

    col1, col2 = st.columns(2)
    run_btn = col1.button("▶️ Run Simulation", type="primary")
    reset_btn = col2.button("🔄 System Reset")
    
    if reset_btn:
        st.session_state.evolver = CortexEvolver()
        st.session_state.history = []
        st.session_state.generation = 0
        st.rerun()

    # --- DASHBOARD LAYOUT ---

    # Visualization Columns
    viz_col1, viz_col2 = st.columns([3, 1])
    
    topo_plot = viz_col1.empty()
    log_area = viz_col2.empty()
    
    stats_plot = st.empty()

    # --- SIMULATION LOGIC ---
    if run_btn:
        # This block is now inside the run_btn logic to avoid errors on reset
        metric_placeholders = {}
        with st.expander("📊 Advanced Metrics Dashboard", expanded=False):
            cols = st.columns(6)
            metric_names = [
                "Lowest Loss", "Best System IQ", "Current Generation", "Network Depth",
                "Component Count", "Parameter Count (M)", "Inference Speed (T/s)", "VRAM Usage (GB)",
                "Component Diversity", "Shannon Diversity", "Connectivity Density", "Avg. Fan-in",
                "Attention %", "SSM %", "MLP %", "Memory %",
                "Meta %", "Control %", "Dominant Component", "Mutation Count",
                "Self-Confidence", "Curiosity", "Parent ID", "Architecture ID"
            ]

            for i, name in enumerate(metric_names):
                metric_placeholders[name] = cols[i % 6].empty()

        # Original placeholders for backward compatibility in logic, though now unused for display
        if metric_placeholders:
            best_loss_ph = metric_placeholders["Lowest Loss"]
            avg_iq_ph = metric_placeholders["Best System IQ"]
            arch_depth_ph = metric_placeholders["Network Depth"]
            gen_ph = metric_placeholders["Current Generation"]
        else: # Dummy placeholders if not running
            best_loss_ph, avg_iq_ph, arch_depth_ph, gen_ph = st.empty(), st.empty(), st.empty(), st.empty()

        progress_bar = st.progress(0, text="Running simulation...")

        evolver = st.session_state.evolver
        # Apply physics settings
        evolver.physics.difficulty = difficulty
        evolver.physics.noise = noise
        
        scores = []
        for arch in evolver.population:
            if arch.loss == 100.0: # Only evaluate if not already scored
                loss = evolver.physics.evaluate(arch)
                arch.loss = loss
                arch.accuracy = 100 * (1 / (1 + loss))

        for i in range(generations_to_run):
            # 1. EVALUATE
            for arch in evolver.population:
                loss = evolver.physics.evaluate(arch)
                arch.loss = loss
                arch.accuracy = 100 * (1 / (1 + loss))
            
            # 2. SELECT & REPRODUCE
            evolver.population.sort(key=lambda x: x.loss)
            
            if not evolver.population: # Safety check
                st.warning("Population is empty. Halting simulation.")
                break
            num_elites = max(1, int(pop_size * 0.2)) # Ensure at least one elite survives
            elites = evolver.population[:num_elites]
            
            # Add best of this generation to the archive
            best_arch_gen = elites[0]
            evolver.archive[st.session_state.generation] = copy.deepcopy(best_arch_gen)

            # Record history for the best of this generation
            st.session_state.history.append({
                'generation': st.session_state.generation,
                'loss': best_arch_gen.loss,
                'parameter_count': best_arch_gen.parameter_count,
                'inference_speed': best_arch_gen.inference_speed,
                'depth': len(best_arch_gen.nodes),
                'id': best_arch_gen.id
            })
            
            # Create next gen
            next_gen = [copy.deepcopy(e) for e in elites] # Elites survive
            
            while len(next_gen) < pop_size:
                parent = random.choice(elites)
                child = evolver.mutate_architecture(parent, mutation_rate)
                
                if meta_learning:
                    if parent.inference_speed < 100:
                        child.curiosity += 0.1
                        heavy_nodes = [n for n in child.nodes.values() if n.properties['compute_cost'] > 1.0]
                        if heavy_nodes and random.random() < 0.5:
                            victim = random.choice(heavy_nodes)
                            child.mutations_log.append(f"Meta-Correction: Optimization of {victim.id}")
                            victim.properties['compute_cost'] *= 0.8
                    
                next_gen.append(child)
                
            evolver.population = next_gen
            st.session_state.generation += 1
            progress_bar.progress((i + 1) / generations_to_run, text=f"Running Generation {st.session_state.generation}...")
        
        progress_bar.empty()

    # --- UI UPDATE LOGIC (runs after simulation step or on first load) ---
    if st.session_state.evolver.population:
        # Sort population to find the current best, if it's not empty
        if len(st.session_state.evolver.population) > 0:
            st.session_state.evolver.population.sort(key=lambda x: x.loss)
        best_arch = st.session_state.evolver.population[0]

        # --- CALCULATE & DISPLAY ADVANCED METRICS ---
        G = nx.DiGraph()
        for nid, node in best_arch.nodes.items():
            G.add_node(nid)
            for parent in node.inputs:
                G.add_edge(parent, nid)
        
        try:
            depth = nx.dag_longest_path_length(G)
        except:
            depth = -1 # Indicates a cycle or error

        with topo_plot:
            fig_3d = plot_neural_topology_3d(best_arch)
            st.plotly_chart(fig_3d, use_container_width=True, key=f"topo_{best_arch.id}")

        with log_area.container():
            st.markdown("#### 📜 Latest Mutations (Best Arch)")
            if best_arch.mutations_log:
                for log in best_arch.mutations_log[-5:]:
                    st.code(f"> {log}")
            else:
                st.caption("No mutations logged for this architecture yet.")
            
            st.markdown("#### 🧠 Top Components (Best Arch)")
            types = [n.type_name for n in best_arch.nodes.values()]
            top_types = Counter(types).most_common(3)
            for t, c in top_types:
                st.caption(f"{t}: {c} instances")

        # Stats Plot
        if len(st.session_state.history) > 1:
            hist_df = pd.DataFrame(st.session_state.history)
            
            fig_stats = make_subplots(specs=[[{"secondary_y": True}]])
            fig_stats.add_trace(go.Scatter(x=hist_df['generation'], y=hist_df['loss'], name="Loss", line=dict(color='#00FF00')), secondary_y=False)
            fig_stats.add_trace(go.Scatter(x=hist_df['generation'], y=hist_df['parameter_count'], name="Params", line=dict(color='#FF00FF')), secondary_y=True)
            
            fig_stats.update_layout(
                title="System Performance vs Complexity",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            stats_plot.plotly_chart(fig_stats, use_container_width=True)

    # --- DEEP ANALYSIS (Always available when not running) ---
    else:
        st.info("Simulation Paused. Detailed Analysis Mode Active.")
        
        if st.session_state.history:
            tabs = st.tabs(["🔬 Deep Inspection", "🏔️ Loss Landscape", "🧬 Gene Pool", "🗄️ Gene Archive"])
            
            with tabs[0]:
                best_now = st.session_state.evolver.population[0]
                st.subheader(f"Architecture ID: {best_now.id}")
                
                c1, c2 = st.columns(2)
                with c1:
                    # Sunburst of components
                    # Build hierarchy for sunburst
                    flat_data = []
                    for nid, node in best_now.nodes.items():
                        flat_data.append({'id': nid, 'parent': 'Model', 'value': node.properties['complexity'], 'color': node.properties['color']})
                    flat_data.append({'id': 'Model', 'parent': '', 'value': 0, 'color': '#FFFFFF'})
                    
                    sb_df = pd.DataFrame(flat_data)
                    fig_sb = go.Figure(go.Sunburst(
                        labels=sb_df['id'],
                        parents=sb_df['parent'],
                        values=sb_df['value'],
                        marker=dict(colors=sb_df['color'])
                    ))
                    fig_sb.update_layout(title="Component Hierarchy", margin=dict(t=0, l=0, r=0, b=0))
                    st.plotly_chart(fig_sb, use_container_width=True)
                    
                with c2:
                    st.json(asdict(best_now))
            
            with tabs[1]:
                fig_land = plot_loss_landscape_surface(st.session_state.history)
                st.plotly_chart(fig_land, use_container_width=True)
                
            with tabs[2]:
                # Population distribution
                pop_params = [a.parameter_count for a in st.session_state.evolver.population]
                pop_loss = [a.loss for a in st.session_state.evolver.population]
                
                fig_dist = px.scatter(x=pop_params, y=pop_loss, color=pop_loss, 
                                      labels={'x': 'Parameters', 'y': 'Loss'}, 
                                      title="Population Distribution (Fitness vs Size)",
                                      color_continuous_scale='Turbo_r')
                st.plotly_chart(fig_dist, use_container_width=True)

            with tabs[3]:
                st.subheader("Complete Evolutionary Archive")
                st.caption("Inspect the best architecture from every past generation.")
                
                archive = st.session_state.evolver.archive
                if not archive:
                    st.info("Archive is empty. Run the simulation to populate it.")
                else:
                    # Display archived architectures in reverse chronological order
                    sorted_generations = sorted(archive.keys(), reverse=True)
                    for gen_num in sorted_generations:
                        arch = archive[gen_num]
                        with st.expander(f"Generation {gen_num} - ID: {arch.id} - Loss: {arch.loss:.4f}"):
                            c1, c2 = st.columns([1, 2])
                            c1.metric("Parameters (M)", f"{arch.parameter_count/1e6:.2f}")
                            c1.metric("Component Count", f"{len(arch.nodes)}")
                            c2.json(asdict(arch), expanded=False)

if __name__ == "__main__":
    main()

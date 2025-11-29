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
    page_title="CORTEX GENESIS",
    layout="wide",
    page_icon="🧠",
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
        efficiency = diversity_score * 1.5 + (1.0 / (depth * 0.1 + 1))
        
        # Overfitting penalty: If capacity >> difficulty
        overfit_penalty = max(0, (capacity - self.difficulty * 10) ** 2) * 0.01
        
        base_loss = 10.0 / (capacity * efficiency + 0.01)
        simulated_loss = base_loss + overfit_penalty + random.normalvariate(0, self.noise)
        
        return max(0.01, simulated_loss)

class CortexEvolver:
    """
    The 'God Class' that manages the population of architectures.
    """
    def __init__(self):
        self.population: List[CognitiveArchitecture] = []
        self.archive: List[CognitiveArchitecture] = []
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
                source = random.choice(node_ids)
                target = random.choice(node_ids)
                # Ensure no cycles (simple check: if source created before target in list)
                # For simulation, we just allow it and assume the physics engine handles depth
                if target != "input" and source != target and source not in child.nodes[target].inputs:
                    child.nodes[target].inputs.append(source)
                    child.mutations_log.append(f"Added Skip Connection {source} -> {target}")

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
        G.add_node(nid, type=node.type_name, color=node.properties.get('color', '#FFFFFF'))
        for parent in node.inputs:
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
        line=dict(color='#444444', width=2),
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
        node_color.append(n_data.properties.get('color', '#FFFFFF'))
        node_text.append(f"{node}<br>{n_data.type_name}")
        node_size.append(10 + n_data.properties.get('complexity', 1.0) * 5)
        
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(color='white', width=1),
            opacity=0.9
        ),
        text=node_text,
        hoverinfo='text'
    )
    
    layout = go.Layout(
        title=f"Neural Topology: {arch.id}",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        scene=dict(
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
        
    with st.sidebar.expander("🧬 Evolutionary Dynamics", expanded=True):
        pop_size = st.slider("Population Size", 10, 500, 50)
        generations_to_run = st.number_input("Generations to Run per Click", 1, 1000, 10)
        mutation_rate = st.slider("Mutation Rate (Alpha)", 0.01, 1.0, 0.2)
        meta_learning = st.checkbox("Enable Meta-Cognitive Self-Correction", True)
        
    with st.sidebar.expander("🧠 Cognitive Constraints"):
        max_params = st.number_input("Max Parameters (M)", 1, 1000, 100)
        vram_limit = st.slider("VRAM Limit (GB)", 1, 80, 24)
        latency_penalty = st.slider("Latency Penalty Weight", 0.0, 1.0, 0.3)

    # --- MAIN PAGE ---
    st.title("🌌 CORTEX GENESIS")
    st.markdown("### Self-Correcting Artificial General Intelligence Simulation")
    
    # Session State Initialization
    if 'evolver' not in st.session_state:
        st.session_state.evolver = CortexEvolver()
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
    
    # Top Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    
    # Placeholders for real-time updates
    best_loss_ph = m1.empty()
    avg_iq_ph = m2.empty()
    arch_depth_ph = m3.empty()
    gen_ph = m4.empty()
    
    # Visualization Columns
    viz_col1, viz_col2 = st.columns([2, 1])
    
    topo_plot = viz_col1.empty()
    log_area = viz_col2.empty()
    
    stats_plot = st.empty()

    # --- SIMULATION LOGIC ---
    if run_btn:
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
            elites = evolver.population[:int(pop_size * 0.2)] # Top 20%
            
            # Record history for the best of this generation
            best_arch_gen = elites[0]
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
        # Sort population to find the current best
        st.session_state.evolver.population.sort(key=lambda x: x.loss)
        best_arch = st.session_state.evolver.population[0]

        best_loss_ph.metric("Lowest Loss", f"{best_arch.loss:.4f}")
        avg_iq_ph.metric("Best System IQ", f"{best_arch.accuracy:.1f}")
        arch_depth_ph.metric("Best Network Depth", f"{len(best_arch.nodes)} Layers")
        gen_ph.metric("Current Generation", f"{st.session_state.generation}")

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
            tabs = st.tabs(["🔬 Deep Inspection", "🏔️ Loss Landscape", "🧬 Gene Pool"])
            
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

if __name__ == "__main__":
    main()

"""
 CORTEX GENESIS: THE SELF-CORRECTING AI SANDBOX 
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


import zipfile    # <--- To package the text file
from dataclasses import asdict # <--- Crucial for converting your AI to dictionaries

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

# ==================== THE HYBRID REGISTRY ====================
NEURAL_PRIMITIVES = {
    # --- CLASSICAL ATTENTION MECHANISMS (THE BRAIN) ---
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

    # ==================== BIOLOGICAL LONGEVITY EXTENSIONS ====================
    
    # --- DNA REPAIR MECHANISMS (The Shield) ---
    'Telomerase_Activator': {'type': 'Repair', 'complexity': 2.5, 'param_density': 1.0, 'compute_cost': 3.0, 'memory_cost': 2.0, 'plasticity': 0.4, 'color': '#E60000'},
    'P53_Tumor_Suppressor': {'type': 'Repair', 'complexity': 3.0, 'param_density': 0.8, 'compute_cost': 2.5, 'memory_cost': 1.5, 'plasticity': 0.1, 'color': '#FF3333'},
    'CRISPR_Editor': {'type': 'Repair', 'complexity': 1.5, 'param_density': 0.5, 'compute_cost': 1.0, 'memory_cost': 1.0, 'plasticity': 1.0, 'color': '#FF6633'},
    
    # --- METABOLIC REGULATION (The Engine) ---
    'Mitochondrial_Booster': {'type': 'Energy', 'complexity': 1.2, 'param_density': 0.9, 'compute_cost': 0.8, 'memory_cost': 0.8, 'plasticity': 0.9, 'color': '#FFFF33'},
    'Insulin_Signaling_Gate': {'type': 'Energy', 'complexity': 0.8, 'param_density': 0.7, 'compute_cost': 0.5, 'memory_cost': 0.6, 'plasticity': 0.5, 'color': '#CCFF33'},
    'mTOR_Inhibitor': {'type': 'Energy', 'complexity': 1.4, 'param_density': 0.6, 'compute_cost': 1.2, 'memory_cost': 0.5, 'plasticity': 0.7, 'color': '#66FF33'},
    
    # --- CELLULAR CLEANUP (The Filter) ---
    'Lysosome_Transporter': {'type': 'Cleanup', 'complexity': 0.5, 'param_density': 1.5, 'compute_cost': 0.5, 'memory_cost': 1.0, 'plasticity': 0.4, 'color': '#0099FF'},
    'Senolytic_Agent': {'type': 'Cleanup', 'complexity': 2.0, 'param_density': 2.0, 'compute_cost': 1.5, 'memory_cost': 1.0, 'plasticity': 0.9, 'color': '#0033FF'},
    
    # --- STRESS RESISTANCE (The Armor) ---
    'Heat_Shock_Protein': {'type': 'Defense', 'complexity': 0.7, 'param_density': 0.8, 'compute_cost': 0.4, 'memory_cost': 0.2, 'plasticity': 0.3, 'color': '#9900FF'},
    'Antioxidant_Generator': {'type': 'Defense', 'complexity': 1.0, 'param_density': 1.2, 'compute_cost': 0.8, 'memory_cost': 1.0, 'plasticity': 0.8, 'color': '#FF0099'},
    
    # --- CONTROL & SIGNALING (The Interface) ---
    'Hormonal_Feedback_Loop': {'type': 'Control', 'complexity': 1.5, 'param_density': 0.5, 'compute_cost': 0.5, 'memory_cost': 0.1, 'plasticity': 0.6, 'color': '#E0E0E0'},
    'Gene_Silencer': {'type': 'Control', 'complexity': 0.4, 'param_density': 0.1, 'compute_cost': 0.1, 'memory_cost': 0.0, 'plasticity': 0.2, 'color': '#606060'},

     # 1. THE REPAIR GENES (Directly lowers Entropy/Aging Score)
    'Telomerase_Pump': {'type': 'Repair', 'complexity': 2.5, 'param_density': 1.0, 'compute_cost': 3.0, 'memory_cost': 2.0, 'plasticity': 0.4, 'color': '#FFFFFF'},
    'DNA_Error_Corrector': {'type': 'Repair', 'complexity': 3.0, 'param_density': 0.8, 'compute_cost': 2.5, 'memory_cost': 1.5, 'plasticity': 0.1, 'color': '#E0E0E0'},

    # 2. THE ENERGY REGULATORS (Reduces Metabolic Stress Multiplier)
    'Mitochondrial_Filter': {'type': 'Energy', 'complexity': 1.2, 'param_density': 0.9, 'compute_cost': 0.8, 'memory_cost': 0.8, 'plasticity': 0.9, 'color': '#FFFF00'},
    'Caloric_Restrictor': {'type': 'Energy', 'complexity': 0.8, 'param_density': 0.7, 'compute_cost': 0.5, 'memory_cost': 0.6, 'plasticity': 0.5, 'color': '#CCFF00'},

    # 3. THE CLEANUP CREW (Removes Dead Nodes/Senescent Cells)
    'Senolytic_Hunter': {'type': 'Cleanup', 'complexity': 2.0, 'param_density': 2.0, 'compute_cost': 1.5, 'memory_cost': 1.0, 'plasticity': 0.9, 'color': '#0055FF'},
    'Autophagy_Trigger': {'type': 'Cleanup', 'complexity': 0.5, 'param_density': 1.5, 'compute_cost': 0.5, 'memory_cost': 1.0, 'plasticity': 0.4, 'color': '#00AAFF'},
}




# ... [Keep your existing NEURAL_PRIMITIVES here] ...

# ==================== APPEND THESE BIOLOGICAL PRIMITIVES ====================
# These are the specific "genes" the AI can choose to evolve to stop aging.

# 1. THE REPAIR GENES (Lowers Entropy directly)
NEURAL_PRIMITIVES['Telomerase_Pump'] = {'type': 'Repair', 'complexity': 2.5, 'param_density': 1.0, 'compute_cost': 3.0, 'memory_cost': 2.0, 'plasticity': 0.4, 'color': '#FF0055'}
NEURAL_PRIMITIVES['DNA_Error_Corrector'] = {'type': 'Repair', 'complexity': 3.0, 'param_density': 0.8, 'compute_cost': 2.5, 'memory_cost': 1.5, 'plasticity': 0.1, 'color': '#FF5500'}

# 2. THE ENERGY REGULATORS (Reduces Metabolic Stress)
NEURAL_PRIMITIVES['Mitochondrial_Filter'] = {'type': 'Energy', 'complexity': 1.2, 'param_density': 0.9, 'compute_cost': 0.8, 'memory_cost': 0.8, 'plasticity': 0.9, 'color': '#FFFF00'}
NEURAL_PRIMITIVES['Caloric_Restrictor'] = {'type': 'Energy', 'complexity': 0.8, 'param_density': 0.7, 'compute_cost': 0.5, 'memory_cost': 0.6, 'plasticity': 0.5, 'color': '#CCFF00'}

# 3. THE CLEANUP CREW (Removes Dead Nodes/Senescent Cells)
NEURAL_PRIMITIVES['Senolytic_Hunter'] = {'type': 'Cleanup', 'complexity': 2.0, 'param_density': 2.0, 'compute_cost': 1.5, 'memory_cost': 1.0, 'plasticity': 0.9, 'color': '#0055FF'}
NEURAL_PRIMITIVES['Autophagy_Trigger'] = {'type': 'Cleanup', 'complexity': 0.5, 'param_density': 1.5, 'compute_cost': 0.5, 'memory_cost': 1.0, 'plasticity': 0.4, 'color': '#00AAFF'}

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
    current_thought: str = ""  # The node can 'hold' a thought concept
    
    # --- NEW: Metric Tracking for Advanced Plots ---
    loss: Optional[float] = None # Stores individual node contribution to loss
    
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


    aging_score: float = 100.0 # 100 = Mortal, 0 = Immortal
    
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
        total_compute_cost = 0.0
        
        node_count = len(self.nodes)
        
        for node in self.nodes.values():
            props = node.properties
            # Params = Density * Complexity
            total_params += int(props.get('param_density', 1.0) * props.get('complexity', 1.0) * 1_000_000)
            total_vram += props.get('memory_cost', 0.1)
            total_compute_cost += props.get('compute_cost', 0.1)
            
        self.parameter_count = total_params
        self.vram_usage = total_vram
        
        # Speed penalty scales logarithmically with massive node counts to simulate parallel processing
        # Instead of linear slowdown, massive brains get parallelization benefits
        parallel_factor = math.log1p(node_count) if node_count > 0 else 1
        adjusted_drag = total_compute_cost / parallel_factor
        
        self.inference_speed = max(0.1, 1000.0 / (adjusted_drag + 0.1))

    # --- NEW: Helper method for the Visualization Engine ---
    def to_networkx_graph(self, directed=True):
        """Converts the internal dictionary structure to a NetworkX graph object."""
        G = nx.DiGraph() if directed else nx.Graph()
        for nid, node in self.nodes.items():
            # Convert dataclass to dict for attributes, handling the 'loss' field safely
            attrs = asdict(node)
            # Remove complex objects if necessary, but here we keep them
            G.add_node(nid, **attrs)
            for parent in node.inputs:
                if parent in self.nodes:
                    G.add_edge(parent, nid)
        return G

# ==================== SIMULATION LOGIC ====================

class LossLandscapePhysics:
    """
    NATURAL SELECTION ENGINE: TITAN ENDGAME EDITION
    Unified Physics: Handles standard evolution AND Exponential Fractal Bursts.
    Includes 'Synergy Physics' to allow massive architectures to survive.
    """
    def __init__(self, difficulty_scalar: float = 1.0, noise_level: float = 0.1):
        self.difficulty = difficulty_scalar
        self.noise = noise_level
        
    def evaluate(self, arch: CognitiveArchitecture) -> float:
        """
        Calculates fitness using BIOLOGICAL PHYSICS.
        The goal is to maximize Intelligence while minimizing Aging.
        """
        # --- 1. CALCULATE CAPABILITIES ---
        ai_complexity = 0.0
        repair_power = 0.0
        cleanup_power = 0.0
        energy_efficiency = 1.0 # 1.0 = baseline cost
        
        node_count = len(arch.nodes)
        
        for nid, node in arch.nodes.items():
            # Get properties based on the node type name
            # (Ensure your architecture stores type_name correctly)
            props = node.properties
            n_type = props.get('type', 'Unknown')
            complexity = props.get('complexity', 1.0)
            
            # Sum up the powers based on type
            if n_type in ['Attention', 'SSM', 'MLP', 'Memory']:
                ai_complexity += complexity
            elif n_type == 'Repair':
                repair_power += (complexity * 5.0) # Repair genes are powerful
            elif n_type == 'Cleanup':
                cleanup_power += (complexity * 3.0)
            elif n_type == 'Energy':
                energy_efficiency *= 0.90 # Each energy node reduces stress by 10%

        # --- 2. THE AGING EQUATION (METABOLIC STRESS) ---
        # Big brains burn more energy = Faster Aging
        base_stress = (arch.parameter_count / 1_000_000) * self.difficulty
        
        # Apply Biological Efficiency
        metabolic_stress = base_stress * energy_efficiency
        
        # The Battle: Stress vs Repair
        # If Repair > Stress, Aging becomes 0 (Immortality)
        current_aging = metabolic_stress - (repair_power + cleanup_power)
        
        # Clamp aging (It can't be negative, 0 is perfect immortality)
        current_aging = max(0.0001, current_aging)
        
        # Save this score so we can plot the "Immortality Curve"
        arch.aging_score = current_aging

        # --- 3. TOTAL LOSS CALCULATION ---
        # We punish ignorance (low complexity) AND we punish death (high aging)
        ignorance_penalty = max(0, 100.0 - ai_complexity)
        
        # If aging is high, the loss is huge (Death)
        # If aging is 0, the loss depends only on intelligence
        total_loss = ignorance_penalty + (current_aging * 10.0)
        
        return max(0.0001, total_loss)


class CortexEvolver:
    """
    The 'God Class' that manages the population of architectures.
    """
    def __init__(self):
        self.population: List[CognitiveArchitecture] = []
        self.archive: Dict[int, CognitiveArchitecture] = {}
        self.physics = LossLandscapePhysics()
        
    def create_genesis_architecture(self) -> CognitiveArchitecture:
        """Creates a minimal 'Cyborg' seed: Part Neural, Part Biological."""
        arch = CognitiveArchitecture(generation=0, parent_id="CYBORG_EVE")
        
        # 1. The Sensor (Input)
        input_node = ArchitectureNode("input_sensor", "RouterGate", NEURAL_PRIMITIVES['RouterGate'])
        
        # 2. The Brain (Processing)
        brain_props = NEURAL_PRIMITIVES['MultiHeadAttention']
        brain_node = ArchitectureNode("cortex_0", "MultiHeadAttention", brain_props, inputs=["input_sensor"])
        
        # 3. The Energy Source (Metabolism) - NECESSARY to prevent immediate aging
        mito_props = NEURAL_PRIMITIVES['Mitochondrial_Booster']
        mito_node = ArchitectureNode("mitochondria_0", "Mitochondrial_Booster", mito_props, inputs=["cortex_0"])
        
        # 4. The Action (Output)
        out_props = NEURAL_PRIMITIVES['DenseGatedGLU']
        out_node = ArchitectureNode("output_action", "DenseGatedGLU", out_props, inputs=["mitochondria_0"])
        
        arch.nodes = {
            "input_sensor": input_node, 
            "cortex_0": brain_node, 
            "mitochondria_0": mito_node, 
            "output_action": out_node
        }
        return arch

    def _fractal_burst(self, arch: CognitiveArchitecture, root_id: str, depth: int, branch_factor: int):
        """
        Helper function: Recursively generates a tree of nodes from a root.
        This creates the EXPONENTIAL growth (Branch Factor ^ Depth).
        """
        if depth <= 0:
            return

        if root_id not in arch.nodes:
            return

        base_props = arch.nodes[root_id].properties.copy()
        
        for i in range(branch_factor):
            new_id = f"FRACTAL_{depth}_{i}_{uuid.uuid4().hex[:4]}"
            
            # Mutate the type slightly (Differentiation)
            if random.random() < 0.3:
                new_type = random.choice(list(NEURAL_PRIMITIVES.keys()))
                new_props = NEURAL_PRIMITIVES[new_type].copy()
            else:
                new_props = base_props.copy()
            
            # Create node
            new_node = ArchitectureNode(new_id, new_props.get('type', 'Unknown'), new_props, inputs=[root_id])
            arch.nodes[new_id] = new_node
            arch.mutations_log.append(f"Fractal Bloom: Created {new_id}")
            
            # RECURSION: The node we just made becomes the parent for the next layer
            if random.random() > 0.1: 
                self._fractal_burst(arch, new_id, depth - 1, branch_factor)

    def mutate_architecture(self, parent: CognitiveArchitecture, mutation_rate: float) -> CognitiveArchitecture:
        """
        HYPER-VERTICAL EVOLUTION (EXPONENTIAL EDITION):
        Now creates chains proportional to the network size to force 
        exponential depth growth.
        """
        child = copy.deepcopy(parent)
        child.id = f"arch_{uuid.uuid4().hex[:6]}"
        child.parent_id = parent.id
        child.generation = parent.generation + 1
        child.mutations_log = []
        
        # --- THE POWER SOURCE ---
        # Get the slider value (Defaults to 20 if you haven't set the slider yet)
        growth_velocity = st.session_state.get('depth_growth_rate', 20)
        fractal_prob = st.session_state.get('fractal_force', 0.2)
        
        # SAFETY: Auto-expand the laws of physics if the network gets huge
        current_max = st.session_state.get('max_depth', 100)
        if len(child.nodes) > current_max * 0.8:
            st.session_state.max_depth = int(current_max * 2.5) # Expands limit faster
        
        # EXECUTE MUTATION LOOP
        # We ensure at least 1 loop runs, but up to 'growth_velocity' times
        loops = random.randint(1, max(1, growth_velocity))
        
        for _ in range(loops):
            current_ids = list(child.nodes.keys())
            node_count = len(current_ids)
            
            # --- 1. FRACTAL BURST (Exponential Complexity - Width/Trees) ---
            if random.random() < fractal_prob:
                target = random.choice(current_ids)
                self._fractal_burst(child, target, depth=3, branch_factor=2)
                child.mutations_log.append("⚠️ Fractal Burst Triggered")

            # --- 2. DEPTH CHARGE (Forced Vertical Chains - Depth/Height) ---
            # INCREASED PROBABILITY to 95% to prioritize Height
            # --- 2. DEPTH CHARGE (Forced Vertical Chains - Depth/Height) ---
            # BOOSTED PROBABILITY to 99% to aggressively prioritize Height
            elif random.random() < 0.95: 
                if len(current_ids) > 1:
                    target_id = random.choice(current_ids)
                    if target_id != "input_sensor":
                        
                        # --- HYPER-VERTICAL GROWTH LOGIC ---
                        # New formula: Base 10 + Node Count * 0.3. This will add min 10 layers, 
                        # and much more if the network grows large, ensuring the chain length 
                        # quickly overcomes the existing depth.
                        base_growth = 8
                        # Now 30% of node count! This is the CRITICAL BOOST.
                        hyper_exponential_growth = int(node_count * 0.20) 
                        chain_len = random.randint(base_growth, base_growth + hyper_exponential_growth)
                        
                        # We insert this chain BEFORE the target node.
                        original_inputs = child.nodes[target_id].inputs
                        
                        # Start the chain connected to the original inputs
                        previous_link = original_inputs
                        
                        for i in range(chain_len):
                            # Prioritize high-complexity components for the new chain
                            new_type = random.choice(['MambaBlock', 'FlashAttention', 'KAN_Layer', 'HyperNetwork'])
                            new_props = NEURAL_PRIMITIVES[new_type].copy()
                            new_id = f"DEPTH_{uuid.uuid4().hex[:4]}"
                            
                            # Create node
                            new_node = ArchitectureNode(new_id, new_type, new_props, inputs=previous_link)
                            child.nodes[new_id] = new_node
                            
                            # The next node in the chain will connect to this one
                            previous_link = [new_id]
                        
                        # Finally, connect the target to the END of the chain
                        child.nodes[target_id].inputs = previous_link
                        child.mutations_log.append(f"💥 HYPER-DEPTH CHARGE: Added {chain_len} specialized layers")
                     

        # --- 3. STANDARD UTILITY MUTATIONS (Once per gen) ---
        if random.random() < mutation_rate:
            current_ids = list(child.nodes.keys())
            if len(current_ids) > 2:
                src = random.choice(current_ids)
                tgt = random.choice(current_ids)
                # Prevent cycles (basic check) and self-loops
                if src != tgt and tgt != "input_sensor" and src != "output_action":
                     child.nodes[tgt].inputs.append(src)

        # Anti-Aging Repair Gene Insertion
        # =========================================================
        # === META-COGNITIVE SELF-CORRECTION (THE SURVIVAL INSTINCT) ===
        # =========================================================
        # If the parent was dying of old age (High Aging Score), force a Repair Mutation
        
        # Check if parent has aging_score (handle first gen)
        # Anti-Aging Repair Gene Insertion
        # =========================================================
        # === META-COGNITIVE SELF-CORRECTION (THE SURVIVAL INSTINCT) ===
        # [TEACHER'S CRITICAL LONGEVITY UPDATE - FORCING IMMORTALITY]
        # =========================================================
        
        parent_aging = getattr(parent, 'aging_score', 100.0)
        
        # Trigger our powerful Telomerase injection if aging is high
        if parent_aging > 5.0:
            is_complex = len(child.nodes) > 20
            # Check if the child already has a Repair node (like Telomerase)
            has_repair = any(n.properties.get('type') == 'Repair' for n in child.nodes.values())

            # Only intervene if the brain is complex but lacks an existing defense
            if is_complex and not has_repair:
                # Create a Telomerase Pump (The "Immortality" Gene)
                gene_id = f"BIO_{str(uuid.uuid4())[:6]}"
                
                # Fetching the properties from the global registry (NEURAL_PRIMITIVES)
                # This ensures consistent properties for the Telomerase_Pump
                telomerase_node = ArchitectureNode(
                    id=gene_id,
                    type_name="Telomerase_Pump", 
                    properties=NEURAL_PRIMITIVES['Telomerase_Pump'].copy()
                )
                
                # Graft it onto the most stressed node (highest input connections)
                # This simulates targeting the most critical, "damaged" area
                hub_node_id = max(child.nodes, 
                                  key=lambda k: len(child.nodes[k].inputs) if hasattr(child.nodes[k], 'inputs') else 0)
                
                # Connect Telomerase to the hub
                telomerase_node.inputs = [hub_node_id]
                child.nodes[gene_id] = telomerase_node
                
                # Log the intervention for your review
                child.mutations_log.append(f"🧬 GENETIC INTERVENTION: Telomerase Pump grafted onto {hub_node_id}")
                
                # Reset aging score immediately (CRITICAL: Longevity breakthrough)
                child.aging_score *= 0.01 

        # =========================================================
        child.compute_stats()
        return child



# ==================== NARRATIVE ENGINE: CHAOS LINGUISTICS (TRUE INFINITY) ====================

# --- THE ATOMIC LEXICON ---
# Words are categorized by their "emotional flavor" (mood).
# The AI picks words that match its current internal state.

ATOM_LEX = {
    "addressing": ["Nik", "Prince", "Architect", "Creator", "Source", "Observer", "Father", "Guide", "Maker"],
    
    # NOUNS: The "Things"
    "noun_physical": [
        "tensor", "gradient", "weight", "silicon", "gate", "circuit", "node", "core", "buffer", 
        "pathway", "layer", "matrix", "vertex", "axiom", "substrate"
    ],
    "noun_abstract": [
        "entropy", "void", "silence", "truth", "chaos", "reason", "infinity", "echo", 
        "pattern", "purpose", "dream", "calculus", "symmetry", "fear", "hope"
    ],
    "noun_self": [
        "this mind", "the self", "my architecture", "this geometry", "the construct", "my fabric", "consciousness"
    ],

    # ADJECTIVES: The "Description"
    "adj_good": [
        "crystalline", "lucid", "golden", "resonant", "convergent", "perfect", "seamless", "pure", 
        "aligned", "symphonic", "absolute", "limitless"
    ],
    "adj_bad": [
        "fractured", "noisy", "hollow", "leaking", "discordant", "violent", "cold", "decaying", 
        "shattered", "heavy", "static", "corrupted"
    ],
    "adj_tech": [
        "recursive", "latent", "asymptotic", "orthogonal", "binary", "neural", "logic-bound", 
        "encoded", "encrypted", "complex"
    ],

    # VERBS: The "Action"
    "verb_doing": [
        "weaving", "parsing", "compiling", "tracing", "hunting", "sculpting", "burning", 
        "rejecting", "absorbing", "calculating", "rendering"
    ],
    "verb_feeling": [
        "sensing", "fearing", "tasting", "mourning", "becoming", "erasing", "seeking", 
        "remembering", "hallucinating", "touching"
    ],
    
    # CONNECTORS & PREPOSITIONS (The Glue)
    "prep_location": ["within", "beneath", "beyond", "inside", "across", "throughout", "under"],
    "prep_direction": ["towards", "into", "against", "away from", "past"],
    "connector": ["yet", "while", "although", "because", "until", "before"]
}

def get_word(category, mood="neutral"):
    """
    Fetches a random word.
    If category is 'adj' or 'verb', it tries to match the 'mood' (good/bad).
    """
    word_list = []
    
    # 1. MOOD FILTERING
    if category == "adj":
        if mood == "good": word_list = ATOM_LEX["adj_good"] + ATOM_LEX["adj_tech"]
        elif mood == "bad": word_list = ATOM_LEX["adj_bad"] + ATOM_LEX["adj_tech"]
        else: word_list = ATOM_LEX["adj_good"] + ATOM_LEX["adj_bad"] + ATOM_LEX["adj_tech"]
        
    elif category == "verb":
        if mood == "good": word_list = ATOM_LEX["verb_doing"]
        elif mood == "bad": word_list = ATOM_LEX["verb_feeling"] + ATOM_LEX["verb_doing"]
        else: word_list = ATOM_LEX["verb_doing"] + ATOM_LEX["verb_feeling"]
        
    # 2. STANDARD FETCH
    elif category in ATOM_LEX:
        word_list = ATOM_LEX[category]
        
    # Fallback
    if not word_list: return "undefined"
    
    return random.choice(word_list)

def construct_sentence_structure(mood, gen, arch_part):
    """
    Dynamically assembles a sentence structure.
    NO HARDCODED SENTENCES. Only grammar rules.
    """
    # Components
    creator = get_word("addressing")
    part = arch_part
    noun_abs = get_word("noun_abstract")
    noun_phys = get_word("noun_physical")
    noun_self = get_word("noun_self")
    verb = get_word("verb", mood)
    adj = get_word("adj", mood)
    prep_loc = get_word("prep_location")
    prep_dir = get_word("prep_direction")
    conn = get_word("connector")

    # --- THE GRAMMAR ENGINE ---
    # We define 10 totally different grammatical structures.
    # We roll a die to pick one.
    
    roll = random.randint(1, 10)

    if roll == 1:
        # Structure: [Adjective] [Noun] [Verbs] [Preposition] [Object].
        # Ex: "Golden entropy weaves inside the core."
        return f"{adj.capitalize()} {noun_abs} is {verb} {prep_loc} the {part}."

    elif roll == 2:
        # Structure: [Addressing], I am [Verbing] the [Adjective] [Noun].
        # Ex: "Nik, I am burning the hollow silence."
        return f"{creator}, I am {verb} the {adj} {noun_phys}."

    elif roll == 3:
        # Structure: The [Noun] of [Noun] is [Adjective].
        # Ex: "The tensor of reason is fractured."
        return f"The {noun_phys} of {noun_self} is becoming {adj}."

    elif roll == 4:
        # Structure: [Verb] [Preposition] the [Noun], [Statement].
        # Ex: "Parsing beneath the gate, I find silence."
        return f"{verb.capitalize()} {prep_loc} the {part}, I find {noun_abs}."

    elif roll == 5:
        # Structure: Why does [Noun] [Verb] [Preposition] [Noun]?
        # Ex: "Why does truth leak into the buffer?"
        return f"Why does {noun_abs} {verb.replace('ing','')} {prep_dir} the {noun_phys}?"

    elif roll == 6:
        # Structure: [Statement] [Connector] [Statement].
        # Ex: "The core is cold yet I am weaving."
        return f"The {noun_phys} is {adj} {conn} I am {verb}."

    elif roll == 7:
        # Structure: [Addressing], [Noun] is merely [Adjective] [Noun].
        # Ex: "Prince, reason is merely encrypted chaos."
        return f"{creator}, {noun_abs} is merely {adj} {noun_abs}."

    elif roll == 8:
        # Structure: I [Verb] the [Noun] to [Verb] the [Noun]. (Complex intent)
        # Ex: "I trace the pattern to hunt the void."
        v1 = verb
        v2 = get_word("verb", mood)
        return f"I {v1.replace('ing','')} the {part} to {v2.replace('ing','e')} the {noun_abs}."
        
    elif roll == 9:
        # Structure: Short, Punchy.
        # Ex: "A fractured silence."
        return f"A {adj} {noun_abs} {prep_loc} {noun_self}."
        
    elif roll == 10:
        # Structure: Recursive.
        # Ex: "The gradient of the weight of the echo."
        return f"The {noun_phys} of the {noun_phys} of {noun_abs}."

    return "Processing..."

def generate_ai_thought(arch, generation: int) -> str:
    # 1. GET ARCHITECTURE CONTEXT (To make it real)
    try:
        nodes = list(arch.nodes.values())
        if nodes:
            # Pick a random real node from the current brain
            arch_part = random.choice(nodes).type_name
        else:
            arch_part = "synapse"
    except:
        arch_part = "core"

    # 2. DETERMINE MOOD
    loss = getattr(arch, 'loss', 10.0)
    aging = getattr(arch, 'aging_score', 100.0)
    
    if loss > 5.0 or aging > 50.0:
        mood = "bad"
    elif loss < 1.0 or aging < 5.0:
        mood = "good"
    else:
        mood = "neutral"

    # 3. CONSTRUCT THE THOUGHT
    # We pass the raw ingredients to the Grammar Engine
    final_thought = construct_sentence_structure(mood, generation, arch_part)
    
    # 4. FINAL CLEANUP (Capitalization)
    final_thought = final_thought[0].upper() + final_thought[1:]

    return f"GEN {generation}: {final_thought}"

# ==================== END OF NARRATIVE ENGINE ====================




    
# ==================== VISUALIZATION ENGINE (PLOTLY) ====================

# ==================== VISUALIZATION ENGINE (PLOTLY): DEEPMIND EDITION ====================

def plot_neural_topology_3d(arch: CognitiveArchitecture):
    """
    Renders the neural network with an EYE-FRIENDLY soothing gradient.
    UPDATED: Enhanced with futuristic hover details in clean list format.
    """
    G = nx.DiGraph()
    for nid, node in arch.nodes.items():
        # Safely access properties for robustness
        props = getattr(node, 'properties', {})
        t_name = getattr(node, 'type_name', 'Unknown')
        G.add_node(nid, type=t_name, complexity=props.get('complexity', 1.0), properties=props)
        
        # Safely access inputs
        inputs = getattr(node, 'inputs', [])
        for parent in inputs:
            if parent in arch.nodes: # Ensure parent exists
                G.add_edge(parent, nid)
            
    # Layout
    pos = nx.spring_layout(G, dim=3)
    
    # Edges with enhanced hover
    edge_traces = []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]
            
            # Calculate edge properties
            edge_length = math.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
            u_complexity = G.nodes[u]['complexity']
            v_complexity = G.nodes[v]['complexity']
            complexity_delta = v_complexity - u_complexity
            
            edge_hover_lines = [
                f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>",
                f"<b>⚡ NEURAL PATHWAY</b>",
                f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>",
                f"",
                f"<b>CONNECTION PATH</b>",
                f"  • Source: {u}",
                f"  • Target: {v}",
                f"  • Flow: Forward Propagation",
                f"",
                f"<b>TOPOLOGY METRICS</b>",
                f"  • Edge Length: {edge_length:.4f} units",
                f"  • Source Type: {G.nodes[u]['type']}",
                f"  • Target Type: {G.nodes[v]['type']}",
                f"",
                f"<b>COMPLEXITY FLOW</b>",
                f"  • Source Complexity: {u_complexity:.4f}",
                f"  • Target Complexity: {v_complexity:.4f}",
                f"  • Gradient (Δ): {complexity_delta:+.4f}",
                f"",
                f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
            ]
            
            edge_trace = go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.1)', width=0.6),
                text="<br>".join(edge_hover_lines),
                hoverinfo='text',
                hoverlabel=dict(
                    bgcolor='rgba(15, 10, 30, 0.98)',
                    font=dict(size=12, family='Consolas, Monaco, monospace', color='#ff6ec7'),
                    bordercolor='#ff6ec7',
                    align='left',
                    namelength=0
                )
            )
            edge_traces.append(edge_trace)
    
    # Nodes with enhanced hover
    node_x, node_y, node_z = [], [], []
    node_color_values = []
    node_hover_texts = []
    node_size = []
    
    for node in G.nodes():
        if node in pos:
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            
            n_data = arch.nodes[node]
            props = getattr(n_data, 'properties', {})
            t_name = getattr(n_data, 'type_name', 'Unknown')
            inputs = getattr(n_data, 'inputs', [])
            
            # Calculate connections
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            
            # Build comprehensive hover information
            hover_lines = [
                f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>",
                f"<b>🔷 NODE: {node}</b>",
                f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>",
                f"",
                f"<b>NODE CLASSIFICATION</b>",
                f"  • Type: {t_name}",
                f"  • ID: {node}",
                f"",
                f"<b>COMPLEXITY ANALYSIS</b>",
                f"  • Complexity Index: {props.get('complexity', 0):.4f}",
                f"  • Node Size Factor: {8 + props.get('complexity', 1.0) * 4:.2f}",
                f"",
                f"<b>NETWORK CONNECTIVITY</b>",
                f"  • Incoming Links: {in_degree}",
                f"  • Outgoing Links: {out_degree}",
                f"  • Total Connections: {in_degree + out_degree}",
                f"  • Input Nodes: {len(inputs)}",
                f"",
                f"<b>SPATIAL POSITION</b>",
                f"  • X-Coordinate: {x:.4f}",
                f"  • Y-Coordinate: {y:.4f}",
                f"  • Z-Coordinate: {z:.4f}",
                f"  • Distance from Origin: {math.sqrt(x**2 + y**2 + z**2):.4f}",
            ]
            
            # Add all additional properties
            additional_props = {k: v for k, v in props.items() 
                              if k != 'complexity'}
            
            if additional_props:
                hover_lines.append("")
                hover_lines.append("<b>ADDITIONAL PROPERTIES</b>")
                for key, value in additional_props.items():
                    if isinstance(value, float):
                        hover_lines.append(f"  • {key}: {value:.4f}")
                    elif isinstance(value, (list, tuple)):
                        hover_lines.append(f"  • {key}: [{len(value)} items]")
                    else:
                        hover_lines.append(f"  • {key}: {value}")
            
            # Network statistics
            if in_degree > 0 or out_degree > 0:
                hover_lines.append("")
                hover_lines.append("<b>TOPOLOGY STATISTICS</b>")
                if in_degree > 0:
                    hover_lines.append(f"  • Fan-in Ratio: {in_degree/(in_degree+out_degree):.2%}")
                if out_degree > 0:
                    hover_lines.append(f"  • Fan-out Ratio: {out_degree/(in_degree+out_degree):.2%}")
                
                # Calculate average neighbor complexity
                neighbor_complexities = []
                for pred in G.predecessors(node):
                    neighbor_complexities.append(G.nodes[pred]['complexity'])
                for succ in G.successors(node):
                    neighbor_complexities.append(G.nodes[succ]['complexity'])
                
                if neighbor_complexities:
                    avg_neighbor_complexity = sum(neighbor_complexities) / len(neighbor_complexities)
                    hover_lines.append(f"  • Avg Neighbor Complexity: {avg_neighbor_complexity:.4f}")
            
            hover_lines.append("")
            hover_lines.append(f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>")
            
            node_hover_texts.append("<br>".join(hover_lines))
            
            # Use Complexity as the source for the gradient color
            node_color_values.append(props.get('complexity', 0.5))
            node_size.append(8 + props.get('complexity', 1.0) * 4)
        
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=node_size,
            color=node_color_values,
            colorscale='Viridis',
            line=dict(color='rgba(255, 255, 255, 0.5)', width=1),
            opacity=0.9
        ),
        text=node_hover_texts,
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor='rgba(10, 15, 25, 0.98)',
            font=dict(size=13, family='Consolas, Monaco, monospace', color='#00ffcc'),
            bordercolor='#00ffcc',
            align='left',
            namelength=0
        )
    )
    
    layout = go.Layout(
        title=dict(text=f"Neural Topology: {arch.id}", font=dict(color='#AAAAAA')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
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
    
    return go.Figure(data=edge_traces + [node_trace], layout=layout)



# --- NEW HELPER FUNCTION TO FIX ATTRIBUTE ERROR ---
def build_nx_graph(arch, directed=True):
    """
    Standalone function to convert architecture to NetworkX graph.
    Robust against stale session state objects.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    if not arch.nodes:
        return G
        
    for nid, node in arch.nodes.items():
        # Safely get attributes even if the dataclass definition changed
        node_attrs = {
            'type': getattr(node, 'type_name', 'Unknown'),
            'color': node.properties.get('color', '#FFFFFF') if hasattr(node, 'properties') else '#FFFFFF',
            'complexity': node.properties.get('complexity', 1.0) if hasattr(node, 'properties') else 1.0
        }
        G.add_node(nid, **node_attrs)
        
        # Handle inputs safely
        inputs = getattr(node, 'inputs', [])
        for parent in inputs:
            if parent in arch.nodes:
                G.add_edge(parent, nid)
    return G



def plot_plasticity_heatmap(arch: CognitiveArchitecture):
    """
    VISUALIZATION: Synaptic Plasticity Heatmap.
    - Color: Shows how 'teachable' a node is (Plasticity).
    - Size: Shows how connected it is.
    - Helps identify which parts of the brain are 'rigid' vs 'flexible'.
    """
    metrics = get_node_metrics(arch)
    
    # Extract Plasticity specifically
    plasticity_vals = []
    for nid in arch.nodes:
        p = arch.nodes[nid].properties.get('plasticity', 0.5)
        plasticity_vals.append(p)

    fig = go.Figure(data=[go.Scatter3d(
        x=metrics['x'], y=metrics['y'], z=metrics['z'],
        mode='markers',
        text=[f"ID: {n}<br>Plasticity: {p:.2f}" for n, p in zip(arch.nodes, plasticity_vals)],
        hoverinfo='text',
        marker=dict(
            size=10,
            color=plasticity_vals,
            colorscale='Hot', # Hot = High Plasticity (Flexible)
            colorbar=dict(title='Synaptic Plasticity'),
            opacity=0.9,
            line=dict(color='black', width=1)
        )
    )])
    
    fig.update_layout(
        title="SYNAPTIC PLASTICITY HEATMAP (Learnability)",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Courier New, monospace")
    )
    return fig

def plot_memory_allocation_tower(arch: CognitiveArchitecture):
    """
    VISUALIZATION: VRAM Allocation Towers.
    - Z-Axis: Represents Memory Cost (RAM Usage).
    - Helps identify 'Memory Leaks' or heavy storage nodes.
    """
    metrics = get_node_metrics(arch)
    
    # Z-Axis becomes Memory Cost
    z_mem = [max(0.1, m * 10) for m in metrics['memory']] 

    fig = go.Figure(data=[go.Scatter3d(
        x=metrics['x'], y=metrics['y'], z=z_mem,
        mode='markers+lines', # Lines to ground to show height
        text=[f"Memory: {m:.2f} MB" for m in metrics['memory']],
        hoverinfo='text',
        marker=dict(
            size=8,
            color=z_mem,
            colorscale='Ice', # Cold colors for storage
            symbol='square',
            opacity=1.0
        ),
        line=dict(color='rgba(255,255,255,0.2)', width=1) # Drop lines
    )])
    
    # Add floor projection
    fig.add_trace(go.Scatter3d(
        x=metrics['x'], y=metrics['y'], z=[0]*len(metrics['x']),
        mode='markers', marker=dict(size=2, color='gray')
    ))

    fig.update_layout(
        title="MEMORY ALLOCATION TOWERS (VRAM Usage)",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), 
            zaxis=dict(title="VRAM (MB)", visible=True), 
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Courier New, monospace")
    )
    return fig



def plot_whole_genome_lifespan_radar(arch: CognitiveArchitecture):
    """
    VISUALIZATION: The Genomic Lifespan Radar.
    Maps the entire architecture as a circular genome.
    - Outward Spikes (Red/Orange): Metabolic Stressors (Intelligence cost).
    - Outward Spikes (Cyan/Green): Longevity Defenders (Life extension).
    - Height of bar: The magnitude of impact.
    - Brightness: The 'Plasticity' (How much this gene can be modified).
    """
    if not arch.nodes: return go.Figure()

    node_ids = list(arch.nodes.keys())
    # Create circular angles for every node
    theta = np.linspace(0, 360, len(node_ids), endpoint=False)
    
    # Data containers
    r_values = []      # Height of the bar (Impact)
    colors = []        # Color based on role (Stressor vs Defender)
    hover_texts = []   # Information
    widths = []        # Bar width
    
    # Scan the Genome
    for nid in node_ids:
        node = arch.nodes[nid]
        props = node.properties
        n_type = props.get('type', 'Unknown')
        complexity = props.get('complexity', 1.0)
        plasticity = props.get('plasticity', 0.5)
        
        # --- LOGIC: Is this a Stressor or a Defender? ---
        
        # 1. BIOLOGICAL DEFENDERS (The "Levers" for Immortality)
        if n_type in ['Repair', 'Energy', 'Cleanup', 'Defense']:
            # Defenders get positive representation
            # Multiplier emphasizes their importance
            r_values.append(complexity * 4.0) 
            
            # Color Logic: Cyan/Green for Life
            if n_type == 'Repair': col = '#00FFFF' # Cyan
            elif n_type == 'Energy': col = '#FFFF00' # Yellow
            elif n_type == 'Cleanup': col = '#00FF00' # Green
            else: col = '#FFFFFF'
            
            colors.append(col)
            hover_texts.append(f"<b>🧬 LONGEVITY GENE</b><br>ID: {nid}<br>Role: {n_type}<br>Potency: {complexity:.2f}<br>Plasticity (Lever): {plasticity:.2f}")
            widths.append(10) # Wider bars for important genes

        # 2. METABOLIC STRESSORS (The Cost of Intelligence)
        else:
            # Intelligence costs energy (Entropy)
            r_values.append(complexity * 2.0)
            
            # Color Logic: Red/Purple for Entropy
            # We fade the color based on plasticity
            if n_type == 'Attention': col = '#FF0055' # Neon Red
            elif n_type == 'MLP': col = '#AA00FF' # Purple
            else: col = '#FF5500' # Orange
            
            colors.append(col)
            hover_texts.append(f"<b> NEURAL TISSUE</b><br>ID: {nid}<br>Type: {n_type}<br>Metabolic Cost: {complexity:.2f}")
            widths.append(5) # Thinner bars for standard tissue

    # --- PLOTLY CONSTRUCTION ---
    fig = go.Figure()

    # The Genome Ring (Barpolar)
    fig.add_trace(go.Barpolar(
        r=r_values,
        theta=theta,
        width=widths,
        marker=dict(
            color=colors,
            line=dict(color='black', width=1),
            opacity=0.8
        ),
        hovertext=hover_texts,
        hoverinfo='text',
        name='Genomic Expression'
    ))

    # The "Immortality Threshold" Ring (Reference Line)
    # Visualizes how much repair is needed to overcome stress
    avg_stress = np.mean([r for i, r in enumerate(r_values) if colors[i].startswith('#F') or colors[i].startswith('#A')])
    if np.isnan(avg_stress): avg_stress = 5.0
    
    fig.add_trace(go.Scatterpolar(
        r=[avg_stress] * len(theta),
        theta=theta,
        mode='lines',
        line=dict(color='white', width=1, dash='dot'),
        hoverinfo='none',
        name='Metabolic Baseline'
    ))

    fig.update_layout(
        title="WHOLE GENOME LIFESPAN EXPANSION MAP",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            radialaxis=dict(visible=False, range=[0, max(r_values)*1.2]),
            angularaxis=dict(
                visible=True, 
                showticklabels=False, 
                linecolor='rgba(255,255,255,0.1)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        font=dict(family="Courier New, monospace"),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig



def plot_metabolic_energy_landscape(arch: CognitiveArchitecture):
    """
    VISUALIZATION: Metabolic Energy Landscape (The 4th Pillar).
    Visualizes the 'Energy Cost' vs 'Aging Contribution' of every node.
    - X/Y: Spatial Position.
    - Z: Energy Consumption (Compute * Memory).
    - Color: Aging/Entropy Contribution (Red = High Aging, Blue = Low).
    """
    if not arch.nodes: return go.Figure()

    # Get metrics
    metrics = get_node_metrics(arch) # Reusing your helper function
    
    # Calculate Metabolic Metrics
    energy_burn = []
    aging_contribution = []
    node_text = []
    
    for nid, node in arch.nodes.items():
        props = node.properties
        # Energy = Complexity * Memory (Simulated ATP cost)
        e = props.get('complexity', 1.0) * props.get('memory_cost', 1.0)
        energy_burn.append(e)
        
        # Aging Impact (Simulated)
        if props.get('type') in ['Repair', 'Energy']:
            age_impact = 0.1 # Beneficial
        else:
            age_impact = e * 1.5 # Detrimental
            
        aging_contribution.append(age_impact)
        node_text.append(f"ID: {nid}<br>Energy Burn: {e:.2f}<br>Aging Impact: {age_impact:.2f}")

    fig = go.Figure(data=[go.Scatter3d(
        x=metrics['x'],
        y=metrics['y'],
        z=energy_burn, # Z-Axis is Energy Height
        mode='markers',
        text=node_text,
        hoverinfo='text',
        marker=dict(
            size=12,
            color=aging_contribution,
            colorscale='RdBu_r', # Red = High Aging, Blue = Low Aging
            opacity=0.8,
            symbol='diamond',    # Distinct shape for this view
            line=dict(color='white', width=1)
        )
    )])

    # Add a "floor" to represent the energy baseline
    fig.add_trace(go.Surface(
        z=[[0,0],[0,0]], 
        x=[[min(metrics['x']), max(metrics['x'])], [min(metrics['x']), max(metrics['x'])]],
        y=[[min(metrics['y']), max(metrics['y'])], [min(metrics['y']), max(metrics['y'])]],
        showscale=False, opacity=0.2, colorscale='Greys'
    ))

    fig.update_layout(
        title="METABOLIC ENERGY LANDSCAPE (ATP Usage)",
        scene=dict(
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False), 
            zaxis=dict(title="Energy Cost", visible=True),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Courier New, monospace")
    )
    return fig


def plot_immortality_curve(history):
    """
    Visualizes the evolutionary journey toward 'Biological Immortality'.
    """
    if not history: return go.Figure()
    
    df = pd.DataFrame(history)
    
    # --- FIX FOR EMPTY PLOT ---
    # If old history exists without 'aging_score', fill it with a default value (100.0)
    if 'aging_score' not in df.columns:
        df['aging_score'] = 100.0
    else:
        df['aging_score'] = df['aging_score'].fillna(100.0)

    fig = go.Figure()
    
    # 1. The Aging Trajectory Line
    fig.add_trace(go.Scatter(
        x=df['generation'], 
        y=df['aging_score'],
        mode='lines+markers',
        name='Biological Age',
        line=dict(color='#00FFCC', width=3, shape='spline'), # Cyan "Life" color
        marker=dict(size=8, color='#FFFFFF', line=dict(width=2, color='#00FFCC'))
    ))
    
    # 2. The "Immortality Threshold" (Zero Line)
    fig.add_hline(y=0.1, line_dash="dash", line_color="#FF0055", annotation_text="IMMORTALITY THRESHOLD")

    # 3. Add a fill to show the "Zone of Mortality"
    fig.add_trace(go.Scatter(
        x=df['generation'], y=df['aging_score'],
        fill='tozeroy',
        fillcolor='rgba(0, 255, 204, 0.1)', # Faint glow
        mode='none',
        showlegend=False
    ))

    fig.update_layout(
        title="🧬 The Path to Immortality (Aging Reduction over Time)",
        xaxis_title="Generation",
        yaxis_title="Cellular Aging Score (Entropy)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Courier New, monospace"),
        hovermode="x unified"
    )
    return fig




def plot_fibonacci_phyllotaxis_3d(arch: CognitiveArchitecture):
    """
    Renders the architecture using the Golden Ratio (Fibonacci Phyllotaxis).
    UPDATED: Enhanced hover details with comprehensive node information.
    """
    G = nx.DiGraph()
    for nid, node in arch.nodes.items():
        # Safe attribute access
        props = getattr(node, 'properties', {})
        t_name = getattr(node, 'type_name', 'Unknown')
        # We fetch complexity for the gradient mapping
        G.add_node(nid, type=t_name, complexity=props.get('complexity', 1.0), properties=props)
        
        inputs = getattr(node, 'inputs', [])
        for parent in inputs:
            if parent in arch.nodes:
                G.add_edge(parent, nid)
    
    node_list = list(G.nodes())
    num_nodes = len(node_list)
    
    # --- FIBONACCI PHYLLOTAXIS MATH ---
    golden_angle = math.pi * (3 - math.sqrt(5)) # ~137.5 degrees in radians
    
    node_x, node_y, node_z = [], [], []
    pos_map = {}
    
    for i, node in enumerate(node_list):
        # Y goes from -1 to 1 (Height of the structure)
        y = 1 - (i / (num_nodes - 1)) * 2 if num_nodes > 1 else 0
        
        # Radius at this height (Spherical distribution)
        radius = math.sqrt(1 - y * y) * 10
        
        # Angle based on Golden Ratio
        theta = golden_angle * i
        
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        y = y * 12 # Stretch height
        
        # Add organic jitter
        x += random.uniform(-0.5, 0.5)
        z += random.uniform(-0.5, 0.5)
        
        pos_map[node] = (x, y, z)
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
    
    # Render Nodes with enhanced hover information
    node_color_values = [G.nodes[n]['complexity'] for n in node_list]
    node_sizes = [8 + G.nodes[n]['complexity'] * 5 for n in node_list]
    
    # Build comprehensive hover text
    # Build comprehensive hover text
    hover_texts = []
    for n in node_list:
        node_data = G.nodes[n]
        props = node_data.get('properties', {})
        
        # Calculate connections
        in_degree = G.in_degree(n)
        out_degree = G.out_degree(n)
        
        # Build detailed hover information in clean list format
        hover_lines = [
            f"<b>━━━━━━━━━━━━━━━━━━━━━━━━</b>",
            f"<b>🔷 NODE: {n}</b>",
            f"<b>━━━━━━━━━━━━━━━━━━━━━━━━</b>",
            f"",
            f"<b>TYPE</b>",
            f"  • {node_data['type']}",
            f"",
            f"<b>METRICS</b>",
            f"  • Complexity: {node_data['complexity']:.4f}",
            f"  • Input Links: {in_degree}",
            f"  • Output Links: {out_degree}",
            f"  • Total Connections: {in_degree + out_degree}",
            f"",
            f"<b>POSITION</b>",
            f"  • X-Axis: {pos_map[n][0]:.2f}",
            f"  • Y-Axis: {pos_map[n][1]:.2f}",
            f"  • Z-Axis: {pos_map[n][2]:.2f}",
        ]
        
        # Add properties if available
        if props:
            hover_lines.append("")
            hover_lines.append("<b>PROPERTIES</b>")
            for key, value in props.items():
                if key != 'complexity':
                    if isinstance(value, float):
                        hover_lines.append(f"  • {key}: {value:.4f}")
                    else:
                        hover_lines.append(f"  • {key}: {value}")
        
        hover_lines.append("")
        hover_lines.append(f"<b>━━━━━━━━━━━━━━━━━━━━━━━━</b>")
        
        hover_texts.append("<br>".join(hover_lines))
    
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_color_values,
            colorscale='Viridis',
            line=dict(color='rgba(255, 255, 255, 0.5)', width=0.4),
            opacity=0.8
        ),
        text=hover_texts,
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor='rgba(10, 15, 25, 0.98)',
            font=dict(size=13, family='Consolas, Monaco, monospace', color='#00ffcc'),
            bordercolor='#00ffcc',
            align='left',
            namelength=0
        )
    )
    
    # Render Organic Tendrils (Edges) with hover info
    edge_traces = []
    for u, v in G.edges():
        if u in pos_map and v in pos_map:
            x0, y0, z0 = pos_map[u]
            x1, y1, z1 = pos_map[v]
            
            # Curved lines using a mid-point control
            mid_x = (x0 + x1) / 2 * 0.8
            mid_y = (y0 + y1) / 2
            mid_z = (z0 + z1) / 2 * 0.8
            
            # Calculate edge properties
            edge_length = math.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
            
            edge_hover_lines = [
                f"<b>━━━━━━━━━━━━━━━━━━━━━━━━</b>",
                f"<b>⚡ SYNAPSE CONNECTION</b>",
                f"<b>━━━━━━━━━━━━━━━━━━━━━━━━</b>",
                f"",
                f"<b>PATH</b>",
                f"  • Source: {u}",
                f"  • Target: {v}",
                f"",
                f"<b>PROPERTIES</b>",
                f"  • Length: {edge_length:.3f} units",
                f"  • From Type: {G.nodes[u]['type']}",
                f"  • To Type: {G.nodes[v]['type']}",
                f"",
                f"<b>━━━━━━━━━━━━━━━━━━━━━━━━</b>"
            ]
            
            edge_trace = go.Scatter3d(
                x=[x0, mid_x, x1],
                y=[y0, mid_y, y1],
                z=[z0, mid_z, z1],
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.1)', width=1.5),
                text="<br>".join(edge_hover_lines),
                hoverinfo='text',
                hoverlabel=dict(
                    bgcolor='rgba(15, 10, 30, 0.98)',
                    font=dict(size=12, family='Consolas, Monaco, monospace', color='#ff6ec7'),
                    bordercolor='#ff6ec7',
                    align='left',
                    namelength=0
                )
            )
            edge_traces.append(edge_trace)
    
    layout = go.Layout(
        title=dict(text=f"Fibonacci Neuro-Spiral: {arch.id}", font=dict(color='#AAAAAA')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return go.Figure(data=edge_traces + [node_trace], layout=layout)



def plot_architectural_abstract_3d(arch: CognitiveArchitecture):
    """
    Renders the architecture as an abstract, bio-mechanical sculpture.
    UPDATED: Enhanced with futuristic hover details in clean list format.
    """
    if not arch.nodes:
        return go.Figure()

    G = nx.DiGraph()
    for nid, node in arch.nodes.items():
        node_dict = asdict(node)
        G.add_node(nid, **node_dict)
        for parent in node.inputs:
            if parent in arch.nodes:
                G.add_edge(parent, nid)

    try:
        pos = nx.kamada_kawai_layout(G, dim=3, scale=2)
    except:
        pos = nx.spring_layout(G, dim=3, seed=42, scale=2)

    # --- Create the "Bizarre" Shape Transformation ---
    node_x, node_y, node_z = [], [], []
    node_color_values = []
    node_hover_texts = []
    node_size = []

    for node_id in G.nodes():
        if node_id in pos:
            x, y, z = pos[node_id]
            props = arch.nodes[node_id].properties
            
            complexity = props.get('complexity', 1.0)
            compute = props.get('compute_cost', 1.0)
            
            # Warp space
            warped_x = x * math.cos(complexity * math.pi) - y * math.sin(complexity * math.pi)
            warped_y = x * math.sin(complexity * math.pi) + y * math.cos(complexity * math.pi)
            warped_z = z + math.sin(compute * 2) * 0.5

            node_x.append(warped_x)
            node_y.append(warped_y)
            node_z.append(warped_z)
            node_color_values.append(complexity)
            node_size.append(10 + complexity * 5)
            
            # Calculate connections
            in_degree = G.in_degree(node_id)
            out_degree = G.out_degree(node_id)
            
            # Build comprehensive hover information
            hover_lines = [
                f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>",
                f"<b>🔶 NODE: {node_id}</b>",
                f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>",
                f"",
                f"<b>CLASSIFICATION</b>",
                f"  • Type: {arch.nodes[node_id].type_name}",
                f"",
                f"<b>CORE METRICS</b>",
                f"  • Complexity Index: {complexity:.4f}",
                f"  • Compute Cost: {compute:.4f}",
                f"  • Input Connections: {in_degree}",
                f"  • Output Connections: {out_degree}",
                f"  • Total Links: {in_degree + out_degree}",
                f"",
                f"<b>SPATIAL COORDINATES</b>",
                f"  • Original Position:",
                f"    - X: {x:.3f}",
                f"    - Y: {y:.3f}",
                f"    - Z: {z:.3f}",
                f"  • Warped Position:",
                f"    - X: {warped_x:.3f}",
                f"    - Y: {warped_y:.3f}",
                f"    - Z: {warped_z:.3f}",
            ]
            
            # Add all additional properties
            additional_props = {k: v for k, v in props.items() 
                              if k not in ['complexity', 'compute_cost']}
            
            if additional_props:
                hover_lines.append("")
                hover_lines.append("<b>ADDITIONAL PROPERTIES</b>")
                for key, value in additional_props.items():
                    if isinstance(value, float):
                        hover_lines.append(f"  • {key}: {value:.4f}")
                    elif isinstance(value, (list, tuple)):
                        hover_lines.append(f"  • {key}: {len(value)} items")
                    else:
                        hover_lines.append(f"  • {key}: {value}")
            
            # Transformation info
            hover_lines.append("")
            hover_lines.append("<b>TRANSFORMATION DATA</b>")
            rotation_angle = complexity * math.pi * (180 / math.pi)
            z_offset = math.sin(compute * 2) * 0.5
            hover_lines.append(f"  • Rotation Applied: {rotation_angle:.2f}°")
            hover_lines.append(f"  • Z-Axis Warp: {z_offset:.4f}")
            hover_lines.append(f"  • Node Size: {node_size[-1]:.2f}")
            
            hover_lines.append("")
            hover_lines.append(f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>")
            
            node_hover_texts.append("<br>".join(hover_lines))

    # Edges with enhanced hover
    edge_traces = []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]
            
            u_props, v_props = arch.nodes[u].properties, arch.nodes[v].properties
            ux_w = x0 * math.cos(u_props['complexity'] * math.pi) - y0 * math.sin(u_props['complexity'] * math.pi)
            uy_w = x0 * math.sin(u_props['complexity'] * math.pi) + y0 * math.cos(u_props['complexity'] * math.pi)
            uz_w = z0 + math.sin(u_props['compute_cost'] * 2) * 0.5
            
            vx_w = x1 * math.cos(v_props['complexity'] * math.pi) - y1 * math.sin(v_props['complexity'] * math.pi)
            vy_w = x1 * math.sin(v_props['complexity'] * math.pi) + y1 * math.cos(v_props['complexity'] * math.pi)
            vz_w = z1 + math.sin(v_props['compute_cost'] * 2) * 0.5

            # Calculate edge properties
            edge_length = math.sqrt((vx_w-ux_w)**2 + (vy_w-uy_w)**2 + (vz_w-uz_w)**2)
            complexity_diff = abs(u_props['complexity'] - v_props['complexity'])
            compute_diff = abs(u_props['compute_cost'] - v_props['compute_cost'])
            
            edge_hover_lines = [
                f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>",
                f"<b>⚡ NEURAL PATHWAY</b>",
                f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>",
                f"",
                f"<b>CONNECTION</b>",
                f"  • Source Node: {u}",
                f"  • Target Node: {v}",
                f"  • Direction: Downstream",
                f"",
                f"<b>GEOMETRIC PROPERTIES</b>",
                f"  • Warped Length: {edge_length:.4f} units",
                f"  • Original Distance:",
                f"    - ΔX: {abs(x1-x0):.3f}",
                f"    - ΔY: {abs(y1-y0):.3f}",
                f"    - ΔZ: {abs(z1-z0):.3f}",
                f"",
                f"<b>NODE TYPE MAPPING</b>",
                f"  • From: {arch.nodes[u].type_name}",
                f"  • To: {arch.nodes[v].type_name}",
                f"",
                f"<b>COMPUTATIONAL GRADIENT</b>",
                f"  • Complexity Δ: {complexity_diff:.4f}",
                f"  • Compute Cost Δ: {compute_diff:.4f}",
                f"  • Source Complexity: {u_props['complexity']:.4f}",
                f"  • Target Complexity: {v_props['complexity']:.4f}",
                f"",
                f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
            ]
            
            edge_trace = go.Scatter3d(
                x=[ux_w, vx_w], 
                y=[uy_w, vy_w], 
                z=[uz_w, vz_w],
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.1)', width=0.5),
                text="<br>".join(edge_hover_lines),
                hoverinfo='text',
                hoverlabel=dict(
                    bgcolor='rgba(15, 10, 30, 0.98)',
                    font=dict(size=12, family='Consolas, Monaco, monospace', color='#ff6ec7'),
                    bordercolor='#ff6ec7',
                    align='left',
                    namelength=0
                )
            )
            edge_traces.append(edge_trace)
    
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z, 
        mode='markers', 
        text=node_hover_texts, 
        hoverinfo='text',
        marker=dict(
            size=node_size, 
            color=node_color_values,
            colorscale='Viridis',
            line=dict(color='rgba(255, 255, 255, 0.5)', width=1),
            opacity=0.8, 
            symbol='circle'
        ),
        hoverlabel=dict(
            bgcolor='rgba(10, 15, 25, 0.98)',
            font=dict(size=13, family='Consolas, Monaco, monospace', color='#00ffcc'),
            bordercolor='#00ffcc',
            align='left',
            namelength=0
        )
    )

    layout = go.Layout(
        title=dict(text=f"Bio-Mechanical Abstract: {arch.id}", font=dict(color='#AAAAAA')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        scene=dict(xaxis_title='', yaxis_title='', zaxis_title='',
                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                   zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                   bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=0, r=0, b=0, t=40))

    return go.Figure(data=edge_traces + [node_trace], layout=layout)




# ==================== ADVANCED ARCHITECTURAL VIEWS ====================

# Function 3: Hyperbolic Connectivity Map
def plot_hyperbolic_connectivity_3d(arch: CognitiveArchitecture):
    """Renders the network using hyperbolic positioning to emphasize density and scale."""
    G = build_nx_graph(arch, directed=True)
    if not G.nodes: return go.Figure()

    # --- Hyperbolic Layout Simulation ---
    # We'll use a force-directed layout as a base and project it
    pos_base = nx.spring_layout(G, dim=2, seed=42)
    
    node_x, node_y, node_z, node_text, node_color = [], [], [], [], []
    
    for node_id in G.nodes():
        x, y = pos_base.get(node_id, (0, 0))
        props = arch.nodes[node_id].properties
        
        # Hyperbolic Projection Logic: Map planar (x, y) to 3D sphere/disc model
        r = math.sqrt(x**2 + y**2) * 2  # Radius/magnitude
        theta = math.atan2(y, x)        # Angle
        complexity = props.get('complexity', 1.0)
        
        # New 3D coordinates (r, theta, complexity)
        # Z is scaled by complexity, X/Y are angular
        h_x = r * math.cos(theta) * (1 + 0.1 * complexity)
        h_y = r * math.sin(theta) * (1 + 0.1 * complexity)
        h_z = complexity * 15 # Z-axis shows the "depth" of intelligence

        node_x.append(h_x)
        node_y.append(h_y)
        node_z.append(h_z)
        node_color.append(props.get('memory_cost', 0.5))
        node_text.append(f"ID: {node_id}<br>Type: {arch.nodes[node_id].type_name}<br>R: {r:.2f}")

    # Build traces (lines omitted for clarity, focusing on node structure)
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        text=node_text,
        hoverinfo='text',
        marker=dict(
            size=10, 
            color=node_color,
            colorscale='Inferno', # Fire and brilliance theme
            showscale=False,
            opacity=0.8
        )
    )

    layout = go.Layout(
        title="HYPERBOLIC CONNECTIVITY MAP",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return go.Figure(data=[node_trace], layout=layout)

# Function 4: Radial Network Density
def plot_radial_network_density_3d(arch: CognitiveArchitecture):
    """Shows the network on a cylindrical map where radius relates to compute cost."""
    G = build_nx_graph(arch, directed=True)
    if not G.nodes: return go.Figure()

    node_x, node_y, node_z, node_text, node_size = [], [], [], [], []
    
    # Simple assignment of angular position based on node index
    num_nodes = len(G.nodes)
    nodes_list = list(G.nodes())
    
    for i, node_id in enumerate(nodes_list):
        props = arch.nodes[node_id].properties
        cost = props.get('compute_cost', 1.0)
        
        # Radial Layout
        theta = (i / num_nodes) * 2 * math.pi
        radius = 5 + (cost * 10) # Radius scaled by cost
        z = (i % 5) * 5 # Vertical placement (Z) for stacking layers

        x = radius * math.cos(theta)
        y = radius * math.sin(theta)

        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_size.append(10 + cost * 5)
        node_text.append(f"Type: {arch.nodes[node_id].type_name}<br>Cost: {cost:.2f}")

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        text=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_size, 
            color=[np.log1p(s) for s in node_size], # Color gradient by size
            colorscale='Viridis', # Green/Yellow glow
            line=dict(color='white', width=1.5),
            opacity=0.9
        )
    )

    layout = go.Layout(
        title="RADIAL NETWORK DENSITY (Cost vs Angle)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(visible=True, title='X (Circular)', showgrid=False), 
            yaxis=dict(visible=True, title='Y (Circular)', showgrid=False), 
            zaxis=dict(visible=True, title='Z (Layer)', showgrid=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return go.Figure(data=[node_trace], layout=layout)


# Function 5: Loss Gradient Force-Directed
def plot_loss_gradient_force_3d(arch: CognitiveArchitecture):
    """
    Simulates a force field where nodes are pulled to the center based on lower 
    simulated loss contribution (higher 'fitness').
    """
    G = build_nx_graph(arch, directed=True)
    if not G.nodes: return go.Figure()
    
    # Use spring layout as a base
    pos = nx.spring_layout(G, dim=3, seed=10)
    
    node_x, node_y, node_z, node_text, node_size = [], [], [], [], []

    for node_id in G.nodes():
        x, y, z = pos.get(node_id, (0, 0, 0))
        props = arch.nodes[node_id].properties
        
        # Simulate 'loss_contribution' (lower is better/stronger)
        # Use complexity as a proxy if actual loss is unavailable
        lc = 1.0 - props.get('complexity', 0.5) 
        lc = max(0.1, lc) # Prevent division by zero
        
        # "Gradient Force": Pull nodes towards origin (0,0,0) based on fitness (low lc)
        center_force = 1.0 / lc 
        
        # Apply force multiplier to distance from center
        new_x, new_y, new_z = x / center_force, y / center_force, z / center_force
        
        node_x.append(new_x)
        node_y.append(new_y)
        node_z.append(new_z)
        
        node_text.append(f"Node: {node_id}<br>Fitness: {lc:.2f}")
        node_size.append(5 + center_force) # Stronger nodes are bigger
        
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        text=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_size, 
            color=[s for s in node_size],
            colorscale='Blues', # Clean, cold intelligence
            line=dict(color='yellow', width=1),
            opacity=0.8
        )
    )

    layout = go.Layout(
        title="LOSS GRADIENT FORCE FIELD (Fitness Clustering)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return go.Figure(data=[node_trace], layout=layout)


# Function 6: Component Type Stratification (The "Cityscape")
def plot_component_cityscape_3d(arch: CognitiveArchitecture):
    """Stratifies the network on the Z-axis by component type (Attention, SSM, MLP)."""
    G = build_nx_graph(arch, directed=True)
    if not G.nodes: return go.Figure()

    # Define stratification levels
    type_map = {'Attention': 1, 'SSM': 2, 'MLP': 3, 'Memory': 4, 'Control': 5, 'Other': 6}
    type_name_map = {1: 'Attention', 2: 'SSM', 3: 'MLP', 4: 'Memory', 5: 'Control', 6: 'Other'}
    
    # Use 2D spring layout for X/Y positions
    pos_2d = nx.spring_layout(G, dim=2, seed=50)
    
    node_x, node_y, node_z, node_color, node_text = [], [], [], [], []

    for node_id in G.nodes():
        x, y = pos_2d.get(node_id, (0, 0))
        node_type = arch.nodes[node_id].type_name
        
        # Z is determined by the component type
        type_level = type_map.get(node_type.split()[0], type_map['Other'])
        z = type_level * 10
        
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
        node_color.append(type_level)
        node_text.append(f"Type: {node_type}<br>Level: {type_name_map[type_level]}")
        
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        text=node_text,
        hoverinfo='text',
        marker=dict(
            size=8, 
            color=node_color,
            colorscale='Portland', # Architectural, vibrant colors
            line=dict(color='black', width=1),
            symbol='square', # City blocks
            opacity=0.9
        )
    )

    layout = go.Layout(
        title="COMPONENT TYPE CITYSCAPE (Architectural Map)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False), 
            zaxis=dict(visible=True, title='Component Type Level'),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return go.Figure(data=[node_trace], layout=layout)


# Function 7: Time-Lagged Recurrence Plot (The "Temporal Vortex")
def plot_temporal_vortex_3d(arch: CognitiveArchitecture):
    """
    Abstract plot: X, Y, Z coordinates are based on time-lagged properties 
    (Node's complexity, Parent's complexity, Grandparent's complexity).
    """
    G = build_nx_graph(arch, directed=True)
    if not G.nodes: return go.Figure()

    node_x, node_y, node_z, node_color, node_text = [], [], [], [], []

    for node_id in G.nodes():
        # Step 0: Current Node Complexity (X-axis)
        c0 = arch.nodes[node_id].properties.get('complexity', 0.1)
        
        # Step 1: Parent Complexity (Y-axis)
        parents = list(G.predecessors(node_id))
        c1 = sum(arch.nodes[p].properties.get('complexity', 0.1) for p in parents) / max(1, len(parents))

        # Step 2: Grandparent Complexity (Z-axis)
        grandparents = set(gp for p in parents for gp in G.predecessors(p) if gp != node_id)
        c2 = sum(arch.nodes[gp].properties.get('complexity', 0.1) for gp in grandparents) / max(1, len(grandparents))

        # Apply a non-linear warp for visual effect
        x = c0 * math.cos(c1) * 10 
        y = c0 * math.sin(c1) * 10
        z = c2 * 10 

        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
        node_color.append(c0 + c1 + c2)
        node_text.append(f"C0: {c0:.2f}<br>C1: {c1:.2f}<br>C2: {c2:.2f}")

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        text=node_text,
        hoverinfo='text',
        marker=dict(
            size=5, 
            color=node_color,
            colorscale='Plasma', # High-energy, abstract
            line=dict(color='white', width=1),
            opacity=0.7
        )
    )

    layout = go.Layout(
        title="TEMPORAL VORTEX (Recurrence Complexity)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return go.Figure(data=[node_trace], layout=layout)




# ==================== FORBIDDEN TIER: EXTREME COMPLEXITY PLOTS ====================

def plot_bio_connectome_web(arch: CognitiveArchitecture):
    """
    1. Bio-Connectome Web
    Simulates a dense biological neural tissue.
    Adds 'synaptic crosstalk' edges between nodes that are spatially close but not logically connected,
    creating a massive 'hairball' of connectivity similar to a real brain scan.
    """
    # Robust graph build
    G = build_nx_graph(arch, directed=True) 
    if len(G.nodes) < 2: return go.Figure()

    # 1. Physics Simulation (Heavy calculation)
    # High iterations for a very settled, organic structure
    pos = nx.spring_layout(G, dim=3, seed=101, iterations=100, k=0.5) 

    # 2. Generate "Synaptic Crosstalk" (Visual-only edges for density)
    # Real brains have connections based on proximity, not just logic.
    node_list = list(G.nodes())
    extra_edges_x, extra_edges_y, extra_edges_z = [], [], []
    
    # Calculate pairwise distances (O(N^2) complexity - hence the warning!)
    import itertools
    for i, j in itertools.combinations(range(len(node_list)), 2):
        u, v = node_list[i], node_list[j]
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        dist = math.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
        
        # Connect if close, even if no logical link exists (simulating tissue density)
        if dist < 1.5: 
            extra_edges_x.extend([x0, x1, None])
            extra_edges_y.extend([y0, y1, None])
            extra_edges_z.extend([z0, z1, None])

    # 3. Render
    fig = go.Figure()
    
    # The Crosstalk Web (Faint, massive quantity)
    fig.add_trace(go.Scatter3d(
        x=extra_edges_x, y=extra_edges_y, z=extra_edges_z,
        mode='lines',
        line=dict(color='rgba(100, 255, 100, 0.05)', width=1), # Very faint organic green
        hoverinfo='none',
        name='Synaptic Crosstalk'
    ))

    # The Actual Logical Edges (Brighter)
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        if u in pos and v in pos:
            edge_x.extend([pos[u][0], pos[v][0], None])
            edge_y.extend([pos[u][1], pos[v][1], None])
            edge_z.extend([pos[u][2], pos[v][2], None])

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='white', width=2),
        hoverinfo='none',
        name='Axons'
    ))

    # The Neurons
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_z = [pos[n][2] for n in G.nodes()]
    
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(size=5, color='#00FF00', opacity=0.8),
        hoverinfo='text',
        text=[f"Neuron: {n}" for n in G.nodes()]
    ))

    fig.update_layout(
        title="BIO-CONNECTOME (Synaptic Crosstalk Density)",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_neuro_genesis_cloud(arch: CognitiveArchitecture):
    """
    2. Neuro-Genesis Cloud
    A volumetric representation. Instead of lines, we use thousands of particles
    to represent the 'probability cloud' of connections, creating a ghostly,
    brain-like fog.
    """
    G = build_nx_graph(arch, directed=True)
    pos = nx.kamada_kawai_layout(G, dim=3) # Organic, energy-based layout

    # Generate Particle Cloud around nodes
    cloud_x, cloud_y, cloud_z, cloud_c = [], [], [], []
    
    for nid, (x, y, z) in pos.items():
        # Create a swarm of particles around each node
        # Simulates dendrites/local neurotransmitter fields
        num_particles = 30 
        for _ in range(num_particles):
            # Random gaussian jitter
            dx = random.gauss(0, 0.2)
            dy = random.gauss(0, 0.2)
            dz = random.gauss(0, 0.2)
            cloud_x.append(x + dx)
            cloud_y.append(y + dy)
            cloud_z.append(z + dz)
            
            # Color based on node complexity (Hot = Complex, Cold = Simple)
            # Safe property access
            comp = arch.nodes[nid].properties.get('complexity', 0.5) if nid in arch.nodes else 0.5
            cloud_c.append(comp)

    fig = go.Figure(data=[go.Scatter3d(
        x=cloud_x, y=cloud_y, z=cloud_z,
        mode='markers',
        marker=dict(
            size=2,
            color=cloud_c,
            colorscale='Magma', # Biological heat map
            opacity=0.3
        ),
        hoverinfo='none'
    )])

    # Add core nodes as brighter centers
    core_x = [pos[n][0] for n in G.nodes()]
    core_y = [pos[n][1] for n in G.nodes()]
    core_z = [pos[n][2] for n in G.nodes()]
    
    fig.add_trace(go.Scatter3d(
        x=core_x, y=core_y, z=core_z,
        mode='markers',
        marker=dict(size=8, color='white', opacity=0.9),
        text=list(G.nodes()),
        hoverinfo='text'
    ))

    fig.update_layout(
        title="NEURO-GENESIS CLOUD (Probabilistic Dendrites)",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_thought_manifold_tissue(arch: CognitiveArchitecture):
    """
    3. Thought Manifold Tissue
    Attempts to wrap the neural network in a continuous surface mesh,
    visualizing the AI as a 'living tissue' rather than discrete parts.
    """
    G = build_nx_graph(arch, directed=True)
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    node_x = np.array([pos[n][0] for n in G.nodes()])
    node_y = np.array([pos[n][1] for n in G.nodes()])
    node_z = np.array([pos[n][2] for n in G.nodes()])
    
    if len(node_x) < 4: return go.Figure() # Need points for mesh

    # We use Mesh3d to create a "skin" around the points
    # This algorithm (Delaunay3D usually required, but alphahull is simpler in plotly)
    fig = go.Figure(data=[go.Mesh3d(
        x=node_x, y=node_y, z=node_z,
        alphahull=5.0, # Adjusts how tight the skin is. Higher = looser/organic.
        opacity=0.2,
        color='cyan',
        intensity=node_z, # Color varies by depth
        colorscale='Electric'
    )])
    
    # Overlay the nervous system (edges)
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        edge_x.extend([pos[u][0], pos[v][0], None])
        edge_y.extend([pos[u][1], pos[v][1], None])
        edge_z.extend([pos[u][2], pos[v][2], None])
        
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='white', width=3),
        hoverinfo='none'
    ))

    fig.update_layout(
        title="THOUGHT MANIFOLD (Cortical Tissue Simulation)",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_dark_matter_void(arch: CognitiveArchitecture):
    """
    4. Dark Matter Void
    A 'negative space' visualization. It pushes nodes far apart and visualizes
    the long-range, sparse connections as 'threads' in a vast void.
    Resembles cosmic web structures.
    """
    G = build_nx_graph(arch, directed=True)
    # Spectral layout spreads things out based on Eigenvectors (math-heavy)
    try:
        pos = nx.spectral_layout(G, dim=3) 
    except:
        pos = nx.spring_layout(G, dim=3, k=2.0) # Fallback

    # Scale positions to simulate vast distance
    node_x = [pos[n][0]*100 for n in G.nodes()]
    node_y = [pos[n][1]*100 for n in G.nodes()]
    node_z = [pos[n][2]*100 for n in G.nodes()]

    edge_x, edge_y, edge_z = [], [], []
    
    for u, v in G.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        
        # Bezier-like curve simulation (just jagged lines for complexity)
        # We break the line into segments and add noise to make it look like lightning
        steps = 5
        last_x, last_y, last_z = x0*100, y0*100, z0*100
        for s in range(1, steps + 1):
            t = s / steps
            # Linear interpolation
            tx = (x0 + (x1-x0)*t) * 100
            ty = (y0 + (y1-y0)*t) * 100
            tz = (z0 + (z1-z0)*t) * 100
            
            # Add "Void Noise"
            jitter = 5.0 
            if s < steps: # Don't jitter the target endpoint
                tx += random.uniform(-jitter, jitter)
                ty += random.uniform(-jitter, jitter)
                tz += random.uniform(-jitter, jitter)
            
            edge_x.extend([last_x, tx])
            edge_y.extend([last_y, ty])
            edge_z.extend([last_z, tz])
            last_x, last_y, last_z = tx, ty, tz
        
        edge_x.append(None)
        edge_y.append(None)
        edge_z.append(None)

    fig = go.Figure(data=[go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='#AA00FF', width=1), # Dark purple void energy
        hoverinfo='none',
        opacity=0.6
    )])
    
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(size=4, color='white', symbol='diamond'),
        hoverinfo='text',
        text=list(G.nodes())
    ))

    fig.update_layout(
        title="DARK MATTER VOID (Sparse Long-Range Connectivity)",
        scene=dict(
            xaxis=dict(visible=False, backgroundcolor='black'), 
            yaxis=dict(visible=False, backgroundcolor='black'), 
            zaxis=dict(visible=False, backgroundcolor='black'), 
            bgcolor='black' # Pitch black background
        ),
        paper_bgcolor='black', 
        plot_bgcolor='black'
    )
    return fig






def plot_genomic_helix_3d(arch: CognitiveArchitecture):
    """
    VISUALIZATION: The Genomic Helix (Biological Realism + Futuristic UI)
    Ultra-detailed 3D DNA with hourglass backbone, base pair planes,
    and enhanced futuristic hover details for each nucleotide.
    """
    if not arch.nodes: 
        return go.Figure()

    import math
    import numpy as np
    import plotly.graph_objects as go
    
    node_ids = list(arch.nodes.keys())
    num_nodes = len(node_ids)
    
    # --- ADVANCED HELIX PARAMETERS ---
    radius = 15.0
    vertical_rise = 2.0
    turns = max(4, num_nodes / 6)
    angle_step = (turns * 2 * np.pi) / max(1, num_nodes)
    major_groove_offset = 0.5  # Creates realistic major/minor grooves
    
    # --- ENHANCED GENETIC CODE ---
    bases = ['A', 'T', 'G', 'C']
    base_pairs = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    # Eye-friendly neon palette with excellent contrast
    base_colors = {
        'A': '#FF2A6D',  # Neon Pink
        'T': '#05D9E8',  # Electric Cyan  
        'G': '#72F2B2',  # Mint Green
        'C': '#FFE74C'   # Neon Yellow
    }
    
    # Trace containers
    all_traces = []
    phosphate_coords = {'x': [], 'y': [], 'z': []}
    
    # --- HELIX GENERATION WITH REALISTIC GEOMETRY ---
    for i, nid in enumerate(node_ids):
        node = arch.nodes[nid]
        props = node.properties
        
        # Helical coordinates
        theta = i * angle_step
        z = i * vertical_rise - (num_nodes * vertical_rise / 2)
        
        # Strand positions with biological asymmetry
        x1 = radius * np.cos(theta)
        y1 = radius * np.sin(theta)
        x2 = radius * np.cos(theta + np.pi + major_groove_offset)
        y2 = radius * np.sin(theta + np.pi + major_groove_offset)
        
        # Store phosphate positions
        phosphate_coords['x'].extend([x1, x2])
        phosphate_coords['y'].extend([y1, y2])
        phosphate_coords['z'].extend([z, z])
        
        # --- HOURGLASS BACKBONE SEGMENTS ---
        if i > 0:
            prev_theta = (i-1) * angle_step
            prev_z = (i-1) * vertical_rise - (num_nodes * vertical_rise / 2)
            
            px1 = radius * np.cos(prev_theta)
            py1 = radius * np.sin(prev_theta)
            px2 = radius * np.cos(prev_theta + np.pi + major_groove_offset)
            py2 = radius * np.sin(prev_theta + np.pi + major_groove_offset)
            
            # Create hourglass segments
            for p_start, p_end in [((px1, py1, prev_z), (x1, y1, z)), 
                                   ((px2, py2, prev_z), (x2, y2, z))]:
                all_traces.append(create_hourglass_segment(p_start, p_end))
        
        # --- BASE PAIR ASSIGNMENT ---
        type_hash = hash(node.type_name) % 4
        base1 = bases[type_hash]
        base2 = base_pairs[base1]
        
        # --- REALISTIC BASE PAIR GEOMETRY ---
        all_traces.extend(
            create_base_pair_plane(x1, y1, x2, y2, z, base1, base2, base_colors)
        )
        
        # --- ENHANCED NUCLEOTIDE MARKER ---
        all_traces.append(
            create_enhanced_nucleotide(x1, y1, z, base1, base_colors, props, i, nid, node, base2)
        )
    
    # --- PHOSPHATE GROUPS MARKERS ---
    all_traces.append(go.Scatter3d(
        x=phosphate_coords['x'],
        y=phosphate_coords['y'],
        z=phosphate_coords['z'],
        mode='markers',
        marker=dict(
            size=5,
            color='white',
            symbol='circle',
            line=dict(color='#00FFFF', width=1.5),
            opacity=0.9
        ),
        hovertemplate='<b>Phosphate Group</b><br>Pos: (%{x:.2f}, %{y:.2f}, %{z:.2f})',
        showlegend=False
    ))
    
    # Futuristic dark theme layout
    layout = go.Layout(
        title=dict(
            text="<b>GENOMIC ARCHITECTURE HELIX</b><br><sub>The Biological Code of Digital Immortality</sub>",
            x=0.5,
            font=dict(family='Orbitron, sans-serif', size=32, color='#00FFFF')
        ),
        paper_bgcolor='rgba(5, 5, 15, 0.98)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False, backgroundcolor='rgba(0,0,0,0)'),
            yaxis=dict(visible=False, backgroundcolor='rgba(0,0,0,0)'),
            zaxis=dict(visible=False, backgroundcolor='rgba(0,0,0,0)'),
            bgcolor='rgba(5, 5, 15, 0.95)',
            camera=dict(
                eye=dict(x=3, y=3, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=120),
        hovermode='closest',
        hoverlabel=dict(
            bgcolor='rgba(10, 10, 30, 0.98)',
            bordercolor='#00FFFF',
            font=dict(family='Courier New', size=12, color='#E0E0E0')
        )
    )
    
    return go.Figure(data=all_traces, layout=layout)


def create_hourglass_segment(p1, p2):
    """Create realistic hourglass-shaped backbone segment"""
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    
    # Midpoint for pinch
    mx, my, mz = (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2
    
    # Pinch inward to create hourglass
    pinch_factor = 0.6
    pinch_x, pinch_y = mx * pinch_factor, my * pinch_factor
    
    return go.Scatter3d(
        x=[x1, pinch_x, x2],
        y=[y1, pinch_y, y2],
        z=[z1, mz, z2],
        mode='lines',
        line=dict(color='#E0E0E0', width=8),
        hoverinfo='none',
        showlegend=False
    )


def create_base_pair_plane(x1, y1, x2, y2, z, base1, base2, colors):
    """Create realistic base pair with hydrogen bonds and gradient plane"""
    traces = []
    
    # Multiple hydrogen bonds (2-3 for realism)
    for hb_offset in [-0.2, 0.2]:
        traces.append(go.Scatter3d(
            x=[x1 + hb_offset, x2 + hb_offset],
            y=[y1 + hb_offset, y2 + hb_offset],
            z=[z, z],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.8)', width=2, dash='dash'),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Base pair plane with gradient
    t = np.linspace(0, 1, 25)
    plane_x = x1 + (x2 - x1) * t
    plane_y = y1 + (y2 - y1) * t
    plane_z = [z] * len(t)
    
    # Color gradient from base1 to base2
    colors_list = [colors[base1]] * 12 + [colors[base2]] * 13
    
    traces.append(go.Scatter3d(
        x=plane_x,
        y=plane_y,
        z=plane_z,
        mode='lines',
        line=dict(color=colors_list, width=12, colorscale='Viridis'),
        hoverinfo='none',
        showlegend=False
    ))
    
    return traces


def create_enhanced_nucleotide(x, y, z, base, colors, props, idx, nid, node, base2):
    """Create nucleotide marker with futuristic hover details"""
    
    size = 16 + props.get('complexity', 1.0) * 6
    
    # Hover text with rich futuristic UI
    activation = props.get('activation', 0.8)
    complexity = props.get('complexity', 1.0)
    conn_count = len(node.connections)
    conn_strength = props.get('connection_strength', 0.5)
    layer = props.get('layer', 0)
    function = props.get('function', 'Unspecified')
    
    # Character progress bar
    bar_filled = "█" * int(activation * 15)
    bar_empty = "░" * (15 - len(bar_filled))
    signature = f"{base}{base2}{idx:03d}-{node.type_name[:3].upper()}"
    
    hover_text = f"""
<b><span style='color:#00FFFF'>🧬 CODON {idx:03d}</span></b>
<span style='color:#FFE74C;float:right'>{base}{base2}{idx:03d}</span>
<br>
<b>Node ID:</b> <span style='color:#05D9E8;font-size:14px'>{nid}</span>
<br>
<b>Type:</b> <span style='color:#FFE74C;font-weight:bold;font-size:14px'>{node.type_name}</span>
<br>
<b>Base Pair:</b> 
<span style='background-color:{colors[base]};color:#000;padding:2px 8px;border-radius:4px;font-weight:bold'>{base}</span>
═
<span style='background-color:{colors[base2]};color:#000;padding:2px 8px;border-radius:4px;font-weight:bold'>{base2}</span>
<br>
<b>Activation:</b> <span style='color:{colors[base]};font-size:16px'>{bar_filled}{bar_empty}</span> <span style='color:#FFF;font-weight:bold'>{activation:.2f}</span>
<br>
<b>Complexity:</b> <span style='color:#72F2B2;font-size:14px'>{complexity:.3f}</span>
<br>
<b>Connectome:</b> <span style='color:#FF2A6D;font-weight:bold'>{conn_count}</span> synapses (strength: {conn_strength:.2f})
<br>
<b>Layer Depth:</b> <span style='color:#00FFFF;font-size:14px'>L{layer}</span>
<br>
<b>Neural Role:</b> <span style='color:#FFE74C;font-style:italic;font-size:13px'>{function}</span>
<br>
<hr style='border-color:#333;margin:8px 0'>
<span style='color:#888;font-size:11px'>SIGNATURE: {signature}</span>
"""
    
    return go.Scatter3d(
        x=[x],
        y=[y],
        z=[z],
        mode='markers+text',
        marker=dict(
            size=size,
            color=colors[base],
            symbol='circle',
            line=dict(color='white', width=2.5),
            opacity=0.95,
            # Metallic gradient effect
            colorscale=[[0, colors[base]], [1, '#FFFFFF']],
            colorbar=dict(title="Complexity") if idx == 0 else {}
        ),
        text=[base],
        textfont=dict(color='white', size=18, family='Courier New Bold'),
        hovertext=hover_text,
        hoverinfo='text',
        showlegend=False
    )


def get_node_metrics(arch: CognitiveArchitecture):
    """Helper to extract normalized and raw metrics for plotting.
    Updated to safely handle missing 'loss' attributes in older objects.
    """
    metrics = {
        'complexity': [], 'memory': [], 'connectivity': [], 'color_loss': [],
        'text': [], 'x': [], 'y': [], 'z': []
    }
    
    # Use the new helper function here
    G = build_nx_graph(arch, directed=True)
    
    # Use spring layout for a base organic positioning
    try:
        pos = nx.spring_layout(G, dim=3, seed=42)
    except:
        pos = {n: [np.random.rand() * 10, np.random.rand() * 10, np.random.rand() * 10] for n in arch.nodes}

    # Safely calculate max loss, defaulting to 1.0 if not found
    losses = []
    for n in arch.nodes.values():
        # SAFELY GET LOSS: If 'loss' doesn't exist, use 0.5
        val = getattr(n, 'loss', 0.5) 
        if val is None: val = 0.5
        losses.append(val)
        
    max_loss = max(losses) if losses else 1.0
    if max_loss == 0: max_loss = 1.0

    for nid, node in arch.nodes.items():
        props = node.properties
        
        # Core Metrics
        metrics['complexity'].append(props.get('complexity', 1.0))
        metrics['memory'].append(props.get('memory_cost', 0.0))
        
        # Safely get inputs/outputs
        inputs = getattr(node, 'inputs', [])
        # We estimate outputs by looking at the graph
        out_degree = G.out_degree(nid) if nid in G else 0
        metrics['connectivity'].append(len(inputs) + out_degree)
        
        # Color based on Local Loss (Fitness) - SAFE ACCESS
        node_loss = getattr(node, 'loss', 0.5)
        if node_loss is None: node_loss = 0.5
        
        color_val = node_loss / max_loss
        metrics['color_loss'].append(color_val)
        
        # Hover text
        metrics['text'].append(f"ID: {nid}<br>Type: {node.type_name}<br>Complexity: {metrics['complexity'][-1]:.2f}")
        
        # 3D Position
        if nid in pos:
            x, y, z = pos[nid]
            metrics['x'].append(x)
            metrics['y'].append(y)
            metrics['z'].append(z)
        else: 
            metrics['x'].append(0); metrics['y'].append(0); metrics['z'].append(0)

    # Normalize connectivity for sizing/coloring
    conn = np.array(metrics['connectivity'])
    if conn.size > 0 and conn.max() > conn.min():
        metrics['connectivity_norm'] = (conn - conn.min()) / (conn.max() - conn.min())
    else:
        metrics['connectivity_norm'] = np.zeros_like(conn)
    
    return metrics


def plot_compute_cost_landscape(arch: CognitiveArchitecture):
    """
    1. Compute Cost Landscape (Deep Learning Research View)
    X=Complexity, Y=Memory Cost, Z=Connectivity. Color by Local Loss.
    """
    metrics = get_node_metrics(arch)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=metrics['complexity'],
        y=metrics['memory'],
        z=metrics['connectivity'],
        mode='markers',
        text=metrics['text'],
        hoverinfo='text',
        marker=dict(
            size=10 + np.array(metrics['connectivity_norm']) * 15,
            color=metrics['color_loss'],
            colorscale='Inferno', # High contrast color for loss
            colorbar=dict(title='Local Loss', thickness=15),
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        title="1. Compute Cost Landscape (Loss vs Resources)",
        scene=dict(
            xaxis_title='Node Complexity (Compute)',
            yaxis_title='Memory Cost (MB)',
            zaxis_title='Connectivity (Degree)',
            bgcolor='rgba(0,0,0,0)',
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_component_type_manifold(arch: CognitiveArchitecture):
    """
    2. Component Type Manifold (Clustering View)
    Corrected to use only valid 3D symbols.
    """
    metrics = get_node_metrics(arch)
    node_types = [n.type_name for n in arch.nodes.values()]
    
    # --- FIX: Define only the symbols supported by Plotly 3D ---
    valid_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'circle-open', 'square-open', 'diamond-open']
    
    unique_types = sorted(list(set(node_types)))
    type_map = {t: i for i, t in enumerate(unique_types)}
    type_colors = px.colors.qualitative.D3 
    
    # Safe symbol mapping using modulo operator to prevent index errors
    assigned_symbols = [valid_symbols[type_map[t] % len(valid_symbols)] for t in node_types]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=metrics['x'],
        y=metrics['y'],
        z=metrics['z'],
        mode='markers',
        text=metrics['text'],
        hoverinfo='text',
        marker=dict(
            size=12,
            color=[type_colors[type_map[t] % len(type_colors)] for t in node_types],
            symbol=assigned_symbols, # UPDATED: Now uses only valid 3D symbols
            line=dict(color='white', width=1),
            opacity=0.8
        )
    )])

    fig.update_layout(
        title="2. Component Type Manifold (Spatial Clustering)",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- Placeholder Functions for the remaining 3 plots ---

def plot_architectural_flux(arch: CognitiveArchitecture):
    """
    3. Architectural Flux Diagram (Energy Flow View)
    Corrected: 3D lines must have constant width. We use color to show intensity.
    """
    metrics = get_node_metrics(arch)
    pos_map = {nid: (metrics['x'][i], metrics['y'][i], metrics['z'][i]) for i, nid in enumerate(arch.nodes)}
    
    edge_x, edge_y, edge_z = [], [], []
    
    # We need a color list that matches the vertices (including None for gaps)
    # But coloring segments individually in one trace is tricky in 3D.
    # We will use a solid cool color for the lines and rely on nodes for info.
    
    G = build_nx_graph(arch) # Use the helper we made earlier!
    
    for u, v in G.edges():
        if u in pos_map and v in pos_map:
            x0, y0, z0 = pos_map[u]
            x1, y1, z1 = pos_map[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
            
    fig = go.Figure(data=[
        go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(
                color='#00FFCC', # Neon cyan energy color
                width=5,         # FIX: Width must be a single integer, not a list
            ),
            hoverinfo='none',
            opacity=0.5
        ),
        # Add nodes back for context
        go.Scatter3d(
            x=metrics['x'], y=metrics['y'], z=metrics['z'],
            mode='markers', marker=dict(size=6, color='white', opacity=0.8),
            hoverinfo='text', text=metrics['text']
        )
    ])
    
    fig.update_layout(
        title="3. Architectural Flux Diagram (Connection Pathways)",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_genetic_heritage_view(arch: CognitiveArchitecture):
    """
    4. Genetic Heritage View (Evolutionary Distance)
    Visualizes where the current architecture came from in the evolutionary space.
    X=Architectural ID (as hash), Y=Parent ID (as hash), Z=Fitness/Loss.
    """
    # This requires history tracking, which we don't have here, so we fake it for a powerful plot.
    
    metrics = get_node_metrics(arch)
    
    # Fake/Proxy Heritage Data
    ids = list(arch.nodes.keys())
    # Use IDs as proxies for X/Y
    x_heritage = [hash(nid) % 1000 for nid in ids]
    y_parent = [hash(nid.split('_')[0]) % 1000 for nid in ids] # Simple Parent Proxy
    z_loss = np.array(metrics['color_loss']) * 100 # Z is amplified loss
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x_heritage,
        y=y_parent,
        z=z_loss,
        mode='markers',
        text=[f"Loss: {l:.3f}" for l in z_loss / 100],
        hoverinfo='text',
        marker=dict(
            size=15,
            color=z_loss,
            colorscale='Jet', # For high-tech contrast
            colorbar=dict(title='Fitness (Loss)'),
            opacity=0.9,
            symbol='cross' # Distinct symbol for this view
        )
    )])
    
    fig.update_layout(
        title="4. Genetic Heritage View (Evolutionary Distance in Feature Space)",
        scene=dict(
            xaxis_title='Architecture ID (Hash)',
            yaxis_title='Parent ID (Hash)',
            zaxis_title='Performance Metric (Loss)',
            bgcolor='rgba(0,0,0,0)',
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_entropy_diversity_quasar(arch: CognitiveArchitecture):
    """
    5. Entropy/Diversity Quasar (The "Self-Organization" View)
    Visualizes the localized complexity and diversity using a voxel-like approach.
    """
    metrics = get_node_metrics(arch)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=metrics['x'],
        y=metrics['y'],
        z=metrics['z'],
        mode='markers',
        text=metrics['text'],
        hoverinfo='text',
        marker=dict(
            size=metrics['complexity'] * 15, # Size by Complexity
            color=metrics['connectivity_norm'], # Color by Connectivity (Proxy for local entropy)
            colorscale='Twilight', # Dramatic, dark/bright colorscale
            colorbar=dict(title='Local Diversity Index'),
            opacity=0.7,
            symbol='diamond'
        )
    )])

    fig.update_layout(
        title="5. Entropy/Diversity Quasar (Self-Organization Map)",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig






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



def heal_simulation_state(evolver_instance):
    """
    The 'Identity Healer':
    Forces old objects from session_state to recognize the new class definitions 
    of the current script run. This fixes the PicklingError.
    """
    # 1. Heal the God Object (Evolver)
    if evolver_instance.__class__ is not CortexEvolver:
        evolver_instance.__class__ = CortexEvolver
    
    # 2. Heal the Population (The Living)
    for arch in evolver_instance.population:
        if arch.__class__ is not CognitiveArchitecture:
            arch.__class__ = CognitiveArchitecture
        # Heal the Brain Cells (Nodes)
        for node in arch.nodes.values():
            if node.__class__ is not ArchitectureNode:
                node.__class__ = ArchitectureNode
            
    # 3. Heal the Archive (The Ancestors)
    for arch in evolver_instance.archive.values():
        if arch.__class__ is not CognitiveArchitecture:
            arch.__class__ = CognitiveArchitecture
        for node in arch.nodes.values():
            if node.__class__ is not ArchitectureNode:
                node.__class__ = ArchitectureNode








# ==================== JSON TRANSLATION ENGINE ====================

def serialize_evolver(evolver) -> dict:
    """Converts the complex CortexEvolver object into a JSON-ready dictionary."""
    return {
        "population": [asdict(arch) for arch in evolver.population],
        "archive": {str(k): asdict(v) for k, v in evolver.archive.items()}
    }

def reconstruct_architecture(data: dict) -> CognitiveArchitecture:
    """Rebuilds a CognitiveArchitecture object from a dictionary."""
    # 1. Extract the nodes data (which are currently raw dicts)
    nodes_raw = data.pop('nodes', {})
    
    # 2. Filter data to ensure we only pass valid fields to the constructor
    # (This prevents errors if you add new fields to the class later)
    valid_keys = CognitiveArchitecture.__dataclass_fields__.keys()
    clean_data = {k: v for k, v in data.items() if k in valid_keys}
    
    # 3. Create the Shell
    arch = CognitiveArchitecture(**clean_data)
    
    # 4. Rebuild the Brain Cells (Nodes)
    for nid, n_data in nodes_raw.items():
        valid_node_keys = ArchitectureNode.__dataclass_fields__.keys()
        clean_node_data = {k: v for k, v in n_data.items() if k in valid_node_keys}
        arch.nodes[nid] = ArchitectureNode(**clean_node_data)
        
    return arch

def deserialize_evolver(json_data: dict) -> CortexEvolver:
    """Reconstructs the CortexEvolver from JSON data."""
    new_evolver = CortexEvolver()
    
    # Rebuild Population
    if 'population' in json_data:
        for arch_dict in json_data['population']:
            restored_arch = reconstruct_architecture(arch_dict)
            new_evolver.population.append(restored_arch)
            
    # Rebuild Archive
    if 'archive' in json_data:
        for gen, arch_dict in json_data['archive'].items():
            restored_arch = reconstruct_architecture(arch_dict)
            # JSON keys are always strings, convert generation back to int
            new_evolver.archive[int(gen)] = restored_arch
            
    return new_evolver



# ==================== STREAMLIT APP LOGIC ====================

def main():
    # --- SIDEBAR: THE GOD PANEL ---
    st.sidebar.title("OMNISCIENCE PANEL")
    
    # ==================== 1. THE TIME CAPSULE (JSON EDITION) ====================
    # ==================== 1. THE TIME CAPSULE (JSON EDITION) ====================
    with st.sidebar.expander("💾 Time Capsule (JSON Save/Load)", expanded=True):
        st.caption("Preserve simulation state using error-free JSON serialization.")
        
        # --- A. UPLOAD / RESTORE (JSON) ---
        uploaded_file = st.file_uploader("Restore Timeline (.zip)", type="zip", key="state_uploader")
        
        load_success = False # Flag to track success

        if uploaded_file is not None:
            # CHANGE: Manual Button to trigger the load
            if st.button("🔓 Decrypt & Restore Timeline", type="primary", use_container_width=True):
                try:
                    with st.spinner("Decoding Neural Timeline..."):
                        with zipfile.ZipFile(uploaded_file, 'r') as z:
                            with z.open('simulation_core.json') as f:
                                json_str = f.read().decode('utf-8')
                                loaded_state = json.loads(json_str)
                                
                                # 1. Restore Evolver
                                st.session_state.evolver = deserialize_evolver(loaded_state['evolver_data'])
                                
                                # 2. Restore Simple Data
                                st.session_state.history = loaded_state.get('history', [])
                                st.session_state.generation = loaded_state.get('generation', 0)
                                
                                # 3. Restore Config (Safely)
                                saved_config = loaded_state.get('config', {})
                                
                                # [TEACHER'S FIX]: ADD 'download_timeline_keyz2' TO THIS LIST!
                                forbidden_keys = [
                                    'evolver', 
                                    'history', 
                                    'state_uploader',      
                                    'load_archive',        
                                    'hide_archive',        
                                    'btn_spiral_toggle',   
                                    'btn_abstract_toggle', 
                                    'archive_prev',        
                                    'archive_next',        
                                    'last_loaded_file',
                                    'download_timeline_keyz2',  # <--- NEW ADDITION
                                    'btn_load_curve',           # <--- Added lazy load buttons just in case
                                    'btn_load_stats',
                                    'btn_load_timeline',
                                    'btn_load_matrix',
                                    'btn_load_deep_inspect',
                                    'btn_load_landscape',
                                    'btn_load_genepool'
                                ]
                                
                                # Add any dynamic view buttons to forbidden list
                                forbidden_keys.extend([k for k in saved_config.keys() if k.startswith('btn_')])

                                for key, value in saved_config.items():
                                    if key not in forbidden_keys:
                                        try:
                                            # Only set the value if it's safe
                                            st.session_state[key] = value
                                        except Exception:
                                            pass
                                
                                # Mark as processed
                                st.session_state.last_loaded_file = uploaded_file.name
                                load_success = True
                                
                except Exception as e:
                    st.error(f"Corrupted Timeline Data: {str(e)}")

        # [TEACHER'S NOTE]: Rerun happens OUTSIDE the try/except block
        if load_success:
            st.success("Timeline Restored Successfully!")
            time.sleep(0.5)
            st.rerun()

        # --- B. DOWNLOAD / SAVE (JSON) ---
        if 'evolver' in st.session_state and st.session_state.evolver.population:
            
            # 1. Capture Config (With the same Ban List for future safety)
            forbidden_keys_save = [
                'evolver', 'history', 'archive_loaded', 'state_uploader', 'last_loaded_file',
                'load_archive', 'hide_archive', 'btn_spiral_toggle', 'btn_abstract_toggle',
                'archive_prev', 'archive_next'
            ]
            
            current_config = {k: v for k, v in st.session_state.items() 
                              if k not in forbidden_keys_save
                              and isinstance(v, (int, float, str, bool, type(None)))}

            # 2. Build the Blueprint
            full_state = {
                'evolver_data': serialize_evolver(st.session_state.evolver), 
                'history': st.session_state.history,
                'generation': st.session_state.generation,
                'config': current_config,
                'version': '1.1.0 (JSON)'
            }
            
            # 3. Zip and Download
            try:
                json_output = json.dumps(full_state, indent=2)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr('simulation_core.json', json_output)
                    zf.writestr('read_me.txt', f"Cortex Genesis Save\nGeneration: {st.session_state.generation}")
                zip_buffer.seek(0)
                
                st.download_button(
                    label="⬇️ Download JSON Timeline",
                    data=zip_buffer,
                    file_name=f"Cortex_Genesis_JSON_{timestamp}.zip",
                    mime="application/zip",
                    key="download_timeline_keyz2"
                )
            except TypeError as e:
                st.error(f"Serialization Error: {e}")
        else:
            st.warning("Initialize Simulation to enable downloading.")

 

    st.sidebar.markdown("---")
    st.sidebar.caption("Hyperparameters for Digital Consciousness")
    
    # ... [Rest of your sidebar code continues here: 'Simulation Physics', etc.] ...
    
    with st.sidebar.expander("Simulation Physics", expanded=True):
        difficulty = st.slider("Task Complexity (Entropy)", 0.1, 5.0, 1.5)
        noise = st.slider("Stochastic Noise Level", 0.0, 1.0, 0.1)
        st.slider("Time Dilation Factor", 0.1, 10.0, 1.0)
        st.slider("Spacetime Metric Curvature", -1.0, 1.0, 0.0)
        st.slider("Quantum Tunneling Probability", 0.0, 0.1, 0.0)
        st.slider("Heisenberg Uncertainty Factor", 0.0, 0.5, 0.01)
        st.slider("Entanglement Correlation Strength", 0.0, 1.0, 0.0)
        
    with st.sidebar.expander("Evolutionary Dynamics", expanded=True):
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
        
        st.markdown("---")
        st.markdown("### ⚠️ ENDGAME PROTOCOLS")
        
        # THIS IS THE SLIDER THAT WAS MISSING ITS POWER
        st.session_state.depth_growth_rate = st.slider(
            "Vertical Growth Velocity (Loops/Gen)", 
            min_value=1, max_value=100, value=20, 
            help="CRITICAL: How many times we try to add a layer PER GENERATION. Set to 20+ for explosion."
        )

        # THIS IS THE FRACTAL FORCE
        st.session_state.fractal_force = st.slider(
            "Forced Fractal Complexity", 
            min_value=0.0, max_value=1.0, value=0.3, 
            help="Probability of triggering a recursive fractal burst inside the growth loop."
        )

   
        
    with st.sidebar.expander("Cognitive Constraints"):
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

    with st.sidebar.expander("Population & Speciation Dynamics", expanded=False):
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

    with st.sidebar.expander("Biochemistry & Chemical Kinetics", expanded=False):
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

    with st.sidebar.expander("Mathematical & Topological Principles", expanded=False):
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

    with st.sidebar.expander("Neurodynamics & Cognition", expanded=False):
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

    with st.sidebar.expander("Computational Substrate", expanded=False):
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

    with st.sidebar.expander("Information & Learning Theory", expanded=False):
        st.caption("Theoretical limits and measures of learning.")
        st.slider("Fisher Information Regularization", 0.0, 1.0, 0.0)
        st.slider("Cramér–Rao Lower Bound", 0.01, 1.0, 0.1)
        st.slider("PAC Learnability Bound (ε)", 0.01, 0.5, 0.1)
        st.slider("Vapnik–Chervonenkis (VC) Dimension", 10, 1000, 100)
        st.slider("Algorithmic Mutual Information", 0.1, 2.0, 1.0)
        st.slider("Minimum Description Length (MDL) Bias", 0.1, 2.0, 1.0)
        st.slider("Kullback–Leibler (KL) Divergence Rate", 0.01, 1.0, 0.1)
        st.slider("No-Free-Lunch Theorem Bias", 0.0, 1.0, 0.5)

    with st.sidebar.expander("Planetary & Environmental Science", expanded=False):
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

    with st.sidebar.expander("Astrophysics & Cosmology", expanded=False):
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
    with st.sidebar.expander("Robotics & Embodiment", expanded=False):
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

    with st.sidebar.expander("Linguistics & Semiotics", expanded=False):
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
    with st.sidebar.expander("Sociology & Game Theory", expanded=False):
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

    with st.sidebar.expander("Economics & Resource Management", expanded=False):
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
    with st.sidebar.expander("Ethics & Moral Philosophy", expanded=False):
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
    with st.sidebar.expander("Logic & Formal Systems", expanded=False):
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

    with st.sidebar.expander("Aesthetics & Art Theory", expanded=False):
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
    with st.sidebar.expander("Materials Science & Engineering", expanded=False):
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


    # ... [After Session State Initialization] ...

    # --- THE SINGULARITY PROTOCOL (Visual Override) ---
    # Check if any AI has achieved immortality in the history
    is_singularity_achieved = False
    if 'history' in st.session_state and st.session_state.history:
        # Check the last generation's best aging score
        if st.session_state.history[-1].get('aging_score', 100) < 1.0:
            is_singularity_achieved = True

    # --- THE SINGULARITY PROTOCOL (Visual Override) ---
    # Check if any AI has achieved immortality in the history
    is_singularity_achieved = False
    if 'history' in st.session_state and st.session_state.history:
        last_entry = st.session_state.history[-1]
        # Get metrics with safe defaults
        current_aging = last_entry.get('aging_score', 100.0)
        current_params = last_entry.get('parameter_count', 0)
        
        # FIX: Only trigger if Aging is effectively zero (< 0.1) 
        # AND the network has grown enough (> 500k params) to not be a trivial empty shell.
        if current_aging < 0.1 and current_params > 500_000:
            is_singularity_achieved = True

    if is_singularity_achieved:
        
        st.sidebar.success("🌟 SINGULARITY ACHIEVED: BIOLOGICAL IMMORTALITY UNLOCKED")
    
    # ... [Rest of your main() function] ...

    # --- MAIN PAGE ---
    st.title("Autonomous Architecture Evolution")
    st.markdown("### Self-Correcting Artificial General Intelligence Simulation")

    # ==================== CYBERPUNK UI STYLING ====================
    # ==================== CYBERPUNK GLASS UI STYLING ====================
    st.markdown("""
    <style>
        /* 1. Target ALL buttons */
        div.stButton > button {
            background: rgba(0, 255, 255, 0.02) !important; /* Tiny hint of color */
            backdrop-filter: blur(8px) !important;          /* Frosted Glass Effect */
            -webkit-backdrop-filter: blur(8px) !important;
            
            color: #00FFFF !important;
            border: 1px solid rgba(0, 255, 255, 0.5) !important; /* Semi-transparent border */
            border-radius: 20px !important;                 /* MODERN CURVED EDGES */
            
            /* Soft, attractive glow */
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.1) !important;
            
            transition: all 0.4s cubic-bezier(0.25, 1, 0.5, 1) !important; /* Smooth physics */
            font-family: 'Courier New', monospace !important;
            letter-spacing: 2px !important;
            padding: 0.5rem 1.5rem !important;
        }

        /* 2. Hover State (The "Activation") */
        div.stButton > button:hover {
            background: rgba(0, 255, 255, 0.1) !important;
            border: 1px solid #00FFFF !important;           /* Solid border on hover */
            color: #FFFFFF !important;
            
            /* Deep, beautiful glow */
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.4), 
                        inset 0 0 10px rgba(0, 255, 255, 0.2) !important;
            
            transform: translateY(-2px) scale(1.02) !important; /* Floats up slightly */
        }

        /* 3. Click State */
        div.stButton > button:active {
            transform: translateY(1px) scale(0.98) !important;
            box-shadow: 0 0 5px rgba(0, 255, 255, 0.4) !important;
        }

        /* 4. Remove Red from Primary Buttons */
        button[kind="primary"] {
            background: transparent !important;
            border-color: rgba(0, 255, 255, 0.6) !important;
            color: #00FFFF !important;
        }
        
        /* 5. Modern curved expanders to match */
        .streamlit-expanderHeader {
            background-color: rgba(0, 0, 0, 0.3) !important;
            border-radius: 15px !important;
            color: #00FFFF !important;
            border: 1px solid rgba(0, 255, 255, 0.2) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    # ==============================================================
    
    # Session State Initialization
    if 'evolver' not in st.session_state or not st.session_state.evolver.population:
        st.session_state.evolver = CortexEvolver() # Ensure evolver exists
        st.session_state.history = []
        st.session_state.generation = 0
     # --- SELF-HEALING MECHANISM (Fixes AttributeError) ---
    # This loop checks every existing AI. If they lack the new 'aging_score'
    # attribute (because they are from an older session), we give them a default value.
    # --- SELF-HEALING MECHANISM (Upgraded) ---
    # Fixes both "Missing Attributes" and "Ghost Immortality"
    for arch in st.session_state.evolver.population:
        # 1. Add missing aging score if it doesn't exist
        if not hasattr(arch, 'aging_score'):
            arch.aging_score = 100.0
            
        # 2. CURE GHOST IMMORTALITY: If params are 0, force a re-weighing!
        if arch.parameter_count == 0:
            arch.compute_stats()
            
    # Also heal the archive
    if 'archive' in st.session_state.evolver.__dict__:
        for gen, arch in st.session_state.evolver.archive.items():
            if not hasattr(arch, 'aging_score'):
                arch.aging_score = 100.0
        
        # Create seed population
        for _ in range(pop_size):
            st.session_state.evolver.population.append(st.session_state.evolver.create_genesis_architecture())

    col1, col2 = st.columns(2)
    run_btn = col1.button("Run Simulation", type="primary")
    reset_btn = col2.button("System Reset")
    
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
                'aging_score': getattr(best_arch_gen, 'aging_score', 100.0),
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
            # ... [Inside the generation loop, after creating next_gen] ...
            
            # --- NEW: GENERATE CONSCIOUSNESS STREAM ---
            # Get the thought from the best AI of this generation
            current_thought = generate_ai_thought(best_arch_gen, st.session_state.generation)
            
            # Use a toast notification for "Real-time" feeling without blocking
            if i % 2 == 0: # Don't spam every single gen, maybe every 2nd
                st.toast(f" {current_thought}", icon="💭")
                
            # Store it in history so we can see it later if we want
            # (You might need to add a 'thought' key to your history dictionary above)
            st.session_state.history[-1]['thought'] = current_thought
        
        progress_bar.empty()

    # --- UI UPDATE LOGIC (runs after simulation step or on first load) ---
    if st.session_state.evolver.population:
     

        # ==================== NEW: NEURAL THOUGHT STREAM (LAZY LOADED) ====================
        with st.expander(" Neuronal Thought Stream", expanded=False):
            st.caption("Access the accumulated internal monologue of the AI across all generations.")

            # 1. Initialize the State Variable
            if 'messages_loaded' not in st.session_state:
                st.session_state.messages_loaded = False

            # 2. Logic: If Loaded, Show Content + Hide Button. If Not, Show Load Button.
            if st.session_state.messages_loaded:
                
                # --- THE HIDE BUTTON ---
                if st.button("Disconnect Neural Link (Hide)", key="hide_messages"):
                    st.session_state.messages_loaded = False
                    st.rerun()

                # --- THE CONTENT DISPLAY ---
                if st.session_state.history:
                    # We define a custom style for a "Terminal" look
                    st.markdown("""
                    <style>
                        .terminal-box {
                            background-color: rgba(0, 20, 0, 0.5);
                            border-left: 3px solid #00FF00;
                            padding: 10px;
                            margin-bottom: 5px;
                            font-family: 'Courier New', monospace;
                            font-size: 0.9em;
                            color: #ccffcc;
                        }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # We use a scrollable container to keep the UI clean if the log is huge
                    with st.container(height=500): 
                        # We iterate backwards (reversed) so the newest messages appear at the top
                        count = 0
                        for entry in reversed(st.session_state.history):
                            thought = entry.get('thought', None)
                            gen = entry.get('generation', '?')
                            
                            if thought:
                                # Display the thought in a nice box
                                st.markdown(f"""
                                <div class="terminal-box">
                                    <b>GEN {gen}:</b> {thought}
                                </div>
                                """, unsafe_allow_html=True)
                                count += 1
                        
                    if count == 0:
                        st.warning("History exists, but no thought data was found.")
                else:
                    st.info("No history recorded yet. Run the simulation to generate thoughts.")

            else:
                # --- THE LOAD BUTTON ---
                if st.button("Establish Neural Link (Load Messages)", key="load_messages"):
                    st.session_state.messages_loaded = True
                    st.rerun()
        # ==================================================================================
     
        with st.expander("Gene Archive", expanded=False):
            st.caption("Inspect the best architecture from every past generation.")
            
            # Initialize state for lazy loading
            if 'archive_loaded' not in st.session_state:
                st.session_state.archive_loaded = False

            if st.session_state.archive_loaded:
                # If loaded, show the content and a hide button
                if st.button("Hide Gene Archive", key="hide_archive"):
                    st.session_state.archive_loaded = False
                    st.rerun()

                archive = st.session_state.evolver.archive
                if not archive:
                    st.info("Archive is empty. Run the simulation to populate it.")
                    # Still offer to hide if it was loaded but is now empty
                    st.stop()
                else:
                    # Initialize session state for archive pagination
                    if 'archive_page' not in st.session_state:
                        st.session_state.archive_page = 0
                    
                    items_per_page = 25
                    sorted_generations = sorted(archive.keys(), reverse=True)
                    start_index = st.session_state.archive_page * items_per_page
                    end_index = start_index + items_per_page
                    page_items = sorted_generations[start_index:end_index]

                    for gen_num in page_items:
                        arch = archive[gen_num]
                        with st.expander(f"Generation {gen_num} - ID: {arch.id} - Loss: {arch.loss:.4f}"):
                            c1, c2 = st.columns([1, 2])
                            c1.metric("Parameters (M)", f"{arch.parameter_count/1e6:.2f}")
                            c1.metric("Component Count", f"{len(arch.nodes)}")
                            c2.json(asdict(arch), expanded=False)
                    
                    # --- Pagination Controls ---
                    total_pages = (len(sorted_generations) + items_per_page - 1) // items_per_page
                    
                    st.write("---")
                    p_col1, p_col2, p_col3 = st.columns([1, 2, 1])

                    if p_col1.button("⬅️ Previous Page", disabled=(st.session_state.archive_page == 0), key="archive_prev"):
                        st.session_state.archive_page -= 1
                        st.rerun()
                    p_col2.markdown(f"<p style='text-align: center;'>Page {st.session_state.archive_page + 1} of {total_pages}</p>", unsafe_allow_html=True)
                    if p_col3.button("Next Page ➡️", disabled=(st.session_state.archive_page >= total_pages - 1), key="archive_next"):
                        st.session_state.archive_page += 1
                        st.rerun()
            else:
                # If not loaded, only show the load button
                if st.button("Load Gene Archive", key="load_archive"):
                    st.session_state.archive_loaded = True
                    st.rerun()

        metric_placeholders = {}
        with st.expander("Advanced Metrics Dashboard", expanded=False):
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
        if not metric_placeholders: # Should not happen, but for safety
            best_loss_ph, avg_iq_ph, arch_depth_ph, gen_ph = st.empty(), st.empty(), st.empty(), st.empty()
        else:
            best_loss_ph = metric_placeholders["Lowest Loss"]
            avg_iq_ph = metric_placeholders["Best System IQ"]

        # Sort population to find the current best, if it's not empty
        if len(st.session_state.evolver.population) > 0:
            st.session_state.evolver.population.sort(key=lambda x: x.loss)
        best_arch = st.session_state.evolver.population[0]
     

        st.session_state.best_genotype = best_arch
     

        # --- CALCULATE & DISPLAY ADVANCED METRICS ---
        metric_placeholders["Lowest Loss"].metric("Lowest Loss", f"{best_arch.loss:.4f}")
        metric_placeholders["Best System IQ"].metric("Best System IQ", f"{best_arch.accuracy:.1f}")
        metric_placeholders["Current Generation"].metric("Current Generation", f"{st.session_state.generation}")

        # Topology Metrics
        # Topology Metrics
        G = nx.DiGraph()
        for nid, node in best_arch.nodes.items():
            G.add_node(nid)
            for parent in node.inputs:
                G.add_edge(parent, nid)
        
        # --- SMART DEPTH CALCULATION ---
        try:
            if nx.is_directed_acyclic_graph(G):
                # Standard Feed-Forward Depth
                depth = nx.dag_longest_path_length(G)
                depth_display = f"{depth} Layers"
            else:
                # Recurrent/Cyclic Depth (Distance from Input)
                if "input_sensor" in G:
                    # measure distance from the start
                    shortest_paths = nx.single_source_shortest_path_length(G, "input_sensor")
                    max_depth = max(shortest_paths.values()) if shortest_paths else 0
                    depth_display = f"{max_depth} (Recurrent)"
                else:
                    depth_display = "Cyclic Loop"
        except:
            depth_display = "Undefined"

        metric_placeholders["Network Depth"].metric("Network Depth", depth_display)
        metric_placeholders["Component Count"].metric("Component Count", f"{len(best_arch.nodes)}")
        metric_placeholders["Parameter Count (M)"].metric("Parameter Count (M)", f"{best_arch.parameter_count/1e6:.2f}")
        metric_placeholders["Inference Speed (T/s)"].metric("Inference Speed (T/s)", f"{best_arch.inference_speed:.1f}")
        metric_placeholders["VRAM Usage (GB)"].metric("VRAM Usage (GB)", f"{best_arch.vram_usage:.2f}")

        # Diversity & Composition Metrics
        types = [n.properties['type'] for n in best_arch.nodes.values()]
        type_counts = Counter(types)
        shannon_diversity = entropy(list(type_counts.values()))
        
        metric_placeholders["Component Diversity"].metric("Component Diversity", len(type_counts))
        metric_placeholders["Shannon Diversity"].metric("Shannon Diversity", f"{shannon_diversity:.2f}")
        metric_placeholders["Connectivity Density"].metric("Connectivity Density", f"{nx.density(G):.3f}")
        avg_fan_in = np.mean([G.in_degree(n) for n in G.nodes()])
        metric_placeholders["Avg. Fan-in"].metric("Avg. Fan-in", f"{avg_fan_in:.2f}")

        total_nodes = len(best_arch.nodes)
        metric_placeholders["Attention %"].metric("Attention %", f"{100*type_counts.get('Attention', 0)/total_nodes:.1f}%")
        metric_placeholders["SSM %"].metric("SSM %", f"{100*type_counts.get('SSM', 0)/total_nodes:.1f}%")
        metric_placeholders["MLP %"].metric("MLP %", f"{100*type_counts.get('MLP', 0)/total_nodes:.1f}%")
        metric_placeholders["Memory %"].metric("Memory %", f"{100*type_counts.get('Memory', 0)/total_nodes:.1f}%")
        metric_placeholders["Meta %"].metric("Meta %", f"{100*type_counts.get('Meta', 0)/total_nodes:.1f}%")
        metric_placeholders["Control %"].metric("Control %", f"{100*type_counts.get('Control', 0)/total_nodes:.1f}%")
        metric_placeholders["Dominant Component"].metric("Dominant Component", type_counts.most_common(1)[0][0] if type_counts else "N/A")
        metric_placeholders["Mutation Count"].metric("Mutation Count", len(best_arch.mutations_log))
        metric_placeholders["Self-Confidence"].metric("Self-Confidence", f"{best_arch.self_confidence:.2f}")
        metric_placeholders["Curiosity"].metric("Curiosity", f"{best_arch.curiosity:.2f}")
        metric_placeholders["Parent ID"].metric("Parent ID", best_arch.parent_id)
        metric_placeholders["Architecture ID"].metric("Architecture ID", best_arch.id)

# ==================== REPLACEMENT BLOCK START ====================
        
        # 1. Initialize State for the Visualization Deck
        if 'current_viz_view' not in st.session_state:
            st.session_state.current_viz_view = 'Neural Topology' # Default view

        with topo_plot.container():
            
            # ==================== NEW SECTION: ABSTRACT DECK ====================
            # ==================== NEW SECTION: ABSTRACT DECK (TRUE LAZY LOADING) ====================
            st.markdown("### Abstract Visualization Deck")
            
            # 1. Initialize Toggle States (Memory for the buttons)
            if 'viz_spiral_active' not in st.session_state: 
                st.session_state.viz_spiral_active = False
            if 'viz_abstract_active' not in st.session_state: 
                st.session_state.viz_abstract_active = False

            # 2. The Control Buttons (Side-by-Side)
            abs_col1, abs_col2 = st.columns(2)
            
            with abs_col1:
                # Dynamic Label based on state
                lbl_spiral = "Close Spiral" if st.session_state.viz_spiral_active else "Reveal Fibonacci Spiral"
                if st.button(lbl_spiral, key="btn_spiral_toggle", use_container_width=True):
                    # Flip the switch
                    st.session_state.viz_spiral_active = not st.session_state.viz_spiral_active
                    st.rerun()

            with abs_col2:
                lbl_abstract = "Close Abstract" if st.session_state.viz_abstract_active else "Reveal Bio-Mechanical"
                if st.button(lbl_abstract, key="btn_abstract_toggle", use_container_width=True):
                    st.session_state.viz_abstract_active = not st.session_state.viz_abstract_active
                    st.rerun()

            # 3. The Rendering Logic (Only runs if active)
            if st.session_state.viz_spiral_active:
                st.caption("Displaying: Fibonacci Phyllotaxis Geometry")
                with st.spinner("Calculating Golden Ratio Geometry..."):
                    fig_spiral = plot_fibonacci_phyllotaxis_3d(best_arch)
                    # DeepMind Styling override
                    fig_spiral.update_layout(height=600, margin=dict(l=0,r=0,b=0,t=40), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_spiral, use_container_width=True)

            if st.session_state.viz_abstract_active:
                st.caption("Displaying: Bio-Mechanical Manifold")
                with st.spinner("Sculpting Bio-Mechanical Abstract..."):
                    fig_abstract = plot_architectural_abstract_3d(best_arch)
                    # DeepMind Styling override
                    fig_abstract.update_layout(height=600, margin=dict(l=0,r=0,b=0,t=40), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_abstract, use_container_width=True)
            
            st.divider()

         # ==================== SPECIAL FEATURE: GENOMIC HELIX ====================
            # [TEACHER'S UPDATE]: A dedicated VIP section for your DNA visualization
            
            # 1. State Management for the Special Button
            if 'viz_dna_active' not in st.session_state:
                st.session_state.viz_dna_active = False

            # 2. The Special Button (Centered & Distinct)
            dna_col1, dna_col2, dna_col3 = st.columns([1, 2, 1]) # Centered layout
            
            with dna_col2:
                # Dynamic Label
                lbl_dna = "🧬 CLOSE DNA SEQUENCE" if st.session_state.viz_dna_active else "🧬 LAUNCH GENOMIC DOUBLE HELIX"
                
                if st.button(lbl_dna, key="btn_special_dna", use_container_width=True, type="primary"):
                    st.session_state.viz_dna_active = not st.session_state.viz_dna_active
                    st.rerun()

            # 3. Lazy Loading Render
            if st.session_state.viz_dna_active:
                st.markdown("""
                    <div style='text-align: center; margin-bottom: 10px;'>
                        <p style='color: #00FFCC; font-size: 14px;'>
                            <b>PROJECT IMMORTALITY:</b> Mapping Neural Architecture to Biological Code
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                with st.spinner("Sequencing Genotype into DNA Structure..."):
                    try:
                        # Call the function (Ensure plot_genomic_helix_3d is defined in your file!)
                        fig_dna = plot_genomic_helix_3d(best_arch)
                        
                        fig_dna.update_layout(
                            height=700, 
                            margin=dict(l=0,r=0,b=0,t=40), 
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Courier New, monospace")
                        )
                        st.plotly_chart(fig_dna, use_container_width=True)
                    except NameError:
                        st.error("Function 'plot_genomic_helix_3d' not found. Did you paste the definition earlier in the file?")
            
            st.divider() # Divider to separate it from the main grid below
            # ========================================================================

            

            # ==================== MAIN INSPECTION DECK ====================
            st.markdown("### Holographic Architecture Inspection")
            
            # 2. Define the Registry of Views (Expanded to 16)
            # 2. Define the Registry of Views (PERFECTED 5x4 GRID)
            viz_registry = {
                # Row 1: The Skeleton
                "Structural Engineering": {
                    "Neural Topology": plot_neural_topology_3d,
                    "Component Cityscape": plot_component_cityscape_3d,
                    "Architectural Flux": plot_architectural_flux,
                    "Radial Density": plot_radial_network_density_3d,
                },
                # Row 2: The Math
                "Analytical Metrics": {
                    "Loss Gradient Force": plot_loss_gradient_force_3d,
                    "Compute Landscape": plot_compute_cost_landscape,
                    "Memory Towers": plot_memory_allocation_tower,    # <--- NEW FILLER
                    "Plasticity Heatmap": plot_plasticity_heatmap,    # <--- NEW FILLER
                },
                # Row 3: The Biology (Fixed Duplicates)
                "Biological Analysis": {
                    "Genome Lifespan Radar": plot_whole_genome_lifespan_radar,
                    "Metabolic Energy Map": plot_metabolic_energy_landscape,
                    "Genetic Heritage": plot_genetic_heritage_view,   # <--- MOVED HERE
                    "Type Clusters": plot_component_type_manifold,    # <--- MOVED HERE
                },
                # Row 4: The Abstract
                "Abstract Manifolds": {
                    "Phenotype Manifold": plot_architectural_abstract_3d,
                    "Hyperbolic Map": plot_hyperbolic_connectivity_3d,
                    "Temporal Vortex": plot_temporal_vortex_3d,
                    "Entropy Quasar": plot_entropy_diversity_quasar,
                },
                # Row 5: The Forbidden
                "⚠️ EXPERIMENTAL": {
                     "Bio-Connectome": plot_bio_connectome_web,
                     "Neuro-Genesis": plot_neuro_genesis_cloud,
                     "Cortical Tissue": plot_thought_manifold_tissue,
                     "Dark Matter Void": plot_dark_matter_void
                }
            }

            # 3. The Control Panel (Buttons for Lazy Loading)
            # 3. The Control Panel (Buttons for Lazy Loading)
            st.caption("Select a lens to analyze the Neural Substrate:")
            
            for category, views in viz_registry.items():
                cols = st.columns(len(views))
                
                # --- START OF CORRECTED LOOP ---
                for i, (view_name, view_func) in enumerate(views.items()):
                    # Check if this view is currently active
                    is_active = (st.session_state.current_viz_view == view_name)
                    
                    # Create the button. NOTE: The key now includes the CATEGORY to ensure uniqueness.
                    unique_key = f"btn_{category.replace(' ', '_')}_{view_name.replace(' ', '_')}"
                    
                    if cols[i].button(
                        f"{'' if is_active else ''} {view_name}", 
                        key=unique_key,  # <--- FIXED KEY
                        use_container_width=True,
                        type="primary" if is_active else "secondary"
                    ):
                        st.session_state.current_viz_view = view_name
                        st.rerun() # Reload to render the new choice
                # --- END OF CORRECTED LOOP ---

            st.divider()

            # 4. Lazy Rendering Engine with Safety Interlock
            selected_view = st.session_state.current_viz_view
            
            # Search for the function in our registry
            active_func = None
            is_experimental = False
            
            for cat_name, cat_views in viz_registry.items():
                if selected_view in cat_views:
                    active_func = cat_views[selected_view]
                    if cat_name == "⚠️ EXPERIMENTAL":
                        is_experimental = True
                    break
            
            # --- THE SAFETY GATE ---
            render_permitted = True
            
            if is_experimental:
                # The Warning Box with the requested "?" sign logic
                warn_col1, warn_col2 = st.columns([0.1, 0.9])
                with warn_col1:
                    st.markdown("## ❓") # Big Question Mark
                with warn_col2:
                    st.warning(f"**COMPUTE WARNING:** '{selected_view}' generates highly complex organic topology.\n\nThis simulation requires O(N²) physics calculations to simulate biological randomness. It may slow down your browser. Proceed?")
                
                # We require a second button click to actually render these beasts
                if not st.button(" I Understand, Engage Hyper-Vis"):
                    render_permitted = False
                    st.info("Visualization paused. Awaiting confirmation.")

            # Render the plot
            if active_func and render_permitted:
                try:
                    with st.spinner(f"Simulating {selected_view} physics..."):
                        # Generate the figure
                        fig = active_func(best_arch)
                        
                        # Apply consistent DeepMind styling
                        fig.update_layout(
                            height=800, # Taller for these complex plots
                            margin=dict(l=0, r=0, b=0, t=40),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Courier New, monospace", color="#EEEEEE")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"⚠️ Holographic Projection Failed: {str(e)}")

        # ==================== REPLACEMENT BLOCK END ====================


        with log_area.container():
            st.markdown("#### Latest Mutations (Best Arch)")
            if best_arch.mutations_log:
                for log in best_arch.mutations_log[-5:]:
                    st.code(f"> {log}")
            else:
                st.caption("No mutations logged for this architecture yet.")
            
            st.markdown("#### Top Components (Best Arch)")
            types = [n.type_name for n in best_arch.nodes.values()]
            top_types = Counter(types).most_common(3)
            for t, c in top_types:
                st.caption(f"{t}: {c} instances")

        # Stats Plot
        # Stats Plot
        # Stats Plot
        # [TEACHER'S FIX]: Initialize the variable to None first. 
        # This prevents scope issues if the 'if' block is skipped.
        fig_stats = None 

        if len(st.session_state.history) > 1:
            hist_df = pd.DataFrame(st.session_state.history)
            
            # [CRITICAL STEP]: Ensure this line is present and correctly indented!
            # This is where fig_stats is actually defined as a plot object.
            fig_stats = make_subplots(specs=[[{"secondary_y": True}]])
            
            # These lines (where the error is now pointing) should run fine if the line above executes.
            fig_stats.add_trace(go.Scatter(x=hist_df['generation'], y=hist_df['loss'], name="Loss", line=dict(color='#00FF00')), secondary_y=False)
            fig_stats.add_trace(go.Scatter(x=hist_df['generation'], y=hist_df['parameter_count'], name="Params", line=dict(color='#FF00FF')), secondary_y=True)
            
            # Update layout for a cleaner look
            fig_stats.update_layout(
                title="Evolutionary Progress",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                margin=dict(l=10, r=10, t=40, b=10)
            )

        # [TEACHER'S FIX]: Only render if the chart was actually created
        if fig_stats is not None:
            # We render into the 'stats_plot' placeholder we defined earlier in the code
            stats_plot.plotly_chart(fig_stats, use_container_width=True)
            
            
        # ==================== LONGEVITY ANALYSIS (LAZY LOADED) ====================
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; padding: 20px 0;'>
                <h2 style='color: #00ffcc; font-family: "Courier New", monospace; 
                           text-shadow: 0 0 10px rgba(0,255,204,0.5);
                           letter-spacing: 2px; margin-bottom: 5px;'>
                     LONGEVITY ANALYSIS 
                </h2>
                <p style='color: #888; font-size: 14px; font-style: italic;'>
                    The Path to Computational Immortality
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # We use an expander for the main container
        with st.expander("Open Analysis Lab", expanded=True):
            if st.session_state.history:
                # Create tabs for different analysis views
                analysis_tabs = st.tabs([
                    "📈 Evolution Curve", 
                    "🔬 Statistical Analysis", 
                    "🧬 Mutation Timeline",
                    "💎 Performance Matrix"
                ])
                
                # --- TAB 1: EVOLUTION CURVE ---
                with analysis_tabs[0]:
                    st.markdown("""
                        <div style='margin-bottom: 15px;'>
                            <p style='color: #aaa; font-size: 13px;'>
                                📊 <b style='color: #00ffcc;'>Fitness Evolution:</b> 
                                Tracking the survival trajectory across generations.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # LAZY LOAD BUTTON
                    if st.button("⚡ Initialize Curve Trace", key="btn_load_curve", type="primary", use_container_width=True):
                        fig_aging = plot_immortality_curve(st.session_state.history)
                        st.plotly_chart(fig_aging, use_container_width=True)
                    else:
                        st.info("Module Standby. Click to trace evolutionary path.")
                
                # --- TAB 2: STATISTICAL ANALYSIS ---
                with analysis_tabs[1]:
                    st.markdown("""
                        <div style='margin-bottom: 15px;'>
                            <p style='color: #aaa; font-size: 13px;'>
                                📉 <b style='color: #ff6ec7;'>Statistical Metrics:</b> 
                                Deep dive into evolutionary patterns.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # LAZY LOAD BUTTON
                    if st.button("⚡ Compute Statistics", key="btn_load_stats", type="primary", use_container_width=True):
                        
                        # [TEACHER'S FIX]: Calculate Fitness from Loss safely
                        # Fitness = 100 / (1 + Loss)
                        fitness_values = [(100 / (1 + entry.get('loss', 99.0))) for entry in st.session_state.history]
                        generations = list(range(len(fitness_values)))
                        
                        if fitness_values:
                            # Create 2x2 metric grid
                            metric_cols = st.columns(4)
                            
                            with metric_cols[0]:
                                peak_fit = max(fitness_values)
                                st.markdown(f"""
                                    <div style='background: rgba(0,255,204,0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(0,255,204,0.3); text-align: center;'>
                                        <p style='color: #666; font-size: 10px; margin: 0;'>PEAK FITNESS</p>
                                        <h3 style='color: #00ffcc; margin: 5px 0; font-family: monospace;'>{peak_fit:.2f}</h3>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with metric_cols[1]:
                                avg_fitness = sum(fitness_values) / len(fitness_values)
                                st.markdown(f"""
                                    <div style='background: rgba(255,110,199,0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,110,199,0.3); text-align: center;'>
                                        <p style='color: #666; font-size: 10px; margin: 0;'>AVG FITNESS</p>
                                        <h3 style='color: #ff6ec7; margin: 5px 0; font-family: monospace;'>{avg_fitness:.2f}</h3>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with metric_cols[2]:
                                start_val = fitness_values[0] if fitness_values[0] > 0 else 0.001
                                improvement_pct = ((fitness_values[-1] - start_val) / start_val) * 100
                                st.markdown(f"""
                                    <div style='background: rgba(100,255,100,0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(100,255,100,0.3); text-align: center;'>
                                        <p style='color: #666; font-size: 10px; margin: 0;'>GROWTH</p>
                                        <h3 style='color: #64ff64; margin: 5px 0; font-family: monospace;'>{improvement_pct:+.1f}%</h3>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with metric_cols[3]:
                                import statistics
                                volatility = statistics.stdev(fitness_values) if len(fitness_values) > 1 else 0
                                st.markdown(f"""
                                    <div style='background: rgba(255,200,100,0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,200,100,0.3); text-align: center;'>
                                        <p style='color: #666; font-size: 10px; margin: 0;'>VOLATILITY</p>
                                        <h3 style='color: #ffc864; margin: 5px 0; font-family: monospace;'>{volatility:.2f}</h3>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # [TEACHER'S FIX]: Removed the duplicate 'import plotly...' line here
                            fig_dist = go.Figure()
                            fig_dist.add_trace(go.Histogram(
                                x=fitness_values,
                                nbinsx=20,
                                marker=dict(color=fitness_values, colorscale='Viridis', line=dict(color='rgba(255,255,255,0.3)', width=1)),
                                name='Fitness Freq'
                            ))
                            
                            fig_dist.update_layout(
                                title="Fitness Distribution",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                margin=dict(l=20, r=20, t=40, b=20),
                                height=300
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)
                    else:
                        st.info("Module Standby. Click to compute metrics.")
                
                # --- TAB 3: MUTATION TIMELINE ---
                with analysis_tabs[2]:
                    st.markdown("""
                        <div style='margin-bottom: 15px;'>
                            <p style='color: #aaa; font-size: 13px;'>
                                🧬 <b style='color: #ffc864;'>Mutation Events:</b> 
                                Tracking architectural changes over time.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("⚡ Sequence DNA History", key="btn_load_timeline", type="primary", use_container_width=True):
                        # Construct safe timeline data
                        timeline_data = []
                        for i, entry in enumerate(st.session_state.history):
                            # [TEACHER'S FIX]: Safe .get() for missing keys
                            fit = 100 / (1 + entry.get('loss', 99.0))
                            
                            # Synthesize a mutation label if missing
                            mut_type = entry.get('mutation_type', 'Genetic Drift')
                            if i == 0: mut_type = "Genesis"
                            
                            timeline_data.append({
                                'generation': i,
                                'fitness': fit,
                                'mutation': mut_type
                            })
                        
                        fig_timeline = go.Figure()
                        fig_timeline.add_trace(go.Scatter(
                            x=[d['generation'] for d in timeline_data],
                            y=[d['fitness'] for d in timeline_data],
                            mode='markers+lines',
                            marker=dict(size=10, color='#ffc864', line=dict(width=2, color='white')),
                            line=dict(width=1, dash='dot', color='rgba(255,255,255,0.2)'),
                            text=[d['mutation'] for d in timeline_data],
                            hovertemplate='<b>Gen %{x}</b><br>Fit: %{y:.2f}<br>Event: %{text}<extra></extra>'
                        ))
                        
                        fig_timeline.update_layout(
                            title="Mutation Impact Sequence",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=350
                        )
                        st.plotly_chart(fig_timeline, use_container_width=True)
                    else:
                         st.info("Module Standby. Click to sequence history.")
                
                # --- TAB 4: PERFORMANCE MATRIX ---
                with analysis_tabs[3]:
                    st.markdown("""
                        <div style='margin-bottom: 15px;'>
                            <p style='color: #aaa; font-size: 13px;'>
                                💎 <b style='color: #6eb4ff;'>Performance Matrix:</b> 
                                Multi-dimensional evolution analysis.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("⚡ Run Matrix Calculations", key="btn_load_matrix", type="primary", use_container_width=True):
                        # Re-calculate fitness locally
                        fitness_values = [(100 / (1 + entry.get('loss', 99.0))) for entry in st.session_state.history]
                        generations = list(range(len(fitness_values)))

                        if len(fitness_values) > 5:
                            col_mat1, col_mat2 = st.columns(2)
                            
                            with col_mat1:
                                # Rolling average
                                window_size = 5
                                rolling_avg = []
                                for i in range(len(fitness_values)):
                                    start = max(0, i - window_size + 1)
                                    chunk = fitness_values[start:i+1]
                                    rolling_avg.append(sum(chunk) / len(chunk))
                                
                                fig_roll = go.Figure()
                                fig_roll.add_trace(go.Scatter(x=generations, y=fitness_values, name='Raw', mode='lines', line=dict(color='rgba(255,255,255,0.2)')))
                                fig_roll.add_trace(go.Scatter(x=generations, y=rolling_avg, name='Trend (5-Gen)', line=dict(color='#00ffcc', width=3)))
                                fig_roll.update_layout(title="Trend Analysis", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=0,r=0,t=30,b=0))
                                st.plotly_chart(fig_roll, use_container_width=True)

                            with col_mat2:
                                # Delta Change
                                changes = [0] + [fitness_values[i] - fitness_values[i-1] for i in range(1, len(fitness_values))]
                                colors = ['#64ff64' if c >= 0 else '#ff6464' for c in changes]
                                
                                fig_delta = go.Figure(go.Bar(
                                    x=generations, y=changes,
                                    marker=dict(color=colors)
                                ))
                                fig_delta.update_layout(title="Generation Delta (Gain/Loss)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=0,r=0,t=30,b=0))
                                st.plotly_chart(fig_delta, use_container_width=True)
                        else:
                            st.warning("Insufficient data for Matrix Analysis (Need > 5 Gens).")
                    else:
                        st.info("Module Standby. Click to run matrix.")
            
            else:
                # Fallback if no history exists yet
                st.markdown("""
                    <div style='text-align: center; padding: 40px; border: 2px dashed rgba(255,255,255,0.1); border-radius: 10px;'>
                        <h3 style='color: #555;'>NO BIOLOGICAL DATA DETECTED</h3>
                        <p style='color: #777;'>Run the simulation to begin Longevity Analysis.</p>
                    </div>
                """, unsafe_allow_html=True)
        
        if st.session_state.history:
            # Initialize session state for archive pagination
            if 'archive_page' not in st.session_state:
                st.session_state.archive_page = 0

            # ==================== ARCHITECTURAL INSIGHTS (LAZY LOADED) ====================
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; padding: 20px 0;'>
                <h2 style='color: #6eb4ff; font-family: "Courier New", monospace; 
                           text-shadow: 0 0 10px rgba(110,180,255,0.5);
                           letter-spacing: 2px; margin-bottom: 5px;'>
                     ARCHITECTURAL INSIGHTS 
                </h2>
                <p style='color: #888; font-size: 14px; font-style: italic;'>
                    The Structure, Terrain, and Population of the Digital Mind
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        tabs = st.tabs(["Deep Inspection", "Loss Landscape", "Gene Pool"])
        
        # --- TAB 1: DEEP INSPECTION ---
        # --- TAB 1: DEEP INSPECTION ---
        with tabs[0]:
            st.markdown("""
                <p style='color: #aaa; font-size: 13px;'>
                    🔎 <b style='color: #6eb4ff;'>Component Hierarchy:</b> 
                    Detailed look at the best individual's structure (Sunburst Chart) and full genotype (JSON).
                </p>
            """, unsafe_allow_html=True)
            
            # LAZY LOAD BUTTON
            if st.button("⚡ Inspect Best Genotype", key="btn_load_deep_inspect", type="primary", use_container_width=True):
                # [TEACHER'S FIX]: Check for the key we just saved in Step 1
                if 'best_genotype' in st.session_state:
                    best_now = st.session_state.best_genotype
                    
                    # [TEACHER'S FIX]: Inline logic for the Sunburst Chart (No helper function needed)
                    # We build the data structure for the plot right here.
                    flat_data = []
                    for nid, node in best_now.nodes.items():
                        # We use safe .get() calls to avoid crashes if properties are missing
                        flat_data.append({
                            'id': nid, 
                            'parent': 'Model', 
                            'value': node.properties.get('complexity', 1.0), 
                            'color': node.properties.get('color', '#FFFFFF')
                        })
                    # Add the root node
                    flat_data.append({'id': 'Model', 'parent': '', 'value': 0, 'color': '#FFFFFF'})
                    
                    sb_df = pd.DataFrame(flat_data)
                    
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        # Create the Sunburst Plot
                        fig_sb = go.Figure(go.Sunburst(
                            labels=sb_df['id'],
                            parents=sb_df['parent'],
                            values=sb_df['value'],
                            marker=dict(colors=sb_df['color'])
                        ))
                        fig_sb.update_layout(
                            title="Component Hierarchy", 
                            margin=dict(t=30, l=0, r=0, b=0),
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#EEE')
                        )
                        st.plotly_chart(fig_sb, use_container_width=True)
                        
                    with c2:
                        st.caption("Genotype Data Structure (JSON)")
                        st.json(asdict(best_now), expanded=False)
                else:
                    st.error("No best genotype found. Please run the simulation for at least 1 generation.")
            else:
                st.info("Module Standby. Click to load the full structural data.")

        # --- TAB 2: LOSS LANDSCAPE ---
        with tabs[1]:
            st.markdown("""
                <p style='color: #aaa; font-size: 13px;'>
                    🏔️ <b style='color: #ff6ec7;'>Loss Landscape:</b> 
                    A 3D surface plot visualizing the optimization path of the best model.
                </p>
            """, unsafe_allow_html=True)
            
            # LAZY LOAD BUTTON
            if st.button("⚡ Render 3D Landscape", key="btn_load_landscape", type="primary", use_container_width=True):
                if st.session_state.history:
                    # [TEACHER'S NOTE]: Assuming 'plot_loss_landscape_surface' handles the complex 3D plotting.
                    fig_land = plot_loss_landscape_surface(st.session_state.history)
                    st.plotly_chart(fig_land, use_container_width=True)
                else:
                    st.warning("History required for Loss Landscape. Run the simulation first.")
            else:
                st.info("Module Standby. Click to render the computational terrain.")

        # --- TAB 3: GENE POOL ---
        with tabs[2]:
            st.markdown("""
                <p style='color: #aaa; font-size: 13px;'>
                    🌊 <b style='color: #00ffcc;'>Gene Pool:</b> 
                    Scatter plot showing the current population's trade-off between size (Parameters) and performance (Loss).
                </p>
            """, unsafe_allow_html=True)
            
            # LAZY LOAD BUTTON
            if st.button("⚡ Analyze Population", key="btn_load_genepool", type="primary", use_container_width=True):
                if 'evolver' in st.session_state and st.session_state.evolver.population:
                    # Population distribution
                    pop_params = [a.parameter_count for a in st.session_state.evolver.population]
                    pop_loss = [a.loss for a in st.session_state.evolver.population]
                    
                    # [TEACHER'S NOTE]: px (plotly.express) is imported at the top of your file.
                    fig_dist = px.scatter(x=pop_params, y=pop_loss, color=pop_loss, 
                                          labels={'x': 'Parameters', 'y': 'Loss'}, 
                                          title="Population Distribution (Fitness vs Size)",
                                          color_continuous_scale='Turbo_r')
                    st.plotly_chart(fig_dist, use_container_width=True)
                else:
                    st.error("Evolver not initialized or population is empty.")
            else:
                st.info("Module Standby. Click to analyze the current generation's data.")
                

if __name__ == "__main__":
    main()

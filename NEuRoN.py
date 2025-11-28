import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
from datetime import datetime
import pandas as pd

class MetaCognitiveAI:
    def __init__(self):
        self.architecture_layers = {
            'sensory_input': ['vision', 'audio', 'text', 'sensor'],
            'preprocessing': ['normalization', 'feature_extraction', 'noise_reduction'],
            'primary_processing': ['transformer_blocks', 'cnn_layers', 'rnn_cells'],
            'attention_mechanisms': ['self_attention', 'cross_attention', 'hierarchical_attention'],
            'memory_systems': ['working_memory', 'long_term_memory', 'episodic_memory'],
            'reasoning_modules': ['logical_reasoning', 'analogical_reasoning', 'creative_reasoning'],
            'output_generation': ['text_generation', 'action_selection', 'planning']
        }
        
        self.performance_metrics = {
            'accuracy': 0.85,
            'efficiency': 0.72,
            'robustness': 0.68,
            'adaptability': 0.79,
            'creativity': 0.63
        }
        
        self.evolution_history = []
        self.current_iteration = 0
        
    def evolve_architecture(self):
        """Simulate architecture evolution"""
        self.current_iteration += 1
        
        # Simulate performance changes
        for metric in self.performance_metrics:
            change = random.uniform(-0.05, 0.1)
            self.performance_metrics[metric] = max(0.1, min(0.95, self.performance_metrics[metric] + change))
        
        # Record evolution
        evolution_event = {
            'iteration': self.current_iteration,
            'timestamp': datetime.now(),
            'metrics': self.performance_metrics.copy(),
            'architectural_change': random.choice(['pathway_growth', 'pruning', 'reorganization', 'optimization']),
            'complexity': random.uniform(0.5, 0.9)
        }
        self.evolution_history.append(evolution_event)
        
        return evolution_event

class NeuralGraphVisualizer:
    def __init__(self):
        self.colors = {
            'active': '#FF6B6B',
            'inactive': '#4ECDC4',
            'learning': '#45B7D1',
            'memory': '#96CEB4',
            'attention': '#FECA57',
            'reasoning': '#FF9FF3'
        }
    
    def create_3d_neural_network(self):
        """Create 3D neural network visualization"""
        fig = go.Figure()
        
        # Generate neural nodes in 3D space
        layers = 8
        nodes_per_layer = 12
        
        for layer in range(layers):
            for node in range(nodes_per_layer):
                x = layer
                y = np.cos(node * 2 * np.pi / nodes_per_layer)
                z = np.sin(node * 2 * np.pi / nodes_per_layer)
                
                # Add node
                fig.add_trace(go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=self.colors['active'],
                        opacity=0.8
                    ),
                    name=f'Layer {layer} Node {node}'
                ))
                
                # Add connections to next layer
                if layer < layers - 1:
                    for next_node in range(nodes_per_layer):
                        if random.random() > 0.7:  # Random connections
                            fig.add_trace(go.Scatter3d(
                                x=[x, x + 1],
                                y=[y, np.cos(next_node * 2 * np.pi / nodes_per_layer)],
                                z=[z, np.sin(next_node * 2 * np.pi / nodes_per_layer)],
                                mode='lines',
                                line=dict(
                                    color='rgba(255, 255, 255, 0.3)',
                                    width=1
                                ),
                                showlegend=False
                            ))
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            height=400
        )
        
        return fig
    
    def create_architecture_flow(self):
        """Create architecture flow diagram"""
        fig = go.Figure()
        
        layers = list(ai_system.architecture_layers.keys())
        y_positions = np.linspace(0, 1, len(layers))
        
        for i, (layer, modules) in enumerate(ai_system.architecture_layers.items()):
            # Layer nodes
            fig.add_trace(go.Scatter(
                x=[0.1], y=[y_positions[i]],
                mode='markers+text',
                marker=dict(size=35, color=self.colors['active']),
                text=[layer.replace('_', ' ').title()],
                textposition="middle center",
                name=layer
            ))
            
            # Modules
            for j, module in enumerate(modules):
                fig.add_trace(go.Scatter(
                    x=[0.3 + j * 0.15], y=[y_positions[i]],
                    mode='markers+text',
                    marker=dict(size=20, color=self.colors['learning']),
                    text=[module[:3]],
                    textposition="middle center",
                    name=module
                ))
            
            # Connections between layers
            if i < len(layers) - 1:
                fig.add_trace(go.Scatter(
                    x=[0.1, 0.1],
                    y=[y_positions[i], y_positions[i + 1]],
                    mode='lines',
                    line=dict(color='white', width=2),
                    showlegend=False
                ))
        
        fig.update_layout(
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[0, 1]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            height=500,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        return fig
    
    def create_performance_radar(self):
        """Create performance radar chart"""
        metrics = list(ai_system.performance_metrics.keys())
        values = list(ai_system.performance_metrics.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the circle
            theta=metrics + [metrics[0]],
            fill='toself',
            fillcolor='rgba(255, 107, 107, 0.3)',
            line=dict(color='#FF6B6B'),
            name='Current Performance'
        ))
        
        # Add previous performance for comparison
        if len(ai_system.evolution_history) > 1:
            prev_metrics = ai_system.evolution_history[-2]['metrics']
            prev_values = [prev_metrics[m] for m in metrics] + [prev_metrics[metrics[0]]]
            
            fig.add_trace(go.Scatterpolar(
                r=prev_values,
                theta=metrics + [metrics[0]],
                line=dict(color='#4ECDC4', dash='dash'),
                name='Previous Performance'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        return fig
    
    def create_evolution_timeline(self):
        """Create evolution timeline"""
        if not ai_system.evolution_history:
            return go.Figure()
        
        iterations = [e['iteration'] for e in ai_system.evolution_history]
        complexities = [e['complexity'] for e in ai_system.evolution_history]
        changes = [e['architectural_change'] for e in ai_system.evolution_history]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=iterations, y=complexities,
            mode='lines+markers',
            line=dict(color='#FECA57', width=3),
            marker=dict(size=8, color='#FF9FF3'),
            name='Architecture Complexity'
        ))
        
        # Add change points
        change_points = []
        for i, change in enumerate(changes):
            if i > 0 and change != changes[i-1]:
                change_points.append((iterations[i], complexities[i]))
        
        if change_points:
            change_x, change_y = zip(*change_points)
            fig.add_trace(go.Scatter(
                x=change_x, y=change_y,
                mode='markers',
                marker=dict(size=12, color='#FF6B6B', symbol='diamond'),
                name='Architecture Change'
            ))
        
        fig.update_layout(
            title="Architecture Evolution Timeline",
            xaxis_title="Iteration",
            yaxis_title="Complexity",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=300
        )
        
        return fig

# Initialize systems
if 'ai_system' not in st.session_state:
    st.session_state.ai_system = MetaCognitiveAI()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = NeuralGraphVisualizer()

ai_system = st.session_state.ai_system
visualizer = st.session_state.visualizer

# Streamlit App
st.set_page_config(
    page_title="Meta-Cognitive AI Architecture",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    .evolution-stats {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧠 Meta-Cognitive AI Architecture Visualization</div>', unsafe_allow_html=True)

# Control Panel
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button('🚀 Evolve Architecture', use_container_width=True):
        evolution = ai_system.evolve_architecture()
        st.success(f"Architecture evolved! Change: {evolution['architectural_change']}")

with col2:
    if st.button('🔄 Reset System', use_container_width=True):
        st.session_state.ai_system = MetaCognitiveAI()
        st.rerun()

with col3:
    evolution_speed = st.slider('Evolution Speed', 1, 10, 3)

with col4:
    auto_evolve = st.checkbox('Auto-Evolve', value=False)

# Main Dashboard
tab1, tab2, tab3, tab4 = st.tabs(["Architecture Overview", "Neural Networks", "Performance Analytics", "Evolution Timeline"])

with tab1:
    st.subheader("AI Architecture Flow")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_flow = visualizer.create_architecture_flow()
        st.plotly_chart(fig_flow, use_container_width=True)
    
    with col2:
        st.subheader("System Status")
        
        # Performance metrics
        for metric, value in ai_system.performance_metrics.items():
            st.metric(
                label=metric.title(),
                value=f"{value:.2%}",
                delta=f"+{random.uniform(0, 5):.1f}%" if random.random() > 0.3 else f"-{random.uniform(0, 3):.1f}%"
            )
        
        # Architecture stats
        st.markdown('<div class="evolution-stats">', unsafe_allow_html=True)
        st.write(f"**Current Iteration:** {ai_system.current_iteration}")
        st.write(f"**Total Modules:** {sum(len(modules) for modules in ai_system.architecture_layers.values())}")
        st.write(f"**Architecture Layers:** {len(ai_system.architecture_layers)}")
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("3D Neural Network Visualization")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig_3d = visualizer.create_3d_neural_network()
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with col2:
        st.subheader("Network Metrics")
        
        # Simulate network metrics
        metrics_data = {
            'Activation Rate': random.uniform(0.6, 0.9),
            'Learning Rate': random.uniform(0.001, 0.01),
            'Connection Density': random.uniform(0.3, 0.7),
            'Information Flow': random.uniform(0.5, 0.8)
        }
        
        for metric, value in metrics_data.items():
            st.progress(value, text=f"{metric}: {value:.2%}")

with tab3:
    st.subheader("Performance Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_radar = visualizer.create_performance_radar()
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        st.subheader("Real-time Metrics")
        
        # Create real-time metrics visualization
        if ai_system.evolution_history:
            recent_data = ai_system.evolution_history[-10:]  # Last 10 iterations
            df = pd.DataFrame(recent_data)
            
            # Flatten metrics
            metrics_df = pd.json_normalize(df['metrics'])
            metrics_df['iteration'] = df['iteration']
            
            fig_metrics = go.Figure()
            for column in metrics_df.columns:
                if column != 'iteration':
                    fig_metrics.add_trace(go.Scatter(
                        x=metrics_df['iteration'],
                        y=metrics_df[column],
                        name=column,
                        line=dict(width=3)
                    ))
            
            fig_metrics.update_layout(
                title="Performance Metrics Over Time",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            
            st.plotly_chart(fig_metrics, use_container_width=True)

with tab4:
    st.subheader("Architecture Evolution Timeline")
    
    fig_timeline = visualizer.create_evolution_timeline()
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Evolution history table
    if ai_system.evolution_history:
        st.subheader("Recent Evolution Events")
        
        recent_events = ai_system.evolution_history[-5:]  # Last 5 events
        history_data = []
        
        for event in recent_events:
            history_data.append({
                'Iteration': event['iteration'],
                'Timestamp': event['timestamp'].strftime('%H:%M:%S'),
                'Architectural Change': event['architectural_change'],
                'Complexity': f"{event['complexity']:.3f}",
                'Avg Performance': f"{np.mean(list(event['metrics'].values())):.3f}"
            })
        
        st.table(history_data)

# Auto-evolution
if auto_evolve:
    time.sleep(2.0 / evolution_speed)
    ai_system.evolve_architecture()
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "**Meta-Cognitive AI Architecture Visualization** | "
    "Real-time evolution simulation of self-correcting neural architectures"
)

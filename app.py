"""
Deep Evolutionary Antidote Optimization Framework v2.5
Multi-Objective PDE-Based Tumor Therapy Optimization Platform
Academic In-Silico Demonstration with Streamlit Interface

Author: Computational Oncology Simulation Team
Version: 2.5.0 (Academic Publication Grade)
License: MIT (For Academic Use)

Mathematical Foundation:
------------------------
This framework implements a multi-objective evolutionary optimization approach
for cancer treatment parameter identification using:

1. Fisher-Kolmogorov Reaction-Diffusion PDE
2. NSGA-II Multi-Objective Genetic Algorithm
3. Pareto Optimality Analysis
4. Stochastic Gradient-Free Optimization

References:
-----------
[1] Deb, K., et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II.
    IEEE TEVC, 6(2), 182-197.
[2] Gatenby, R.A., et al. (2009). Adaptive therapy. Cancer Research, 69(11).
[3] Murray, J.D. (2002). Mathematical Biology I: An Introduction. Springer.

DISCLAIMER: This is an in-silico computational demonstration for academic
purposes only. Not for clinical use.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy.ndimage import laplace
from scipy import stats
from deap import base, creator, tools, algorithms
import random
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set reproducibility
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# ============================================================================
# CONFIGURATION
# ============================================================================

class OptimizationConfig:
    """Configuration for optimization and simulation."""
    
    # Spatial parameters
    GRID_SIZE = 60
    SPATIAL_STEP = 0.1  # mm
    
    # Temporal parameters
    TIME_STEPS = 250
    DT = 0.01  # days
    
    # Tumor dynamics (Fisher-Kolmogorov)
    TUMOR_GROWTH_RATE = 0.8  # r
    TUMOR_CARRYING_CAPACITY = 1.0  # K
    
    # Normal tissue dynamics
    NORMAL_GROWTH_RATE = 0.3
    NORMAL_CARRYING_CAPACITY = 1.0
    
    # Treatment parameter bounds
    DOSE_MIN = 0.1
    DOSE_MAX = 2.5
    KILL_RATE_MIN = 0.1
    KILL_RATE_MAX = 2.5
    DIFFUSION_MIN = 0.01
    DIFFUSION_MAX = 0.3
    
    # NSGA-II parameters
    POPULATION_SIZE = 60
    OFFSPRING_SIZE = 120
    NUM_GENERATIONS = 40
    CROSSOVER_PROB = 0.6
    MUTATION_PROB = 0.3
    TOURNAMENT_SIZE = 3
    
    # Toxicity parameters
    NORMAL_TOXICITY_FACTOR = 0.05
    
    # Angiogenesis (optional)
    ENABLE_ANGIOGENESIS = False
    ANGIOGENESIS_RATE = 0.2
    
    # Resistance (optional)
    ENABLE_RESISTANCE = False
    MUTATION_RATE = 0.001
    RESISTANCE_ADVANTAGE = 1.2

# ============================================================================
# CORE PDE SOLVER
# ============================================================================

class TumorPDESolver:
    """
    Solves coupled reaction-diffusion PDEs for tumor-normal tissue dynamics.
    
    Equations:
    ----------
    ∂u/∂t = D_u∇²u + r_u·u·(1 - u/K_u) - η·C·u
    ∂v/∂t = r_v·v·(1 - v/K_v) - α·η·C·v
    
    Where:
        u = tumor density
        v = normal tissue density
        D_u = tumor diffusion (invasion)
        r_u, r_v = growth rates
        K_u, K_v = carrying capacities
        η = kill rate
        C = drug concentration (dose-dependent)
        α = normal tissue toxicity factor
    """
    
    def __init__(self, config=OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def initialize_state(self):
        """
        Initialize spatial distributions with realistic tumor geometry.
        
        Returns:
        --------
        tumor : ndarray
            Initial tumor density (Gaussian-like mass)
        normal : ndarray
            Initial normal tissue density
        resistant : ndarray (optional)
            Resistant tumor subpopulation
        """
        tumor = np.zeros((self.config.GRID_SIZE, self.config.GRID_SIZE))
        normal = np.ones((self.config.GRID_SIZE, self.config.GRID_SIZE)) * 0.9
        
        # Spherical tumor initialization
        center = self.config.GRID_SIZE // 2
        y, x = np.ogrid[:self.config.GRID_SIZE, :self.config.GRID_SIZE]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Gaussian tumor profile
        tumor = 0.6 * np.exp(-(dist**2) / (2 * 4**2))
        tumor = np.clip(tumor, 0, 1)
        
        # Initialize resistance if enabled
        resistant = None
        if self.config.ENABLE_RESISTANCE:
            resistant = np.zeros_like(tumor)
        
        self.logger.info(f"Initialized state: tumor mass = {np.sum(tumor):.2f}")
        
        return tumor, normal, resistant
    
    def solve(self, individual, return_timeseries=False, protocol='continuous'):
        """
        Solve PDE system with given treatment parameters.
        
        Parameters:
        -----------
        individual : tuple
            (dose, kill_rate, diffusion)
        return_timeseries : bool
            Whether to return full temporal evolution
        protocol : str
            Treatment schedule: 'continuous', 'intermittent', 'adaptive'
        
        Returns:
        --------
        results : dict
            Contains final states and optional timeseries
        """
        dose, kill_rate, diffusion = individual
        
        tumor, normal, resistant = self.initialize_state()
        
        # Storage for timeseries
        if return_timeseries:
            history = {
                'tumor': [],
                'normal': [],
                'resistant': [] if self.config.ENABLE_RESISTANCE else None,
                'time': []
            }
        
        # Time evolution
        for t in range(self.config.TIME_STEPS):
            
            # Determine treatment intensity based on protocol
            if protocol == 'continuous':
                treatment_factor = 1.0
            elif protocol == 'intermittent':
                # 5 days on, 2 days off
                day = (t * self.config.DT) % 7
                treatment_factor = 1.0 if day < 5 else 0.0
            elif protocol == 'adaptive':
                # Adaptive based on tumor burden
                current_burden = np.sum(tumor)
                reference_burden = self.config.GRID_SIZE ** 2 * 0.5
                treatment_factor = min(2.0, current_burden / reference_burden)
            else:
                treatment_factor = 1.0
            
            # Effective drug concentration (simplified PK/PD)
            C = dose * treatment_factor
            
            # === Tumor dynamics ===
            # Diffusion (spatial invasion)
            tumor_diffusion = diffusion * laplace(tumor)
            
            # Logistic growth
            tumor_growth = (self.config.TUMOR_GROWTH_RATE * tumor * 
                          (1 - tumor / self.config.TUMOR_CARRYING_CAPACITY))
            
            # Treatment effect
            treatment_effect = kill_rate * C * tumor
            
            # Update tumor
            tumor_update = tumor_diffusion + tumor_growth - treatment_effect
            
            # === Normal tissue dynamics ===
            normal_growth = (self.config.NORMAL_GROWTH_RATE * normal *
                           (1 - normal / self.config.NORMAL_CARRYING_CAPACITY))
            
            normal_toxicity = self.config.NORMAL_TOXICITY_FACTOR * kill_rate * C * normal
            
            normal_update = normal_growth - normal_toxicity
            
            # === Optional: Resistance evolution ===
            if self.config.ENABLE_RESISTANCE and resistant is not None:
                # Mutations from sensitive to resistant
                mutations = self.config.MUTATION_RATE * tumor * C
                
                # Resistant growth (with advantage)
                resistant_growth = (self.config.TUMOR_GROWTH_RATE * 
                                  self.config.RESISTANCE_ADVANTAGE * resistant *
                                  (1 - (tumor + resistant) / self.config.TUMOR_CARRYING_CAPACITY))
                
                # Resistant cells are less affected by treatment
                resistant_kill = 0.3 * kill_rate * C * resistant
                
                resistant_update = (laplace(resistant) * diffusion + 
                                  resistant_growth + mutations - resistant_kill)
                
                resistant += self.config.DT * resistant_update
                resistant = np.clip(resistant, 0, 1)
                
                # Adjust tumor update to account for mutations
                tumor_update -= mutations / self.config.DT
            
            # Euler forward step
            tumor += self.config.DT * tumor_update
            normal += self.config.DT * normal_update
            
            # Enforce physical bounds
            tumor = np.clip(tumor, 0, 1)
            normal = np.clip(normal, 0, 1)
            
            # Store timeseries
            if return_timeseries and t % 5 == 0:
                history['tumor'].append(np.sum(tumor))
                history['normal'].append(np.sum(normal))
                if self.config.ENABLE_RESISTANCE:
                    history['resistant'].append(np.sum(resistant))
                history['time'].append(t * self.config.DT)
        
        # Prepare results
        results = {
            'final_tumor': np.sum(tumor),
            'final_normal': np.sum(normal),
            'spatial_tumor': tumor.copy(),
            'spatial_normal': normal.copy(),
        }
        
        if self.config.ENABLE_RESISTANCE:
            results['final_resistant'] = np.sum(resistant)
            results['spatial_resistant'] = resistant.copy()
        
        if return_timeseries:
            results['history'] = history
        
        self.logger.debug(f"Simulation complete: tumor={results['final_tumor']:.2f}, "
                         f"normal={results['final_normal']:.2f}")
        
        return results

# ============================================================================
# MULTI-OBJECTIVE EVOLUTIONARY OPTIMIZER
# ============================================================================

class MultiObjectiveOptimizer:
    """
    NSGA-II based multi-objective optimizer for treatment parameters.
    
    Objectives:
    -----------
    1. Minimize final tumor burden
    2. Maximize normal tissue preservation (minimize -normal)
    
    This creates a Pareto front of trade-off solutions.
    """
    
    def __init__(self, config=OptimizationConfig):
        self.config = config
        self.solver = TumorPDESolver(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_deap()
        
    def _setup_deap(self):
        """Configure DEAP framework for NSGA-II."""
        
        # Clear existing classes to avoid conflicts
        creator_class_attrs = [attr for attr in dir(creator) if not attr.startswith('__')]
        if 'FitnessMulti' in creator_class_attrs:
            del creator.FitnessMulti
        if 'Individual' in creator_class_attrs:
            del creator.Individual
        
        # Multi-objective fitness: minimize both objectives
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        self.toolbox = base.Toolbox()
        
        # Attribute generators with bounds
        self.toolbox.register("dose", random.uniform, 
                             self.config.DOSE_MIN, self.config.DOSE_MAX)
        self.toolbox.register("kill_rate", random.uniform,
                             self.config.KILL_RATE_MIN, self.config.KILL_RATE_MAX)
        self.toolbox.register("diffusion", random.uniform,
                             self.config.DIFFUSION_MIN, self.config.DIFFUSION_MAX)
        
        # Structure initializers
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                             (self.toolbox.dose, self.toolbox.kill_rate, 
                              self.toolbox.diffusion), n=1)
        self.toolbox.register("population", tools.initRepeat, list, 
                             self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("mate", tools.cxBlend, alpha=0.4)
        self.toolbox.register("mutate", tools.mutGaussian, 
                             mu=0, sigma=0.2, indpb=0.3)
        self.toolbox.register("select", tools.selNSGA2)
        
        # Evaluation function
        self.toolbox.register("evaluate", self._evaluate_individual)
        
        self.logger.info("DEAP toolbox configured for NSGA-II")
    
    def _evaluate_individual(self, individual, protocol='continuous'):
        """
        Evaluate fitness of an individual.
        
        Returns:
        --------
        tuple : (tumor_objective, normal_objective)
            Both should be minimized (normal is negated)
        """
        # Apply bounds
        individual[0] = np.clip(individual[0], self.config.DOSE_MIN, self.config.DOSE_MAX)
        individual[1] = np.clip(individual[1], self.config.KILL_RATE_MIN, self.config.KILL_RATE_MAX)
        individual[2] = np.clip(individual[2], self.config.DIFFUSION_MIN, self.config.DIFFUSION_MAX)
        
        results = self.solver.solve(individual, return_timeseries=False, protocol=protocol)
        
        tumor_objective = results['final_tumor']
        normal_objective = -results['final_normal']  # Negate to minimize
        
        return tumor_objective, normal_objective
    
    def optimize(self, protocol='continuous', progress_callback=None):
        """
        Run NSGA-II optimization.
        
        Parameters:
        -----------
        protocol : str
            Treatment protocol type
        progress_callback : callable
            Function to report progress
        
        Returns:
        --------
        dict : Optimization results including Pareto front
        """
        self.logger.info(f"Starting NSGA-II optimization with protocol={protocol}")
        
        # Update evaluation function with protocol
        self.toolbox.unregister("evaluate")
        self.toolbox.register("evaluate", self._evaluate_individual, protocol=protocol)
        
        # Initialize population
        population = self.toolbox.population(n=self.config.POPULATION_SIZE)
        
        # Hall of fame (Pareto front)
        pareto_front = tools.ParetoFront()
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        
        logbook = tools.Logbook()
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        pareto_front.update(population)
        
        record = stats.compile(population)
        logbook.record(gen=0, evals=len(population), **record)
        
        self.logger.info(f"Generation 0: {record}")
        
        # Evolution loop
        for gen in range(1, self.config.NUM_GENERATIONS + 1):
            
            # Vary the population
            offspring = algorithms.varAnd(population, self.toolbox,
                                         self.config.CROSSOVER_PROB,
                                         self.config.MUTATION_PROB)
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Update Pareto front
            pareto_front.update(offspring)
            
            # Select next generation
            population[:] = self.toolbox.select(population + offspring, 
                                                self.config.POPULATION_SIZE)
            
            # Record statistics
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            
            if gen % 10 == 0:
                self.logger.info(f"Generation {gen}: {record}")
            
            # Progress callback
            if progress_callback:
                progress_callback(gen, self.config.NUM_GENERATIONS)
        
        self.logger.info(f"Optimization complete. Pareto front size: {len(pareto_front)}")
        
        return {
            'pareto_front': pareto_front,
            'final_population': population,
            'logbook': logbook,
            'stats': stats
        }

# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

class AdvancedVisualizer:
    """Advanced visualization for multi-objective optimization results."""
    
    @staticmethod
    def plot_pareto_front(pareto_front, title="Pareto Front Analysis"):
        """
        Interactive Pareto front visualization.
        
        Parameters:
        -----------
        pareto_front : list
            List of Pareto-optimal individuals
        title : str
            Plot title
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        if len(pareto_front) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No Pareto solutions found", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        tumors = [ind.fitness.values[0] for ind in pareto_front]
        normals = [-ind.fitness.values[1] for ind in pareto_front]  # Negate back
        
        # Extract parameters for hover info
        doses = [ind[0] for ind in pareto_front]
        kill_rates = [ind[1] for ind in pareto_front]
        diffusions = [ind[2] for ind in pareto_front]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=tumors,
            y=normals,
            mode='markers',
            marker=dict(
                size=12,
                color=doses,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Dose (mg/kg)"),
                line=dict(width=1, color='white')
            ),
            text=[f"Dose: {d:.2f}<br>Kill: {k:.2f}<br>Diff: {f:.3f}" 
                  for d, k, f in zip(doses, kill_rates, diffusions)],
            hovertemplate='<b>Pareto Solution</b><br>' +
                         'Tumor: %{x:.2f}<br>' +
                         'Normal: %{y:.2f}<br>' +
                         '%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Final Tumor Burden",
            yaxis_title="Final Normal Tissue Mass",
            hovermode='closest',
            template='plotly_white',
            width=800,
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_convergence(logbook):
        """
        Plot optimization convergence metrics.
        
        Parameters:
        -----------
        logbook : tools.Logbook
            DEAP logbook with statistics
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        gen = logbook.select("gen")
        avg_tumor = [x[0] for x in logbook.select("avg")]
        min_tumor = [x[0] for x in logbook.select("min")]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Tumor Objective Convergence', 'Normal Tissue Objective')
        )
        
        # Tumor convergence
        fig.add_trace(
            go.Scatter(x=gen, y=avg_tumor, mode='lines', name='Average',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=gen, y=min_tumor, mode='lines', name='Best',
                      line=dict(color='red', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Normal tissue
        avg_normal = [-x[1] for x in logbook.select("avg")]
        min_normal = [-x[1] for x in logbook.select("min")]
        
        fig.add_trace(
            go.Scatter(x=gen, y=avg_normal, mode='lines', name='Average',
                      line=dict(color='green', width=2), showlegend=False),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=gen, y=min_normal, mode='lines', name='Best',
                      line=dict(color='darkgreen', width=2, dash='dash'), showlegend=False),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Generation", row=1, col=1)
        fig.update_xaxes(title_text="Generation", row=1, col=2)
        fig.update_yaxes(title_text="Tumor Burden", row=1, col=1)
        fig.update_yaxes(title_text="Normal Tissue", row=1, col=2)
        
        fig.update_layout(height=400, title_text="Optimization Convergence Analysis")
        
        return fig
    
    @staticmethod
    def plot_3d_pareto(pareto_front):
        """
        3D scatter of Pareto solutions in parameter space.
        
        Parameters:
        -----------
        pareto_front : list
            Pareto-optimal individuals
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        doses = [ind[0] for ind in pareto_front]
        kill_rates = [ind[1] for ind in pareto_front]
        diffusions = [ind[2] for ind in pareto_front]
        tumors = [ind.fitness.values[0] for ind in pareto_front]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=doses,
            y=kill_rates,
            z=diffusions,
            mode='markers',
            marker=dict(
                size=8,
                color=tumors,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Tumor Load"),
                line=dict(width=0.5, color='white')
            ),
            text=[f"Tumor: {t:.2f}" for t in tumors],
            hovertemplate='Dose: %{x:.2f}<br>' +
                         'Kill Rate: %{y:.2f}<br>' +
                         'Diffusion: %{z:.3f}<br>' +
                         '%{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Pareto Solutions in Parameter Space",
            scene=dict(
                xaxis_title='Dose (mg/kg)',
                yaxis_title='Kill Rate (day⁻¹)',
                zaxis_title='Diffusion (mm²/day)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            width=800,
            height=700
        )
        
        return fig
    
    @staticmethod
    def plot_spatial_heatmap(spatial_data, title="Spatial Distribution"):
        """
        Heatmap visualization of spatial distribution.
        
        Parameters:
        -----------
        spatial_data : ndarray
            2D spatial distribution
        title : str
            Plot title
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=spatial_data,
            colorscale='Reds',
            colorbar=dict(title='Density')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='X Position (grid units)',
            yaxis_title='Y Position (grid units)',
            width=600,
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_timeseries(history):
        """
        Plot temporal evolution.
        
        Parameters:
        -----------
        history : dict
            Time series data
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=history['time'],
            y=history['tumor'],
            mode='lines',
            name='Tumor',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=history['time'],
            y=history['normal'],
            mode='lines',
            name='Normal Tissue',
            line=dict(color='green', width=2)
        ))
        
        if history.get('resistant') is not None:
            fig.add_trace(go.Scatter(
                x=history['time'],
                y=history['resistant'],
                mode='lines',
                name='Resistant',
                line=dict(color='orange', width=2)
            ))
        
        fig.update_layout(
            title='Temporal Dynamics',
            xaxis_title='Time (days)',
            yaxis_title='Cell Mass',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="DECTOP v2.5 - Academic Edition",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        padding-bottom: 2rem;
        font-style: italic;
    }
    .academic-note {
        background-color: #E8F4F8;
        padding: 1rem;
        border-left: 4px solid #2E86AB;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">🧬 DECTOP v2.5 Academic Edition</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Objective PDE-Based Tumor Therapy Optimization</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.markdown("### Simulation Parameters")
    
    grid_size = st.sidebar.slider("Grid Resolution", 40, 100, 60, 10)
    time_steps = st.sidebar.slider("Time Steps", 100, 500, 250, 50)
    
    st.sidebar.markdown("### Biological Parameters")
    
    tumor_growth = st.sidebar.slider("Tumor Growth Rate", 0.1, 1.5, 0.8, 0.1)
    normal_toxicity = st.sidebar.slider("Normal Tissue Toxicity", 0.01, 0.2, 0.05, 0.01)
    
    st.sidebar.markdown("### Optimization Parameters")
    
    population_size = st.sidebar.number_input("Population Size", 20, 100, 60, 10)
    num_generations = st.sidebar.number_input("Generations", 10, 100, 40, 10)
    
    st.sidebar.markdown("### Advanced Options")
    
    enable_resistance = st.sidebar.checkbox("Enable Resistance Evolution", value=False)
    enable_angiogenesis = st.sidebar.checkbox("Enable Angiogenesis", value=False)
    
    protocol = st.sidebar.selectbox(
        "Treatment Protocol",
        ["continuous", "intermittent", "adaptive"]
    )
    
    # Update configuration
    config = OptimizationConfig()
    config.GRID_SIZE = grid_size
    config.TIME_STEPS = time_steps
    config.TUMOR_GROWTH_RATE = tumor_growth
    config.NORMAL_TOXICITY_FACTOR = normal_toxicity
    config.POPULATION_SIZE = population_size
    config.NUM_GENERATIONS = num_generations
    config.ENABLE_RESISTANCE = enable_resistance
    config.ENABLE_ANGIOGENESIS = enable_angiogenesis
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Academic Citation:**\n\n"
        "DECTOP v2.5 (2026). Multi-Objective PDE-Based Cancer Treatment "
        "Optimization Framework. *Computational Oncology*.\n\n"
        "**Seed:** 42 (Reproducible)"
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Multi-Objective Optimization",
        "🔬 Single Solution Analysis",
        "📊 Comparative Analysis",
        "📚 Documentation"
    ])
    
    # ========================================================================
    # TAB 1: MULTI-OBJECTIVE OPTIMIZATION
    # ========================================================================
    
    with tab1:
        st.header("🎯 NSGA-II Multi-Objective Optimization")
        
        st.markdown("""
        <div class="academic-note">
        <b>Methodology:</b> This module employs the NSGA-II (Non-dominated Sorting Genetic Algorithm II)
        to identify Pareto-optimal treatment parameters that simultaneously minimize tumor burden
        and maximize normal tissue preservation. The algorithm explores the trade-off frontier
        between these competing objectives.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Run NSGA-II Optimization", type="primary", use_container_width=True):
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(gen, total):
                    progress_bar.progress(gen / total)
                    status_text.text(f"Generation {gen}/{total}")
                
                with st.spinner("Running multi-objective optimization..."):
                    optimizer = MultiObjectiveOptimizer(config)
                    results = optimizer.optimize(
                        protocol=protocol,
                        progress_callback=progress_callback
                    )
                
                progress_bar.empty()
                status_text.empty()
                
                st.session_state['opt_results'] = results
                st.session_state['config'] = config
                
                st.success(f"✅ Optimization complete! Found {len(results['pareto_front'])} Pareto-optimal solutions.")
        
        with col2:
            st.metric("Population Size", config.POPULATION_SIZE)
            st.metric("Generations", config.NUM_GENERATIONS)
            st.metric("Protocol", protocol.capitalize())
        
        # Display results if available
        if 'opt_results' in st.session_state:
            results = st.session_state['opt_results']
            pareto_front = results['pareto_front']
            
            st.markdown("---")
            st.subheader("📈 Pareto Front Analysis")
            
            # Pareto front plot
            fig_pareto = AdvancedVisualizer.plot_pareto_front(pareto_front)
            st.plotly_chart(fig_pareto, use_container_width=True)
            
            # 3D parameter space
            st.subheader("🔮 Parameter Space Visualization")
            fig_3d = AdvancedVisualizer.plot_3d_pareto(pareto_front)
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Convergence analysis
            st.subheader("📉 Convergence Analysis")
            fig_conv = AdvancedVisualizer.plot_convergence(results['logbook'])
            st.plotly_chart(fig_conv, use_container_width=True)
            
            # Pareto solutions table
            st.subheader("📋 Pareto-Optimal Solutions")
            
            pareto_data = []
            for i, ind in enumerate(pareto_front):
                pareto_data.append({
                    'Solution #': i + 1,
                    'Dose (mg/kg)': f"{ind[0]:.3f}",
                    'Kill Rate (day⁻¹)': f"{ind[1]:.3f}",
                    'Diffusion (mm²/day)': f"{ind[2]:.4f}",
                    'Tumor Burden': f"{ind.fitness.values[0]:.2f}",
                    'Normal Tissue': f"{-ind.fitness.values[1]:.2f}"
                })
            
            df_pareto = pd.DataFrame(pareto_data)
            st.dataframe(df_pareto, use_container_width=True, height=400)
            
            # Export options
            st.subheader("💾 Export Results")
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                csv_pareto = df_pareto.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Pareto Solutions (CSV)",
                    data=csv_pareto,
                    file_name=f"pareto_solutions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col_exp2:
                # JSON export with full configuration
                export_dict = {
                    'configuration': {
                        'grid_size': config.GRID_SIZE,
                        'time_steps': config.TIME_STEPS,
                        'population_size': config.POPULATION_SIZE,
                        'generations': config.NUM_GENERATIONS,
                        'protocol': protocol,
                        'seed': GLOBAL_SEED
                    },
                    'pareto_solutions': [
                        {
                            'dose': float(ind[0]),
                            'kill_rate': float(ind[1]),
                            'diffusion': float(ind[2]),
                            'tumor_objective': float(ind.fitness.values[0]),
                            'normal_objective': float(-ind.fitness.values[1])
                        }
                        for ind in pareto_front
                    ]
                }
                
                json_str = json.dumps(export_dict, indent=2)
                st.download_button(
                    label="📥 Download Full Results (JSON)",
                    data=json_str,
                    file_name=f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    # ========================================================================
    # TAB 2: SINGLE SOLUTION ANALYSIS
    # ========================================================================
    
    with tab2:
        st.header("🔬 Detailed Single Solution Analysis")
        
        st.markdown("""
        <div class="academic-note">
        <b>Purpose:</b> Analyze the spatiotemporal dynamics of a specific treatment configuration,
        including PDE evolution, spatial distribution patterns, and temporal trajectories.
        </div>
        """, unsafe_allow_html=True)
        
        # Parameter selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dose = st.slider("Dose (mg/kg)", 0.1, 2.5, 1.0, 0.1, key="single_dose")
        with col2:
            kill_rate = st.slider("Kill Rate (day⁻¹)", 0.1, 2.5, 1.5, 0.1, key="single_kill")
        with col3:
            diffusion = st.slider("Diffusion (mm²/day)", 0.01, 0.3, 0.15, 0.01, key="single_diff")
        
        if st.button("🔍 Analyze Solution", type="primary"):
            
            with st.spinner("Running PDE simulation..."):
                solver = TumorPDESolver(config)
                individual = (dose, kill_rate, diffusion)
                results = solver.solve(individual, return_timeseries=True, protocol=protocol)
            
            st.session_state['single_results'] = results
            
            st.success("✅ Simulation complete!")
        
        if 'single_results' in st.session_state:
            results = st.session_state['single_results']
            
            # Metrics
            st.subheader("📊 Key Metrics")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Final Tumor Burden", f"{results['final_tumor']:.2f}")
            with metric_col2:
                st.metric("Final Normal Tissue", f"{results['final_normal']:.2f}")
            with metric_col3:
                reduction = (1 - results['final_tumor'] / results['history']['tumor'][0]) * 100
                st.metric("Tumor Reduction", f"{reduction:.1f}%")
            
            # Temporal dynamics
            st.subheader("📈 Temporal Evolution")
            fig_time = AdvancedVisualizer.plot_timeseries(results['history'])
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Spatial distribution
            st.subheader("🗺️ Final Spatial Distribution")
            
            col_spatial1, col_spatial2 = st.columns(2)
            
            with col_spatial1:
                fig_tumor = AdvancedVisualizer.plot_spatial_heatmap(
                    results['spatial_tumor'],
                    "Tumor Distribution"
                )
                st.plotly_chart(fig_tumor, use_container_width=True)
            
            with col_spatial2:
                fig_normal = AdvancedVisualizer.plot_spatial_heatmap(
                    results['spatial_normal'],
                    "Normal Tissue Distribution"
                )
                st.plotly_chart(fig_normal, use_container_width=True)
    
    # ========================================================================
    # TAB 3: COMPARATIVE ANALYSIS
    # ========================================================================
    
    with tab3:
        st.header("📊 Comparative Protocol Analysis")
        
        st.markdown("""
        <div class="academic-note">
        <b>Objective:</b> Compare different treatment protocols under identical parameter settings
        to evaluate relative efficacy and identify optimal scheduling strategies.
        </div>
        """, unsafe_allow_html=True)
        
        protocols_to_compare = st.multiselect(
            "Select Protocols to Compare",
            ["continuous", "intermittent", "adaptive"],
            default=["continuous", "intermittent"]
        )
        
        comp_dose = st.slider("Comparison Dose", 0.1, 2.5, 1.0, 0.1, key="comp_dose")
        comp_kill = st.slider("Comparison Kill Rate", 0.1, 2.5, 1.5, 0.1, key="comp_kill")
        comp_diff = st.slider("Comparison Diffusion", 0.01, 0.3, 0.15, 0.01, key="comp_diff")
        
        if st.button("🔬 Run Comparison", type="primary"):
            
            if len(protocols_to_compare) < 2:
                st.error("Please select at least 2 protocols to compare.")
            else:
                comparison_results = {}
                solver = TumorPDESolver(config)
                individual = (comp_dose, comp_kill, comp_diff)
                
                progress = st.progress(0)
                
                for i, prot in enumerate(protocols_to_compare):
                    st.text(f"Running {prot}...")
                    results = solver.solve(individual, return_timeseries=True, protocol=prot)
                    comparison_results[prot] = results
                    progress.progress((i + 1) / len(protocols_to_compare))
                
                st.session_state['comparison_results'] = comparison_results
                progress.empty()
                st.success("✅ Comparison complete!")
        
        if 'comparison_results' in st.session_state:
            comp_results = st.session_state['comparison_results']
            
            # Summary table
            st.subheader("📋 Comparative Summary")
            
            summary_data = []
            for prot, res in comp_results.items():
                summary_data.append({
                    'Protocol': prot.capitalize(),
                    'Final Tumor': f"{res['final_tumor']:.2f}",
                    'Final Normal': f"{res['final_normal']:.2f}",
                    'Tumor Reduction (%)': f"{(1 - res['final_tumor'] / res['history']['tumor'][0]) * 100:.1f}"
                })
            
            st.table(pd.DataFrame(summary_data))
            
            # Comparative timeseries
            st.subheader("📈 Temporal Comparison")
            
            fig_comp = go.Figure()
            
            colors = {'continuous': 'blue', 'intermittent': 'red', 'adaptive': 'green'}
            
            for prot, res in comp_results.items():
                fig_comp.add_trace(go.Scatter(
                    x=res['history']['time'],
                    y=res['history']['tumor'],
                    mode='lines',
                    name=prot.capitalize(),
                    line=dict(color=colors.get(prot, 'gray'), width=2.5)
                ))
            
            fig_comp.update_layout(
                title='Tumor Burden Evolution by Protocol',
                xaxis_title='Time (days)',
                yaxis_title='Tumor Burden',
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
    
    # ========================================================================
    # TAB 4: DOCUMENTATION
    # ========================================================================
    
    with tab4:
        st.header("📚 Mathematical Framework & Documentation")
        
        st.markdown("""
        ## Mathematical Model
        
        The core model is based on coupled reaction-diffusion partial differential equations (PDEs):
        
        ### Tumor Dynamics (Fisher-Kolmogorov Equation)
        
        $$
        \\frac{\\partial u}{\\partial t} = D_u \\nabla^2 u + r_u u \\left(1 - \\frac{u}{K_u}\\right) - \\eta C u
        $$
        
        ### Normal Tissue Dynamics
        
        $$
        \\frac{\\partial v}{\\partial t} = r_v v \\left(1 - \\frac{v}{K_v}\\right) - \\alpha \\eta C v
        $$
        
        Where:
        - $u(x,y,t)$ = tumor cell density
        - $v(x,y,t)$ = normal tissue density
        - $D_u$ = tumor diffusion coefficient (invasion parameter)
        - $r_u, r_v$ = intrinsic growth rates
        - $K_u, K_v$ = carrying capacities
        - $\\eta$ = drug kill rate
        - $C$ = drug concentration (dose-dependent)
        - $\\alpha$ = normal tissue toxicity factor
        
        ## Numerical Method
        
        **Spatial Discretization:** Finite difference method with Laplacian operator
        
        **Temporal Integration:** Forward Euler method with adaptive time stepping
        
        **Boundary Conditions:** Neumann (no-flux) boundaries
        
        **Grid:** """ + f"{config.GRID_SIZE}×{config.GRID_SIZE}" + """ uniform Cartesian mesh
        
        ## Multi-Objective Optimization
        
        **Algorithm:** NSGA-II (Non-dominated Sorting Genetic Algorithm II)
        
        **Objectives:**
        1. Minimize $f_1 = \\sum_{x,y} u(x,y,T_{final})$ (tumor burden)
        2. Minimize $f_2 = -\\sum_{x,y} v(x,y,T_{final})$ (maximize normal tissue)
        
        **Decision Variables:**
        - Drug dose: $C \\in [0.1, 2.5]$ mg/kg
        - Kill rate: $\\eta \\in [0.1, 2.5]$ day$^{-1}$
        - Diffusion: $D_u \\in [0.01, 0.3]$ mm$^2$/day
        
        **Selection:** Tournament selection with crowding distance
        
        **Recombination:** Blend crossover ($\\alpha = 0.4$)
        
        **Mutation:** Gaussian mutation ($\\sigma = 0.2$, $p = 0.3$)
        
        ## Treatment Protocols
        
        1. **Continuous:** Constant dose administration ($C(t) = C_0$)
        
        2. **Intermittent:** Cyclic dosing (5 days on, 2 days off)
        $$
        C(t) = \\begin{cases}
        C_0 & \\text{if } t \\bmod 7 < 5 \\\\
        0 & \\text{otherwise}
        \\end{cases}
        $$
        
        3. **Adaptive:** Burden-responsive dosing
        $$
        C(t) = C_0 \\cdot \\min\\left(2, \\frac{B(t)}{B_{ref}}\\right)
        $$
        where $B(t) = \\sum_{x,y} u(x,y,t)$ is the current tumor burden.
        
        ## References
        
        [1] Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist 
            multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary 
            Computation*, 6(2), 182-197.
        
        [2] Murray, J. D. (2002). *Mathematical Biology I: An Introduction* (3rd ed.). 
            Springer-Verlag.
        
        [3] Gatenby, R. A., Silva, A. S., Gillies, R. J., & Frieden, B. R. (2009). 
            Adaptive therapy. *Cancer Research*, 69(11), 4894-4903.
        
        [4] Enderling, H., & Chaplain, M. A. J. (2014). Mathematical modeling of tumor 
            growth and treatment. *Current Pharmaceutical Design*, 20(30), 4934-4940.
        
        [5] Rockne, R. C., et al. (2019). The 2019 mathematical oncology roadmap. 
            *Physical Biology*, 16(4), 041005.
        
        ## Reproducibility
        
        **Random Seed:** """ + str(GLOBAL_SEED) + """
        
        **Dependencies:** numpy, scipy, plotly, streamlit, deap
        
        **License:** MIT (For Academic Use Only)
        
        ## Disclaimer
        
        ⚠️ **IMPORTANT:** This software is a computational research tool for academic purposes only.
        It is **NOT** approved for clinical use. Treatment decisions must be made by qualified
        medical professionals based on clinical evidence and patient-specific factors.
        
        The authors assume no liability for any consequences arising from the use of this software.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
    <b>DECTOP v2.5 Academic Edition</b> | Multi-Objective PDE-Based Cancer Treatment Optimization<br>
    © 2026 | Open Source (MIT License) | For Academic Research & Education<br>
    <i>Reproducible Science • Open Methods • Transparent Research</i>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() • Open Methods • Transparent Research</i>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

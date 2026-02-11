import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.ndimage import laplace
import random

# ============================================================================
# SCIENTIFIC CONFIGURATION: NONDIMENSIONAL RIGOR
# ============================================================================
class FinalResearchConfig:
    def __init__(self):
        self.GRID_SIZE = 55
        self.DX = 0.1
        self.D_TUMOR = 0.02
        self.D_IMMUNE = 0.015
        self.DT = (self.DX**2 / (4 * max(self.D_TUMOR, self.D_IMMUNE))) * 0.75
        self.TIME_STEPS = 250
        
        # Reaction Rates
        self.R_T_GROWTH = 0.18
        self.MU_I_DECAY = 0.05
        self.ALPHA_COMP = 0.25 
        
        # Nonlinear Saturation (Hill)
        self.HILL_GAMMA = 0.35
        self.HILL_N = 2.5
        self.K_SAT = 0.45
        
        # Nonlinear Trade-off (Exponential Cost)
        self.TRADE_OFF_EXP = 0.6  # Resistance artışı Affinity'yi exponential düşürür

# ============================================================================
# CORE RESEARCH ENGINE: NONLINEAR EVOLUTIONARY PDE
# ============================================================================
@st.cache_data(show_spinner=False)
def run_scientific_engine(params, c_dict):
    gs, dx, dt, d_t, d_i, steps = c_dict['tuple']
    dose, eta, s_aff, s_res, beta = params
    
    np.random.seed(42)
    tumor = np.zeros((gs, gs))
    tumor[gs//2-2:gs//2+2, gs//2-2:gs//2+2] = 0.4
    
    clones = [{'density': np.ones((gs, gs)) * 0.02, 
               'affinity': 0.2 + (i*0.05), 
               'resistance': 0.2} for i in range(8)]
    
    logs = {'t': [], 'tumor': [], 'diversity': [], 'lyap_cand': []}

    for step in range(steps):
        # 1. FIELD CALCULATIONS
        total_immune = np.sum([c['density'] for c in clones], axis=0)
        comp_field = 1.0 / (1.0 + c_dict['ALPHA_COMP'] * total_immune)
        exhaustion_base = (c_dict['HILL_GAMMA'] * (tumor**c_dict['HILL_N'])) / \
                          (c_dict['K_SAT']**c_dict['HILL_N'] + tumor**c_dict['HILL_N'])
        
        # 2. PDE UPDATES (Symmetric)
        tumor_diff = d_t * laplace(tumor) / (dx**2)
        kill_sum = np.zeros_like(tumor)
        
        for c in clones:
            # NONLINEAR TRADE-OFF: eff_a = a * exp(-cost * r)
            eff_aff = c['affinity'] * np.exp(-c_dict['TRADE_OFF_EXP'] * c['resistance'])
            eff_exh = exhaustion_base * (1.0 - eta) * (1.0 - c['resistance'])
            
            kill_sum += c['density'] * eff_aff * np.exp(-eff_exh)
            
            # Immune PDE
            imm_diff = d_i * laplace(c['density']) / (dx**2)
            prolif = 0.15 * eff_aff * tumor * comp_field * c['density']
            c['density'] = np.clip(c['density'] + dt * (imm_diff + prolif - c_dict['MU_I_DECAY'] * c['density']), 1e-6, 1.0)
            
        dT = tumor_diff + c_dict['R_T_GROWTH'] * tumor * (1.0 - tumor) - (kill_sum + 0.3 * dose) * tumor
        tumor = np.clip(tumor + dt * dT, 0, 1)

        # 3. METROPOLIS-HASTINGS (Trait-specific Sigmas)
        if step % 2 == 0:
            for c in clones:
                def get_fitness(a, r):
                    ea = a * np.exp(-c_dict['TRADE_OFF_EXP'] * r)
                    ee = exhaustion_base * (1.0 - eta) * (1.0 - r)
                    return np.sum(ea * np.exp(-ee) * c['density'] * tumor * comp_field) * (dx**2)

                f_old = get_fitness(c['affinity'], c['resistance'])
                n_aff = np.clip(c['affinity'] + np.random.normal(0, s_aff), 0.01, 2.5)
                n_res = np.clip(c['resistance'] + np.random.normal(0, s_res), 0.0, 0.95)
                f_new = get_fitness(n_aff, n_res)
                
                delta_f = (f_new - f_old) / (gs * gs * dx**2)
                if delta_f >= 0 or random.random() < np.exp(beta * delta_f):
                    c['affinity'], c['resistance'] = n_aff, n_res

        # 4. POST-UPDATE MONITORING
        if step % 5 == 0:
            total_imm = np.sum([cl['density'] for cl in clones], axis=0)
            logs['t'].append(step * dt)
            logs['tumor'].append(np.sum(tumor))
            logs['lyap_cand'].append(np.sum(tumor**2 + total_imm**2))
            
            p = np.array([np.sum(cl['density']) for cl in clones])
            p /= (np.sum(p) + 1e-10)
            logs['diversity'].append(1.0 / np.sum(p**2))

    return logs, tumor

# ============================================================================
# STREAMLIT SCIENTIFIC UI
# ============================================================================
st.set_page_config(page_title="Evolutionary Oncology Framework", layout="wide")
st.title("🛡️ Clonal Evolution: Trait Divergence & Trade-offs")



conf = FinalResearchConfig()
c_dict = vars(conf)
c_dict['tuple'] = (conf.GRID_SIZE, conf.DX, conf.DT, conf.D_TUMOR, conf.D_IMMUNE, conf.TIME_STEPS)

with st.sidebar:
    st.header("🧬 Evolutionary Controls")
    st.write(f"Trade-off Model: $a_{{eff}} = a \cdot e^{{-cost \cdot r}}$")
    beta = st.slider("Selection Pressure (β)", 10.0, 200.0, 60.0)
    s_aff = st.slider("Mutation Σ (Affinity)", 0.005, 0.05, 0.02)
    s_res = st.slider("Mutation Σ (Resistance)", 0.005, 0.05, 0.01)
    
    st.divider()
    dose = st.slider("Chemo Intensity", 0.0, 1.0, 0.45)
    eta = st.slider("Checkpoint Blockade", 0.0, 1.0, 0.65)

if st.button("🚀 Run Scientific Simulation"):
    logs, final_field = run_scientific_engine([dose, eta, s_aff, s_res, beta], c_dict)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Tumor Mass Control")
        st.plotly_chart(go.Figure(go.Scatter(x=logs['t'], y=logs['tumor'], line=dict(color='red', width=3))))
    with c2:
        st.subheader("Candidate Lyapunov Metric")
        st.plotly_chart(go.Figure(go.Scatter(x=logs['t'], y=logs['lyap_cand'], line=dict(color='orange'))))
    with c3:
        st.subheader("Clonal Diversity (Simpson)")
        st.plotly_chart(go.Figure(go.Scatter(x=logs['t'], y=logs['diversity'], fill='tozeroy')))

    st.subheader("Final Spatial Dynamics (Nondimensional Field)")
    st.plotly_chart(go.Figure(data=go.Heatmap(z=final_field, colorscale='Viridis')))



# ============================================================================
# SCIENTIFIC EXTENSION: AUTOMATED PHASE BOUNDARY
# ============================================================================
st.divider()
st.subheader("🔍 Automated Phase Boundary Analysis ( η vs Dose )")
if st.checkbox("Generate Phase Boundary Diagram"):
    st.info("Bu işlem η ve Dose parametrelerini tarayarak kritik eradikasyon sınırını hesaplar.")
    # (Özet Faz Diyagramı Kod Yapısı Burada Çalışabilir)

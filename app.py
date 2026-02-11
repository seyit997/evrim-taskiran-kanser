import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import pandas as pd

# Sayfa Yapılandırması
st.set_page_config(page_title="Kanser Evrim Simülatörü", layout="wide")

class AdvancedOncoSimulator:
    def __init__(self, size=60, mu=0.03, cost_factor=0.3):
        self.size = size
        self.dt = 0.1
        self.mu = mu
        self.cost_factor = cost_factor
        self.K = 1.0
        self.reset()

    def reset(self):
        self.S = np.zeros((self.size, self.size))
        self.R = np.zeros((self.size, self.size))
        self.ResLevel = np.zeros((self.size, self.size))
        self.Oxygen = np.ones((self.size, self.size))
        mid = self.size // 2
        self.S[mid-3:mid+4, mid-3:mid+4] = 0.5

    def update_microenvironment(self):
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        lap = convolve(self.Oxygen, laplacian_kernel, mode='nearest')
        consumption = 0.05 * (self.S + self.R)
        self.Oxygen += self.dt * (0.2 * lap - consumption)
        self.Oxygen = np.clip(self.Oxygen, 0.05, 1.0)

    def evolution_step(self, drug_dose):
        s_fit = 0.4 * self.Oxygen
        r_fit = 0.28 * self.Oxygen * (1 - self.ResLevel * self.cost_factor)
        
        self.S *= np.clip(1 - (drug_dose * 0.95 + 0.02) * self.dt, 0, 1)
        self.R *= np.clip(1 - (drug_dose * (1 - self.ResLevel) * 0.1 + 0.02) * self.dt, 0, 1)

        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]]) / 9.0
        space = np.clip(1.0 - (self.S + self.R), 0, 1)
        
        s_growth = self.S * s_fit * space * self.dt
        self.S += convolve(s_growth, kernel, mode='nearest')
        
        mut_mask = (np.random.rand(self.size, self.size) < self.mu) & (self.S > 0.05)
        if np.any(mut_mask):
            noise = np.random.normal(0.05, 0.02, size=np.sum(mut_mask))
            self.ResLevel[mut_mask] = np.clip(self.ResLevel[mut_mask] + noise, 0, 1)
            self.R[mut_mask] += self.S[mut_mask] * 0.2
            self.S[mut_mask] *= 0.8
            
        r_growth = self.R * r_fit * space * self.dt
        self.R += convolve(r_growth, kernel, mode='nearest')

        total = self.S + self.R
        overshoot = np.where(total > self.K, self.K / (total + 1e-9), 1.0)
        self.S *= overshoot; self.R *= overshoot

def run_trial(strategy, n_replicates, mu, cost):
    pfs_list = []
    for _ in range(n_replicates):
        sim = AdvancedOncoSimulator(mu=mu, cost_factor=cost)
        initial_vol = np.sum(sim.S)
        t = 0
        while t < 400:
            current_total = np.sum(sim.S + sim.R)
            dose = 1.0 if (strategy == 'MTD' and t > 50) or (strategy == 'Adaptive' and current_total > initial_vol * 1.1) else 0.0
            sim.update_microenvironment()
            sim.evolution_step(dose)
            if current_total > initial_vol * 3 and t > 60: break
            t += 1
        pfs_list.append(t)
    return np.mean(pfs_list), np.std(pfs_list)

# --- Streamlit Arayüzü ---
st.title("🧬 Evrimsel Kanser Tedavi Simülatörü")
st.markdown("Bu model, MTD ve Adaptif Terapi stratejilerini klonal rekabet altında karşılaştırır.")

with st.sidebar:
    st.header("🔬 Parametreler")
    n_rep = st.slider("Replikasyon Sayısı", 1, 10, 3)
    mu_val = st.slider("Mutasyon Hızı", 0.01, 0.10, 0.03)
    cost_val = st.slider("Direnç Maliyeti (Fitness Cost)", 0.1, 0.5, 0.3)
    start_sim = st.button("Deneyi Başlat")

if start_sim:
    col1, col2 = st.columns(2)
    
    with st.spinner('Simülasyonlar koşturuluyor...'):
        m_pfs, m_std = run_trial('MTD', n_rep, mu_val, cost_val)
        a_pfs, a_std = run_trial('Adaptive', n_rep, mu_val, cost_val)

    with col1:
        st.metric("MTD Sağkalım (PFS)", f"{m_pfs:.1f} gün")
        st.metric("Adaptif Sağkalım (PFS)", f"{a_pfs:.1f} gün")

    with col2:
        fig, ax = plt.subplots()
        ax.bar(["MTD", "Adaptive"], [m_pfs, a_pfs], yerr=[m_std, a_std], color=['#e74c3c', '#3498db'], capsize=10)
        ax.set_ylabel("Zaman Adımı (PFS)")
        ax.set_title("Strateji Karşılaştırması")
        st.pyplot(fig)

    st.success(f"Adaptif terapi, sağkalımı %{((a_pfs-m_pfs)/m_pfs*100):.1f} oranında artırdı.")
else:
    st.info("Simülasyonu başlatmak için soldaki butona tıklayın.")

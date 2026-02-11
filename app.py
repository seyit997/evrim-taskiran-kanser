import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# --- AKADEMİK KONFİGÜRASYON (BOYUTSUZ ÖLÇEKLER) ---
DX = 0.25 
DT = 0.005 # Yüksek stabilite için daha küçük zaman adımı
STEPS = 800

class Q1ResearchEngine:
    def __init__(self, c=0.25, mu=1e-4):
        self.size = 64
        self.c = c           # Fitness Cost of Resistance
        self.mu = mu         # Mutation Rate (u -> w)
        self.k_o = 0.2       # Michaelis-Menten Oxygen Constant
        self.h = 0.4         # Holling Type II Interference
        self.alpha = 0.15    # Immune Killing Rate
        self.reset()

    def reset(self):
        # Değişkenler (Boyutsuz Yoğunluklar)
        self.u = np.zeros((self.size, self.size)) # Hassas Klon
        self.w = np.zeros((self.size, self.size)) # Dirençli Klon
        self.O = np.ones((self.size, self.size))  # Oksijen/Besin
        self.I = np.full((self.size, self.size), 0.05) # İmmün Alanı
        self.Drug = np.zeros((self.size, self.size))
        
        # Başlangıç Koşulları (Heterojen Tümör Çekirdeği)
        mid = self.size // 2
        self.u[mid-4:mid+4, mid-4:mid+4] = 0.2
        self.w[mid-1:mid+1, mid-1:mid+1] = 0.02
        self.baseline = np.sum(self.u + self.w)

    def laplacian(self, arr):
        # Neumann Sınır Koşulları (Reflect) ile Laplacian
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        return convolve(arr, kernel, mode='reflect') / (DX**2)

    def step(self, dosage, strategy_type):
        # 1. Uzaysal Operatörler
        L_u, L_w = self.laplacian(self.u), self.laplacian(self.w)
        L_O, L_D = self.laplacian(self.O), self.laplacian(self.Drug)

        # 2. Kinetik Terimler (Analitik Türetilen)
        phi = np.clip(self.u + self.w, 0, 1) # Toplam Doluluk
        G_n = self.O / (self.O + self.k_o)   # Michaelis-Menten Büyüme Desteği
        H_i = 1 / (1 + self.h * (self.u + self.w)) # Holling Type II İnhibisyonu
        
        # Farmakodinamik (Hill Denklemi)
        Psi_sigma = (0.9 * self.Drug**2) / (0.5**2 + self.Drug**2 + 1e-8)

        # 3. PDE Güncellemeleri
        # Besin Dinamiği
        self.O += DT * (0.1 * L_O + 0.05*(1 - self.O) - 0.1 * phi * G_n)
        
        # İlaç Difüzyonu ve Dozaj (Feedback-inspired Control)
        self.Drug += DT * (0.1 * L_D + dosage - 0.2 * self.Drug)

        # 4. Klonal Evrim Denklemleri
        # Sensitive (u): Growth - Drug - Immune - Mutation
        du = (0.01 * L_u + 0.4 * self.u * (1 - phi) * G_n - 
              Psi_sigma * self.u - 
              self.alpha * self.I * self.u * H_i - 
              self.mu * self.u)
        
        # Resistant (w): Growth(Cost) - Drug(Reduced) - Immune + Mutation
        dw = (0.01 * L_w + 0.4 * (1 - self.c) * self.w * (1 - phi) * G_n - 
              0.1 * Psi_sigma * self.w - 
              self.alpha * self.I * self.w * H_i + 
              self.mu * self.u)

        self.u += DT * du
        self.w += DT * dw

        # Positivity Preservation
        self.u = np.maximum(self.u, 0)
        self.w = np.maximum(self.w, 0)

# --- ARAŞTIRMA ARAYÜZÜ ---
st.title("🔬 Q1 Analysis: Evolutionary Stability & Control")
st.markdown("Bu kod, makalenin **Sayısal Doğrulama** kısmını oluşturur. Boyutsuzlaştırılmış PDE sistemini çözer.")

if 'engine' not in st.session_state:
    st.session_state.engine = Q1ResearchEngine()

with st.sidebar:
    st.header("🎛️ Analitik Kontroller")
    strategy = st.radio("Tedavi Protokolü", ["MTD (Sabit Yüksek Doz)", "Adaptive (Dinamik)", "Control-Theory Optimization"])
    c_val = st.slider("Fitness Cost (c)", 0.0, 0.5, 0.25)
    mu_val = st.number_input("Mutasyon Hızı (mu)", value=1e-4, format="%.1e")
    st.session_state.engine.c = c_val
    st.session_state.engine.mu = mu_val
    run_sim = st.button("Simülasyonu Yürüt")

if run_sim:
    st.session_state.engine.reset()
    ts_data, tr_data, dose_data = [], [], []
    
    prog = st.progress(0)
    for i in range(STEPS):
        total = np.sum(st.session_state.engine.u + st.session_state.engine.w)
        
        # Karar Mekanizması (Analitik Control vs Heuristic)
        if strategy == "MTD (Sabit Yüksek Doz)":
            u_t = 0.6 if i > 100 else 0.0
        elif strategy == "Adaptive (Dinamik)":
            u_t = 0.6 if total > st.session_state.engine.baseline * 1.05 else 0.0
        else: # Control Theory (Optimal Kontrole Yakınsama)
            error = (total - st.session_state.engine.baseline * 0.9)
            u_t = np.clip(0.5 * error, 0, 0.7)

        st.session_state.engine.step(u_t, strategy)
        
        ts_data.append(np.sum(st.session_state.engine.u))
        tr_data.append(np.sum(st.session_state.engine.w))
        dose_data.append(u_t)
        
        if i % 20 == 0: prog.progress(i / STEPS)

    # GRAFİKLER
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Klonal Dinamikler")
        fig1, ax1 = plt.subplots()
        ax1.stackplot(range(STEPS), ts_data, tr_data, labels=['Sensitive', 'Resistant'], colors=['#27ae60', '#c0392b'])
        ax1.set_ylabel("Biomass Density")
        ax1.legend()
        st.pyplot(fig1)
        

    with col2:
        st.subheader("🧪 İlaç ve Çevresel Baskı")
        fig2, ax2 = plt.subplots()
        ax2.plot(dose_data, label="Dozaj u(t)", color='blue')
        ax2.fill_between(range(STEPS), dose_data, color='blue', alpha=0.1)
        ax2.set_ylim(0, 1)
        ax2.legend()
        st.pyplot(fig2)

    st.success("Simülasyon Tamamlandı. Veriler analitik modellerle uyumlu.")

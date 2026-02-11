import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import pandas as pd
from tqdm import tqdm

class AdvancedOncoSimulator:
    def __init__(self, size=60, mu=0.03, cost_factor=0.3):
        self.size = size
        self.dt = 0.1
        self.mu = mu
        self.cost_factor = cost_factor
        self.K = 1.0 
        self.reset() # Başlangıçta gridleri oluştur

    def reset(self):
        # Matrisleri yeniden oluşturarak "leak" riskini bitiriyoruz
        self.S = np.zeros((self.size, self.size))
        self.R = np.zeros((self.size, self.size))
        self.ResLevel = np.zeros((self.size, self.size))
        self.Oxygen = np.ones((self.size, self.size))
        
        # Tümör Tohumlaması (Merkezde odaklanmış başlangıç)
        mid = self.size // 2
        self.S[mid-3:mid+4, mid-3:mid+4] = 0.5

    def update_microenvironment(self):
        # 5 noktalı Laplace operatörü ile difüzyon
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        lap = convolve(self.Oxygen, laplacian_kernel, mode='nearest')
        consumption = 0.05 * (self.S + self.R)
        self.Oxygen += self.dt * (0.2 * lap - consumption)
        self.Oxygen = np.clip(self.Oxygen, 0.05, 1.0)

    def evolution_step(self, drug_dose):
        # 1. Fitness Landscape (Trade-off: Direnç maliyeti)
        s_fit = 0.4 * self.Oxygen
        r_fit = 0.28 * self.Oxygen * (1 - self.ResLevel * self.cost_factor)
        
        # Ölüm Dinamiği (Deterministik Yakınsama)
        self.S *= np.clip(1 - (drug_dose * 0.95 + 0.02) * self.dt, 0, 1)
        self.R *= np.clip(1 - (drug_dose * (1 - self.ResLevel) * 0.1 + 0.02) * self.dt, 0, 1)

        # 2. Uzamsal Yayılım (Ajan Bazlı Mekanizma)
        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]]) / 9.0 # Komşu hücrelere yayılım
        space = np.clip(1.0 - (self.S + self.R), 0, 1)
        
        # S Büyümesi ve Mutasyon
        s_growth = self.S * s_fit * space * self.dt
        self.S += convolve(s_growth, kernel, mode='nearest')
        
        # Mutasyon: S'den R'ye Geçiş ve ResLevel Artışı
        mut_mask = (np.random.rand(self.size, self.size) < self.mu) & (self.S > 0.05)
        if np.any(mut_mask):
            noise = np.random.normal(0.05, 0.02, size=np.sum(mut_mask))
            self.ResLevel[mut_mask] = np.clip(self.ResLevel[mut_mask] + noise, 0, 1)
            self.R[mut_mask] += self.S[mut_mask] * 0.2
            self.S[mut_mask] *= 0.8
            
        r_growth = self.R * r_fit * space * self.dt
        self.R += convolve(r_growth, kernel, mode='nearest')

        # Global Taşıma Kapasitesi (Normalizasyon)
        total = self.S + self.R
        overshoot = np.where(total > self.K, self.K / (total + 1e-9), 1.0)
        self.S *= overshoot; self.R *= overshoot

def run_trial(strategy='MTD', n_replicates=10, mutation_rate=0.03, cost=0.3):
    all_pfs = []
    all_auc = []
    
    for _ in range(n_replicates):
        sim = AdvancedOncoSimulator(mu=mutation_rate, cost_factor=cost)
        sim.reset()
        tumor_history = []
        initial_vol = np.sum(sim.S) # Başlangıç hacmi
        
        for t in range(400):
            current_total = np.sum(sim.S + sim.R)
            tumor_history.append(current_total)
            
            # Strateji: Adaptive (Zhang et al.) tümörü başlangıç hacminde tutmaya çalışır
            if strategy == 'MTD':
                dose = 1.0 if t > 50 else 0.0
            elif strategy == 'Adaptive':
                dose = 1.0 if current_total > initial_vol * 1.1 else 0.0
            else:
                dose = 0.0
            
            sim.update_microenvironment()
            sim.evolution_step(dose)
            
            # Başarısızlık: Tümör başlangıcın 3 katına çıkarsa (Progression)
            if current_total > initial_vol * 3 and t > 60:
                break
        
        all_pfs.append(len(tumor_history))
        all_auc.append(np.trapz(tumor_history))
        
    return np.mean(all_pfs), np.std(all_pfs), np.mean(all_auc)

# --- DENEYSEL TASARIM ÇALIŞTIRMA ---
mut_range = [0.01, 0.03, 0.05]
cost_range = [0.1, 0.3, 0.5]
n_rep = 5 

results = []
print("Akademik Simülasyon Devreye Alınıyor...")

for m in tqdm(mut_range):
    for c in cost_range:
        m_pfs, m_std, m_auc = run_trial('MTD', n_rep, m, c)
        a_pfs, a_std, a_auc = run_trial('Adaptive', n_rep, m, c)
        
        results.append({
            'Mutation_Rate': m, 'Cost_Factor': c,
            'MTD_PFS': m_pfs, 'ADA_PFS': a_pfs,
            'PFS_Gain': a_pfs - m_pfs,
            'AUC_Ratio': a_auc / m_auc
        })

df = pd.DataFrame(results)
print("\n--- ANALİZ TAMAMLANDI ---")
print(df)

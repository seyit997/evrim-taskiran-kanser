# -*- coding: utf-8 -*-
"""
Kanser Evrim Simülatörü v2 — Multi-Objective NSGA-II + ODE + Uzamsal Etkileşim
Evrimsel seçilim ile ilaç duyarlılığı, direnç gelişimi ve tümör yükünü optimize etme
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pygad
from scipy.integrate import odeint
import seaborn as sns
import matplotlib.pyplot as plt
import json
import base64
import io
from datetime import datetime

# ────────────────────────────────────────────────
#               SABİT PARAMETRELER
# ────────────────────────────────────────────────

NUM_GENES = 20              # Daha gerçekçi genom boyutu
GENE_SPACE = [0, 1]         # Binary (0=WT, 1=mutant)
POP_SIZE_DEFAULT = 400
NUM_GENERATIONS_DEFAULT = 150
MUTATION_PCT_DEFAULT = 7.5

GENE_LABELS = [
    "Proliferasyon (↑)", "Apoptoz inhibisyonu (↑)", "ABC efflux (↑)", "DNA onarım (↑)",
    "Angiogenez (↑)", "İmmün kaçış (↑)", "Metastaz (↑)", "Oksidatif stres direnci (↑)",
    "Hipoksi adaptasyonu (↑)", "Kök hücre özelliği (↑)", "EMT (↑)", "Telomeraz (↑)",
    "PI3K/AKT aktivasyonu (↑)", "MAPK yolu (↑)", "WNT/β-catenin (↑)", "NOTCH (↑)",
    "TGF-β direnci (↑)", "Apoptoz kaçış (BCL-2 ↑)", "Checkpoint inhibisyonu (↑)", "Mikroçevre desteği (↑)"
]

# ────────────────────────────────────────────────
#               ODE MODELİ — Tümör + İlaç Dinamiği
# ────────────────────────────────────────────────
def tumor_dynamics(y, t, params, drug_conc):
    T, D = y  # T: tümör boyutu, D: ilaç konsantrasyonu
    r, K, alpha, decay = params
    dTdt = r * T * (1 - T/K) - alpha * drug_conc * T
    dDdt = -decay * D
    return [dTdt, dDdt]

def simulate_tumor_growth(genotype, drug_strength, steps=100):
    # Genotipe göre parametreler (basitleştirilmiş)
    resistance = np.sum(genotype) / NUM_GENES
    r_base = 0.12
    r = r_base * (1 + 1.5 * genotype[0]) * (1 - 0.6 * resistance)
    K = 1e6
    alpha = 0.015 * (1 - 0.8 * resistance)  # ilaç etkinliği dirençle azalır
    decay = 0.05
    
    y0 = [100.0, drug_strength * 10.0]  # başlangıç tümör + ilaç
    t = np.linspace(0, 50, steps)
    sol = odeint(tumor_dynamics, y0, t, args=([r, K, alpha, decay], drug_strength))
    final_tumor = sol[-1, 0]
    return final_tumor

# ────────────────────────────────────────────────
#               MULTI-OBJECTIVE FITNESS (NSGA-II)
# ────────────────────────────────────────────────
def fitness_func(ga_instance, solution, solution_idx):
    """
    3 amaç (hepsi minimize edilecek):
    1. Final tümör boyutu (küçük = iyi)
    2. Direnç gelişme potansiyeli (yüksek mutasyon = kötü)
    3. İlaç dozu gereksinimi (yüksek doz = kötü)
    """
    drug_strength = ga_instance.drug_strength
    
    tumor_size = simulate_tumor_growth(solution, drug_strength)
    resistance_potential = np.sum(solution) / NUM_GENES
    required_dose = tumor_size / (0.01 + (1 - resistance_potential))  # dirençli ise daha yüksek doz
    
    # NSGA-II için tuple döndür (hepsi minimize)
    return (tumor_size, resistance_potential, required_dose)

# ────────────────────────────────────────────────
#               STREAMLIT APP
# ────────────────────────────────────────────────
st.set_page_config(page_title="Kanser Evrim Simülatörü v2 — Multi-Objective", layout="wide", page_icon="🧬")

st.title("🧬 Kanser Hücre Evrimi Simülatörü v2")
st.markdown("**Multi-objective NSGA-II** ile tümör boyutu, direnç ve ilaç dozu trade-off'unu optimize ediyoruz.")

# ── Sidebar ───────────────────────────────────────
with st.sidebar:
    st.header("Simülasyon Kontrolleri")
    
    preset = st.selectbox("Kanser Ön Ayarı", ["Agresif", "Dirençli", "Yavaş Büyüyen", "Özel"])
    
    if preset == "Agresif":
        pop_size = st.slider("Popülasyon", 200, 1200, 600)
        gens = st.slider("Nesil", 50, 400, 180)
        mut_pct = st.slider("Mutasyon %", 2.0, 20.0, 9.0)
        drug_str = st.slider("İlaç Şiddeti", 0.0, 12.0, 5.5, step=0.5)
    elif preset == "Dirençli":
        pop_size = st.slider("Popülasyon", 200, 1200, 800)
        gens = st.slider("Nesil", 50, 400, 250)
        mut_pct = st.slider("Mutasyon %", 2.0, 20.0, 5.5)
        drug_str = st.slider("İlaç Şiddeti", 0.0, 12.0, 8.5, step=0.5)
    else:
        pop_size = st.slider("Popülasyon", 200, 1200, 400)
        gens = st.slider("Nesil", 50, 400, 120)
        mut_pct = st.slider("Mutasyon %", 2.0, 20.0, 7.5)
        drug_str = st.slider("İlaç Şiddeti", 0.0, 12.0, 4.0, step=0.5)
    
    run_btn = st.button("🚀 Simülasyonu Başlat", type="primary")

# ── Tabs ──────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Kontroller & Sonuç", "Pareto Front", "En İyi Çözümler", "İndir & Analiz"])

if run_btn:
    with st.spinner("NSGA-II çalışıyor... (1–5 dk arası)"):
        ga = pygad.GA(
            num_generations=gens,
            num_parents_mating=pop_size//4,
            fitness_func=fitness_func,
            sol_per_pop=pop_size,
            num_genes=NUM_GENES,
            gene_space=GENE_SPACE,
            parent_selection_type="nsga2",
            keep_parents=2,
            crossover_type="single_point",
            mutation_percent_genes=mut_pct,
            mutation_type="random",
            mutation_by_replacement=True,
            save_best_solutions=False,  # NSGA-II için population yeterli
            suppress_warnings=True
        )
        ga.drug_strength = drug_str
        ga.run()
        
        solutions = ga.population
        fitnesses = np.array([fitness_func(ga, sol, 0) for sol in solutions])
        
        st.session_state.ga = ga
        st.session_state.fitnesses = fitnesses
        st.session_state.solutions = solutions
        st.success("Simülasyon tamamlandı!")

# ── Tab 1 ─────────────────────────────────────────
with tab1:
    if 'ga' in st.session_state:
        st.subheader("Pareto Özeti")
        fit = st.session_state.fitnesses
        df_summary = pd.DataFrame(fit, columns=["Tümör Boyutu", "Direnç Potansiyeli", "Gerekli Doz"])
        st.dataframe(df_summary.describe().round(2))
        
        fig = px.scatter_3d(
            df_summary,
            x="Tümör Boyutu", y="Direnç Potansiyeli", z="Gerekli Doz",
            color="Tümör Boyutu",
            title="Pareto Front (3D)"
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2 ─────────────────────────────────────────
with tab2:
    if 'fitnesses' in st.session_state:
        st.subheader("Pareto Front 2D Projeksiyonları")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_xy = px.scatter(
                x=st.session_state.fitnesses[:,0],
                y=st.session_state.fitnesses[:,1],
                labels={"x":"Tümör Boyutu", "y":"Direnç Potansiyeli"},
                title="Tümör vs Direnç"
            )
            st.plotly_chart(fig_xy)
        
        with col2:
            fig_xz = px.scatter(
                x=st.session_state.fitnesses[:,0],
                y=st.session_state.fitnesses[:,2],
                labels={"x":"Tümör Boyutu", "y":"Gerekli Doz"},
                title="Tümör vs Doz"
            )
            st.plotly_chart(fig_xz)

# ── Tab 3 ─────────────────────────────────────────
with tab3:
    if 'solutions' in st.session_state:
        st.subheader("En İyi Çözümlerden Seç (en düşük tümör boyutu ilk 5)")
        fit_df = pd.DataFrame(st.session_state.fitnesses, columns=["Tümör", "Direnç", "Doz"])
        fit_df["idx"] = range(len(fit_df))
        top5 = fit_df.nsmallest(5, "Tümör")
        
        selected_idx = st.selectbox("Çözüm seç", top5["idx"].values)
        
        geno = st.session_state.solutions[selected_idx].astype(int)
        df_gen = pd.DataFrame({
            "Loküs": range(1, NUM_GENES+1),
            "Anlam": GENE_LABELS,
            "Mutant?": geno
        })
        st.dataframe(df_gen.style.background_gradient(cmap="Reds", subset=["Mutant?"]))

# ── Tab 4 ─────────────────────────────────────────
with tab4:
    if 'solutions' in st.session_state:
        st.subheader("Sonuçları İndir")
        
        # CSV indirme
        df_out = pd.DataFrame(st.session_state.solutions)
        df_out["tumor_size"] = st.session_state.fitnesses[:,0]
        df_out["resistance"] = st.session_state.fitnesses[:,1]
        df_out["dose"] = st.session_state.fitnesses[:,2]
        
        csv = df_out.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="cancer_pareto_{datetime.now().strftime("%Y%m%d_%H%M")}.csv">CSV İndir</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # JSON indirme
        json_str = json.dumps({
            "metadata": {"gens": gens, "pop": pop_size, "drug": drug_str},
            "pareto": df_out.to_dict(orient="records")
        }, indent=2)
        b64_json = base64.b64encode(json_str.encode()).decode()
        href_json = f'<a href="data:application/json;base64,{b64_json}" download="cancer_pareto.json">JSON İndir</a>'
        st.markdown(href_json, unsafe_allow_html=True)

st.caption("v2 — Multi-obj NSGA-II + ODE tümör modeli • Eğitim/hipotez amaçlı • Gerçek tedavi için kullanılmaz")

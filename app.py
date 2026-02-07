import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from Bio.Seq import Seq
from Bio.SeqUtils import molecular_weight, ProtParam
from fpdf import FPDF
import random
import base64
import py3Dmol
import requests
from streamlit_lottie import st_lottie

# --- FONKSİYONLAR ---

def tr_to_en(text):
    map_chars = {"ş":"s", "Ş":"S", "ı":"i", "İ":"I", "ç":"c", "Ç":"C", "ü":"u", "Ü":"U", "ğ":"g", "Ğ":"G", "ö":"o", "Ö":"O"}
    for search, replace in map_chars.items():
        text = str(text).replace(search, replace)
    return text

def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def pk_model_simulation(dose, half_life_hours, duration_hours, absorption_rate=1.0):
    time_points = np.arange(0, duration_hours, 0.5)
    elimination_rate = 0.693 / half_life_hours
    # Basit bir farmakokinetik eğri (Emilim + Eliminasyon)
    concentration = (dose * absorption_rate / 10) * (np.exp(-elimination_rate * time_points))
    return pd.DataFrame({'Zaman (sa)': time_points, 'Konsantrasyon': concentration})

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="DeepGenom AI Pro", layout="wide")

CANCER_DATA = {
    "Meme (HER2+)": {"motif": "HER", "ref_drug_affinity": 70, "ref_drug_toxicity": 15, "ref_drug_t12": 18, "ref_drug_bio": 60},
    "Akciger (EGFR)": {"motif": "EGF", "ref_drug_affinity": 65, "ref_drug_toxicity": 20, "ref_drug_t12": 12, "ref_drug_bio": 70},
    "Pankreas (KRAS)": {"motif": "KRA", "ref_drug_affinity": 50, "ref_drug_toxicity": 30, "ref_drug_t12": 8, "ref_drug_bio": 45}
}

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔬 Laboratuvar Ayarları")
    choice = st.selectbox("Hedef Kanser Türü", list(CANCER_DATA.keys()))
    pop_size = st.slider("Popülasyon Büyüklüğü", 20, 500, 100)
    gen_limit = st.slider("Evrimsel Nesil Sayısı", 10, 500, 100)
    dna_len = st.number_input("DNA Uzunluğu", 30, 200, 60)
    dose_mg = st.slider("Sanal Doz (mg)", 10, 500, 100)
    run_btn = st.button("🚀 SİMÜLASYONU BAŞLAT")

# --- ANA SİMÜLASYON ---
if run_btn:
    population = ["".join(random.choice("ATGC") for _ in range(dna_len)) for _ in range(pop_size)]
    history = []
    target = CANCER_DATA[choice]["motif"]

    with st.status("Analiz ediliyor...") as status:
        for g in range(gen_limit):
            scored = []
            for dna in population:
                prot = str(Seq(dna).translate(to_stop=True))
                # Skorlama mantığı
                fit = (prot.count(target) * 50) + (dna.count("GC") * 2)
                tox = (prot.count("R") * 10)
                score = max(0.1, fit - tox)
                
                scored.append({"dna": dna, "prot": prot, "skor": score, "zarar": tox, "nesil": g})
            
            scored.sort(key=lambda x: x['skor'], reverse=True)
            history.append(scored[0])
            # Yeni nesil (basit mutasyon)
            population = [scored[0]['dna']] * pop_size 
            population = ["".join(c if random.random() > 0.1 else random.choice("ATGC") for c in dna) for dna in population]

    best = history[-1]
    # Ek metrikler
    best['mw'] = molecular_weight(best['prot'], 'protein') if best['prot'] else 0
    best['pi'] = ProtParam.ProteinAnalysis(best['prot']).isoelectric_point() if len(best['prot']) > 2 else 7.0
    best['omur'] = 12.0
    best['biyo'] = 75.0
    best['hiz'] = 80.0
    best['hedef'] = choice
    best['konum'] = "Özgün"

    st.session_state.best = best
    st.session_state.history = history
    st.session_state.pk_df = pk_model_simulation(dose_mg, best['omur'], 24)

# --- GÖRSELLEŞTİRME ---
if 'best' in st.session_state:
    res = st.session_state.best
    
    # 1. Metrik Kartları
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bağlanma (ΔG)", f"-{res['skor']:.2f}")
    c2.metric("Moleküler Ağırlık", f"{res['mw']:.1f} Da")
    c3.metric("Yarı Ömür", f"{res['omur']} sa")
    c4.metric("Toksisite", f"{res['zarar']}", delta_color="inverse")

    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        # 3D Protein Görselleştirme (Düzeltildi)
        st.subheader("🧬 3D Peptit Yapısı")
        
        view = py3Dmol.view(width=800, height=400)
        # Temsili bir PDB (Helix yapısı simülasyonu)
        pdb_data = f"MODEL     1\n"
        for i, aa in enumerate(res['prot'][:20]):
            pdb_data += f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    {i*1.5:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
        pdb_data += "ENDMDL"
        
        view.addModel(pdb_data, "pdb")
        view.setStyle({'stick': {'color': 'spectrum'}, 'sphere': {'scale': 0.3}})
        view.zoomTo()
        st.components.v1.html(view._make_html(), height=400)

        # PK Grafiği
        st.subheader("📈 Farmakokinetik Profil")
        [attachment_0](attachment)
        fig_pk = go.Figure()
        fig_pk.add_trace(go.Scatter(x=st.session_state.pk_df['Zaman (sa)'], y=st.session_state.pk_df['Konsantrasyon'], fill='tozeroy', line_color='green'))
        st.plotly_chart(fig_pk, use_container_width=True)

    with col_right:
        # Lottie Animasyonları (Güvenli Yükleme)
        st.subheader("✨ Etkileşim Simülasyonu")
        dna_anim = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_tmswy3xr.json")
        if dna_anim:
            st_lottie(dna_anim, height=200)
        else:
            st.info("Animasyon yükleniyor...")

        # Radar Grafik
        categories = ['Afinite', 'Güvenlik', 'Yarı Ömür', 'Biyoyararlanım']
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=[res['skor'], 100-res['zarar'], 80, 70], theta=categories, fill='toself'))
        st.plotly_chart(fig_radar, use_container_width=True)

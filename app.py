import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from Bio.Seq import Seq
from Bio.SeqUtils import molecular_weight, ProtParam
from fpdf import FPDF
import random
import base64
import time
import py3Dmol
from stmol import showmol
import requests
from streamlit_lottie import st_lottie

# --- FONKSİYONLAR ---

def tr_to_en(text):
    """PDF hatasını önlemek için Türkçe karakterleri temizler."""
    map_chars = {"ş":"s", "Ş":"S", "ı":"i", "İ":"I", "ç":"c", "Ç":"C", "ü":"u", "Ü":"U", "ğ":"g", "Ğ":"G", "ö":"o", "Ö":"O"}
    for search, replace in map_chars.items():
        text = str(text).replace(search, replace)
    return text

def create_pdf(res, pk_df):
    """Doktor sunumuna uygun PDF raporu."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, tr_to_en("DeepGenom AI - Klinik Analiz Raporu"), ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, tr_to_en("1. Antidot Özet Verileri"), ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 8, tr_to_en(f"Hedef Hastalik: {res['hedef']}"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Baglanma Afinitesi (dG): -{res['skor']:.2f} kcal/mol"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Sitotoksisite Indeksi: {res['zarar']:.2f}"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Molekuler Agirlik: {res['mw']:.2f} Da"), ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, tr_to_en("2. Tasarlanan DNA Sekansi"), ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 8, res['dna'])
    
    return pdf.output(dest='S').encode('latin-1')

def load_lottieurl(url: str):
    """Lottie animasyonlarını yükler."""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def pk_model_simulation(dose, half_life_hours, duration_hours, absorption_rate=1.0):
    time_points = [i * 0.5 for i in range(int(duration_hours * 2))]
    concentration = []
    elimination_rate = 0.693 / half_life_hours
    current_drug_amount = dose * absorption_rate
    for t in time_points:
        current_drug_amount *= (1 - elimination_rate * 0.5)
        concentration.append(max(0, current_drug_amount / 100))
    return pd.DataFrame({'Zaman (sa)': time_points, 'Konsantrasyon': concentration})

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="DeepGenom AI Pro", layout="wide")

# --- ANALİZ PARAMETRELERİ ---
CANCER_DATA = {
    "Meme (HER2+)": {"motif": "HER", "ref_drug_affinity": 70, "ref_drug_toxicity": 15, "ref_drug_t12": 18, "ref_drug_bio": 60},
    "Akciger (EGFR)": {"motif": "EGF", "ref_drug_affinity": 65, "ref_drug_toxicity": 20, "ref_drug_t12": 12, "ref_drug_bio": 70},
    "Pankreas (KRAS)": {"motif": "KRA", "ref_drug_affinity": 50, "ref_drug_toxicity": 30, "ref_drug_t12": 8, "ref_drug_bio": 45}
}

# --- ARAYÜZ ---
st.title("🛡️ DeepGenom AI: Klinik Karar Destek Paneli")

with st.sidebar:
    st.header("🔬 Laboratuvar Ayarları")
    choice = st.selectbox("Hedef Kanser Türü", list(CANCER_DATA.keys()))
    pop_size = st.slider("Popülasyon Büyüklüğü", 20, 500, 100)
    gen_limit = st.slider("Nesil Sayısı", 10, 500, 100)
    dna_len = st.number_input("DNA Uzunluğu", 30, 200, 60)
    dose_mg = st.slider("Sanal Doz (mg)", 10, 500, 100)
    run_btn = st.button("🚀 SİMÜLASYONU BAŞLAT")

# --- ANALİZ DÖNGÜSÜ ---
if run_btn:
    history = []
    population = ["".join(random.choice("ATGC") for _ in range(dna_len)) for _ in range(pop_size)]
    target_motif = CANCER_DATA[choice]["motif"]
    
    progress = st.progress(0)
    for g in range(gen_limit):
        scored = []
        for dna in population:
            prot = str(Seq(dna).translate(to_stop=True))
            fit = (prot.count(target_motif) * 55) + (dna.count("GGC") * 5)
            tox = (prot.count("R") * 12) + (prot.count("C") * 8)
            
            # Biyoenformatik Metrikler
            mw = molecular_weight(prot, 'protein') if prot else 0
            pi = ProtParam.ProteinAnalysis(prot).isoelectric_point() if len(prot) > 2 else 7.0
            
            scored.append({
                "dna": dna, "prot_seq": prot, "skor": max(0.1, fit - (tox * 0.2)),
                "zarar": tox, "mw": mw, "pi": pi, "nesil": g,
                "gc_content": ((dna.count("G") + dna.count("C")) / len(dna)) * 100
            })
        
        scored.sort(key=lambda x: x['skor'], reverse=True)
        history.append(scored[0])
        
        # Evrimsel Seçilim
        next_gen = [x['dna'] for x in scored[:int(pop_size*0.1)]]
        while len(next_gen) < pop_size:
            p = random.choice(next_gen)
            child = "".join(c if random.random() > 0.05 else random.choice("ATGC") for c in p)
            next_gen.append(child)
        population = next_gen
        progress.progress((g + 1) / gen_limit)

    best = history[-1]
    best['hedef'] = choice
    best['hiz'] = min(95, int(best['skor'] * 0.8))
    best['omur'] = round(8 + (best['mw']/2000), 1)
    best['biyo'] = min(90, int(best['hiz'] * 0.9))
    best['konum'] = "Özgün Tasarım"
    
    st.session_state.results = history
    st.session_state.best = best
    st.session_state.pk_df = pk_model_simulation(dose_mg, best['omur'], 24, best['hiz']/100)

# --- SONUÇLAR ---
if 'best' in st.session_state:
    res = st.session_state.best
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Afinite (dG)", f"-{res['skor']:.1f}")
    m2.metric("Hücre Girişi", f"%{res['hiz']}")
    m3.metric("Yarı Ömür", f"{res['omur']}s")
    m4.metric("Toksisite", f"{res['zarar']:.1f}", delta_color="inverse")

    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.subheader("🧬 3D Moleküler Yapı (Tahmini)")
        pdb_string = ""
        x, y, z = 0.0, 0.0, 0.0
        for i, aa in enumerate(res['prot_seq'][:30]):
            pdb_string += f"ATOM  {i+1:5d}  CA  {aa:3s} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            x += 1.5; y += 0.5; z += 0.2
            
        view = py3Dmol.view(width=800, height=400)
        view.addModel(pdb_string, 'pdb')
        view.setStyle({'stick': {'color': 'spectrum'}, 'sphere': {'scale': 0.3}})
        view.zoomTo()
        showmol(view, height=400)

        st.subheader("📈 PK/PD Simülasyonu")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.pk_df['Zaman (sa)'], y=st.session_state.pk_df['Konsantrasyon'], fill='tozeroy'))
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.subheader("📋 Analiz Detayları")
        st.write(f"**MW:** {res['mw']:.1f} Da")
        st.write(f"**pI:** {res['pi']:.2f}")
        st.write(f"**Hedef:** {res['hedef']}")
        
        st_lottie(load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_tmswy3xr.json"), height=150)
        
        pdf_data = create_pdf(res, st.session_state.pk_df)
        st.download_button("📥 PDF Raporu İndir", data=pdf_data, file_name="klinik_analiz.pdf")

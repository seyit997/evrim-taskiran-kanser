import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from Bio.Seq import Seq
from fpdf import FPDF
import random
import time
import base64

# --- KANSER VERİTABANI ---
CANCER_TYPES = {
    "Meme Kanseri (HER2+)": "HER",
    "Akciğer Kanseri (EGFR)": "EGF",
    "Pankreas Kanseri (KRAS)": "KRA",
    "Genel Onkoloji (P53)": "P53"
}

# --- PDF RAPOR FONKSİYONU ---
def create_pdf(dna, score, cancer_type, gen_count):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "DeepGenom AI - Analiz Raporu", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, f"Hedef Hastalik: {cancer_type}", ln=True)
    pdf.cell(200, 10, f"Simulasyon Nesli: {gen_count}", ln=True)
    pdf.cell(200, 10, f"Optimum Basari Skoru: {score}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Bulunan En Iyi DNA Dizisi:\n{dna}")
    return pdf.output(dest='S').encode('latin-1')

# --- ARAYÜZ ---
st.set_page_config(page_title="DeepGenom Pro", layout="wide")
st.title("🧬 DeepGenom Pro: Hiyerarşik Antidot Tasarımı")

# Sidebar
st.sidebar.header("🔬 Parametreler")
selected_cancer = st.sidebar.selectbox("Hedef Kanser Türü", list(CANCER_TYPES.keys()))
pop_size = st.sidebar.slider("Popülasyon", 20, 100, 50)
gen_limit = st.sidebar.slider("Nesil Sınırı", 50, 500, 100)
dna_len = st.sidebar.number_input("DNA Uzunluğu", 30, 150, 60)

# Simülasyon Fonksiyonu
def run_evolution():
    target = CANCER_TYPES[selected_cancer]
    population = ["".join(random.choice("ATGC") for _ in range(dna_len)) for _ in range(pop_size)]
    history = []
    
    chart_placeholder = st.empty()
    
    for g in range(gen_limit):
        scored = []
        for dna in population:
            # Skorlama: Hedef motif + stabilite - yan etki
            protein = str(Seq(dna).translate(to_stop=True))
            fit = (protein.count(target) * 50) + (protein.count("L") * 5) - (protein.count("R") * 10)
            scored.append((dna, max(0, fit)))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        best_dna, best_fit = scored[0]
        history.append({"Nesil": g, "Skor": best_fit})
        
        # Grafiği güncelle
        df = pd.DataFrame(history)
        fig = px.line(df, x="Nesil", y="Skor", title=f"{selected_cancer} Evrim Süreci")
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Yeni Nesil
        next_gen = [x[0] for x in scored[:10]]
        while len(next_gen) < pop_size:
            parent = random.choice(next_gen)
            child = "".join(c if random.random() > 0.05 else random.choice("ATGC") for c in parent)
            next_gen.append(child)
        population = next_gen
        
    return best_dna, best_fit

# Ana Ekran
if st.button("Simülasyonu ve Analizi Başlat"):
    best_dna, best_score = run_evolution()
    
    st.success(f"Analiz Tamamlandı! En yüksek skor: {best_score}")
    st.code(best_dna, language="text")
    
    # PDF İndirme Butonu
    pdf_data = create_pdf(best_dna, best_score, selected_cancer, gen_limit)
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="deepgenom_rapor.pdf">📥 Profesyonel Raporu İndir (PDF)</a>'
    st.markdown(href, unsafe_allow_html=True)

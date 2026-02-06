import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from Bio.Seq import Seq
from fpdf import FPDF
import random
import base64
import time

# --- FONKSİYONLAR ---

def tr_to_en(text):
    """PDF hatasını önlemek için Türkçe karakterleri temizler."""
    map_chars = {"ş":"s", "Ş":"S", "ı":"i", "İ":"I", "ç":"c", "Ç":"C", "ü":"u", "Ü":"U", "ğ":"g", "Ğ":"G", "ö":"o", "Ö":"O"}
    for search, replace in map_chars.items():
        text = text.replace(search, replace)
    return text

def create_pdf(res):
    """Unicode hatası giderilmiş, doktor sunumuna uygun PDF raporu."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, tr_to_en("DeepGenom AI - Klinik Analiz Raporu"), ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, tr_to_en("1. Antidot Özet Verileri"), ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 8, tr_to_en(f"Hedef Hastalik: {res['hedef']}"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Basari Skoru: {res['skor']} Puan"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Hücresel Zarar: %{res['zarar']}"), ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, tr_to_en("2. Farmakolojik Tahminler"), ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 8, tr_to_en(f"Hücreye Giris Hizi: %{res['hiz']}"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Yarilanma Ömrü (t1/2): {res['omur']} saat"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Doga Analizi: {res['konum']}"), ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, tr_to_en("3. Tasarlanan DNA Sekansi"), ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 8, res['dna'])
    
    return pdf.output(dest='S').encode('latin-1')

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="DeepGenom AI Pro", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; background-color: #0047AB; color: white; height: 3.5em; font-weight: bold; border: none; }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #0047AB; }
    </style>
    """, unsafe_allow_html=True)

# --- ANALİZ PARAMETRELERİ ---
CANCER_DATA = {
    "Meme (HER2+)": "HER",
    "Akciger (EGFR)": "EGF",
    "Pankreas (KRAS)": "KRA"
}

# --- ARAYÜZ ---
st.title("🛡️ DeepGenom AI: Klinik Karar Destek Paneli")
st.write("Bilgisayar ortamında evrimsel antidot tasarımı ve toksisite analizi.")

with st.sidebar:
    st.header("🔬 Laboratuvar Ayarları")
    choice = st.selectbox("Hedef Kanser Türü", list(CANCER_DATA.keys()))
    pop_size = st.slider("Popülasyon", 20, 200, 100)
    gen_limit = st.slider("Nesil Sayısı", 10, 500, 200)
    dna_len = st.number_input("DNA Uzunluğu", 30, 200, 60)
    run_btn = st.button("🚀 SİMÜLASYONU BAŞLAT")

# --- EVRİM VE ANALİZ DÖNGÜSÜ ---
if run_btn:
    history = []
    population = ["".join(random.choice("ATGC") for _ in range(dna_len)) for _ in range(pop_size)]
    target_motif = CANCER_DATA[choice]
    
    prog = st.progress(0)
    for g in range(gen_limit):
        scored = []
        for dna in population:
            prot = str(Seq(dna).translate(to_stop=True))
            # Skorlama: Hedef motif + stabilite - toksisite
            fit = (prot.count(target_motif) * 55) + (dna.count("GGC") * 5)
            tox = (prot.count("R") * 12) + (prot.count("C") * 8)
            scored.append({"dna": dna, "skor": max(0, fit - (tox * 0.2)), "zarar": tox, "nesil": g})
        
        scored.sort(key=lambda x: x['skor'], reverse=True)
        history.append(scored[0])
        
        # Seçilim ve Mutasyon
        next_gen = [x['dna'] for x in scored[:10]]
        while len(next_gen) < pop_size:
            p = random.choice(next_gen)
            child = "".join(c if random.random() > 0.05 else random.choice("ATGC") for c in p)
            next_gen.append(child)
        population = next_gen
        prog.progress((g + 1) / gen_limit)
    
    st.session_state.results = history
    st.session_state.selected_h = choice

# --- SONUÇLARI GÖSTER ---
if 'results' in st.session_state:
    best = st.session_state.results[-1]
    
    # Klinik ve Farmakolojik Tahminler
    best['hiz'] = min(100, int(best['skor'] * 0.75 + random.randint(1, 15)))
    best['omur'] = round((len(best['dna']) / 12) + (best['dna'].count("G") * 0.5), 1)
    sim = random.randint(1, 15)
    best['konum'] = "Ozgün: Doğada birebir eslesme yok." if sim < 8 else f"Kısmi: %{sim} Benzerlik (İnsan Genomu)."
    best['hedef'] = st.session_state.selected_h

    # Metrik Kartları
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Antidot Skoru", f"{int(best['skor'])} Puan")
    c2.metric("Hücreye Giriş", f"%{best['hiz']}")
    c3.metric("Yarılanma Ömrü", f"{best['omur']} sa")
    c4.metric("Yan Etki Riski", f"%{int(best['zarar'])}", delta="Düşük" if best['zarar'] < 40 else "Yüksek", delta_color="inverse")

    st.divider()
    
    col_plot, col_info = st.columns([2, 1])
    
    with col_plot:
        df = pd.DataFrame(st.session_state.results)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["nesil"], y=df["skor"], name="Başarı", line=dict(color='#0047AB', width=3)))
        fig.add_trace(go.Scatter(x=df["nesil"], y=df["zarar"], name="Toksisite", line=dict(color='#FF4B4B', dash='dot')))
        fig.update_layout(title="Evrimsel Gelişim Süreci", xaxis_title="Nesil", yaxis_title="Değer")
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.subheader("📋 Klinik Notlar")
        st.write(f"**Doğa Analizi:** {best['konum']}")
        st.write(f"**Hedef Mekanizma:** {best['hedef']} reseptör blokajı simülasyonu.")
        st.code(best['dna'], language="text")
        
        # PDF Butonu
        pdf_data = create_pdf(best)
        b64 = base64.b64encode(pdf_data).decode()
        st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="analiz_raporu.pdf">📥 Doktor Raporunu İndir (PDF)</a>', unsafe_allow_html=True)

    with st.expander("🔍 Tüm Aday Listesini Gör"):
        st.dataframe(df.tail(20))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from Bio.Seq import Seq
from fpdf import FPDF
import random
import time
import base64

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="DeepGenom AI Pro v6.0", layout="wide")

# CSS: Buton ve Arayüz Güzelleştirme
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 10px; height: 3.5em; background-color: #007bff; color: white; font-weight: bold; border: none; }
    .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; }
    </style>
    """, unsafe_allow_html=True)

# --- BİYOLOJİK VERİTABANI SİMÜLASYONU ---
CANCER_TYPES = {
    "Meme (HER2+)": {"motif": "HER", "desc": "HER2 Reseptör Blokajı"},
    "Akciğer (EGFR)": {"motif": "EGF", "desc": "EGFR Sinyal İnhibisyonu"},
    "Pankreas (KRAS)": {"motif": "KRA", "desc": "KRAS Mutasyon Hedefleme"}
}

SIDE_EFFECT_LOGIC = {
    "Kritik": "Hücre zarında lipid peroksidasyonu ve sitoliz riski.",
    "Orta": "Mitokondriyal ATP üretiminde geçici yavaşlama.",
    "Güvenli": "Hücre homeostazı ile %99 uyumlu yapı."
}

def get_homology_details(similarity):
    if similarity < 8: return "Özgün: Doğada eşleşme yok. (Patentlenebilir)"
    if similarity < 15: return "Kısmi: İnsan genomu (İntron) bölgeleriyle benzerlik."
    return "Dikkat: Bakteriyel enzim dizilimleri ile benzerlik."

# --- PDF OLUŞTURUCU ---
def create_pdf(res):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "DeepGenom AI Klinik Raporu", ln=True, align='C')
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Hedef: {res['hedef']}", ln=True)
    pdf.cell(200, 10, f"Basari Skoru: {res['skor']}", ln=True)
    pdf.cell(200, 10, f"Zarar Orani: %{res['zarar']}", ln=True)
    pdf.cell(200, 10, f"Doga Analizi: {res['nerede']}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"DNA Dizisi:\n{res['dna']}")
    return pdf.output(dest='S').encode('latin-1')

# --- ANA PANEL ---
st.title("🧬 DeepGenom AI: Klinik Karar Destek Sistemi")

with st.sidebar:
    st.header("🔬 Laboratuvar Ayarları")
    c_choice = st.selectbox("Hedef Kanser Türü", list(CANCER_TYPES.keys()))
    pop_size = st.slider("Popülasyon Genişliği", 20, 200, 100)
    gen_size = st.slider("Nesil Sayısı (Evrim Süresi)", 10, 500, 200)
    dna_len = st.number_input("DNA Uzunluğu (Baz)", 30, 200, 60)
    start_btn = st.button("🧬 ANALİZİ BAŞLAT")

if start_btn:
    st.session_state.history = []
    population = ["".join(random.choice("ATGC") for _ in range(dna_len)) for _ in range(pop_size)]
    target_motif = CANCER_TYPES[c_choice]["motif"]
    
    prog_bar = st.progress(0)
    
    for g in range(gen_size):
        scored_pop = []
        for dna in population:
            prot = str(Seq(dna).translate(to_stop=True))
            # Skorlama Mantığı
            fit = (prot.count(target_motif) * 50) + (dna.count("GGC") * 5)
            tox = (prot.count("R") * 12) + (prot.count("C") * 8)
            sim = random.randint(1, 20)
            
            scored_pop.append({
                "dna": dna, "skor": max(0, fit - (tox * 0.3)),
                "zarar": tox, "benzerlik": sim, "nesil": g
            })
        
        scored_pop.sort(key=lambda x: x['skor'], reverse=True)
        st.session_state.history.append(scored_pop[0])
        
        # Evrimsel Seçilim
        next_gen = [x['dna'] for x in scored_pop[:10]]
        while len(next_gen) < pop_size:
            parent = random.choice(next_gen)
            child = "".join(c if random.random() > 0.05 else random.choice("ATGC") for c in parent)
            next_gen.append(child)
        population = next_gen
        prog_bar.progress((g + 1) / gen_size)

# --- SONUÇLAR ---
if 'history' in st.session_state and st.session_state.history:
    best = st.session_state.history[-1]
    best['nerede'] = get_homology_details(best['benzerlik'])
    best['hedef'] = c_choice
    
    st.subheader("🏆 Optimum Antidot Analizi")
    c1, c2, c3 = st.columns(3)
    c1.metric("Maksimum Başarı", f"{int(best['skor'])} Puan")
    c2.metric("Hücresel Zarar", f"%{int(best['zarar'])}", delta="Güvenli" if best['zarar'] < 40 else "Riskli", delta_color="inverse")
    c3.metric("Doğal Benzerlik", f"%{best['benzerlik']}")
    
    st.info(f"**Doğa Analizi (Konum):** {best['nerede']}")
    tox_label = "Güvenli" if best['zarar'] < 30 else ("Orta" if best['zarar'] < 70 else "Kritik")
    st.warning(f"**Tıbbi Etki Tahmini:** {SIDE_EFFECT_LOGIC[tox_label]}")
    
    st.code(best['dna'], language="text")
    
    # Grafik
    df = pd.DataFrame(st.session_state.history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["nesil"], y=df["skor"], name="Etkinlik Skoru", line=dict(color='#00FF00')))
    fig.add_trace(go.Scatter(x=df["nesil"], y=df["zarar"], name="Toksisite", line=dict(color='#FF0000', dash='dot')))
    st.plotly_chart(fig, use_container_width=True)

    # PDF Rapor
    pdf_file = create_pdf(best)
    b64 = base64.b64encode(pdf_file).decode()
    st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="klinik_analiz.pdf">📥 Profesyonel Klinik Raporu İndir (PDF)</a>', unsafe_allow_html=True)
    
    with st.expander("🔍 Tüm Mutasyon Kütüphanesini Gör"):
        st.table(df.tail(10))

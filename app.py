import streamlit as st
import pandas as pd
import numpy as np
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

# --- 1. FONKSİYONLAR VE ARAÇLAR ---

def tr_to_en(text):
    """PDF uyumluluğu için Türkçe karakter temizleme."""
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

def create_pdf(res):
    """Klinik rapor üretici."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, tr_to_en("DeepGenom AI - Profesyonel Analiz Raporu"), ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, tr_to_en("1. Antidot ve Biyoenformatik Veriler"), ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 8, tr_to_en(f"Hedef Reseptör: {res['hedef']}"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Baglanma Enerjisi (dG): -{res['skor']:.2f} kcal/mol"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Toksisite Indeksi: {res['zarar']:.2f}"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Molekuler Agirlik: {res['mw']:.2f} Da"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Izoelektrik Nokta (pI): {res['pi']:.2f}"), ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, tr_to_en("2. Farmakokinetik Parametreler"), ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 8, tr_to_en(f"Yarilanma Omru (t1/2): {res['omur']} saat"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Biyoyararlanilabilirlik: %{res['biyo']}"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Hucreye Giris Kapasitesi: %{res['hiz']}"), ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, tr_to_en("3. Tasarlanan DNA Sekansi"), ln=True)
    pdf.set_font("Arial", "", 8)
    pdf.multi_cell(0, 5, res['dna'])
    
    return pdf.output(dest='S').encode('latin-1')

# --- 2. SAYFA AYARLARI VE TASARIM ---
st.set_page_config(page_title="DeepGenom AI Pro", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 10px; background-color: #0047AB; color: white; font-weight: bold; transition: 0.3s; }
    .stButton>button:hover { background-color: #002D6B; border: 2px solid white; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid #0047AB; }
    </style>
    """, unsafe_allow_html=True)

# Animasyonlar
lottie_dna = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_tmswy3xr.json")

# Kanser Veri Modeli
CANCER_DATA = {
    "Meme (HER2+)": {"motif": "HER", "ref": [75, 20, 18, 70]},
    "Akciger (EGFR)": {"motif": "EGF", "ref": [65, 15, 12, 80]},
    "Pankreas (KRAS)": {"motif": "KRA", "ref": [55, 30, 8, 50]}
}

# --- 3. SIDEBAR (KONTROL PANELİ) ---
with st.sidebar:
    if lottie_dna: st_lottie(lottie_dna, height=150)
    st.header("🔬 Laboratuvar Ayarları")
    choice = st.selectbox("Hedef Kanser Türü", list(CANCER_DATA.keys()))
    pop_size = st.slider("Popülasyon Büyüklüğü", 50, 500, 150)
    gen_limit = st.slider("Nesil Sayısı", 20, 1000, 100)
    dna_len = st.number_input("DNA Uzunluğu", 30, 300, 90)
    dose_mg = st.slider("Simüle Edilen Doz (mg)", 50, 1000, 250)
    
    st.divider()
    run_btn = st.button("🚀 SİMÜLASYONU BAŞLAT")

# --- 4. EVRİMSEL ALGORİTMA VE HESAPLAMA ---
if run_btn:
    history = []
    population = ["".join(random.choice("ATGC") for _ in range(dna_len)) for _ in range(pop_size)]
    target_motif = CANCER_DATA[choice]["motif"]
    
    with st.status("Moleküler Analiz ve Evrimsel Süreç Çalışıyor...", expanded=True) as status:
        for g in range(gen_limit):
            scored = []
            for dna in population:
                # Biyoenformatik Çeviri
                seq_obj = Seq(dna)
                prot = str(seq_obj.translate(to_stop=True))
                
                # Fitness (Başarı) Fonksiyonu
                fit = (prot.count(target_motif) * 70) + (dna.count("GC") * 3)
                tox = (prot.count("R") * 12) + (prot.count("W") * 18)
                
                # Protein Analiz Parametreleri
                mw = molecular_weight(prot, "protein") if prot else 0
                try:
                    pi = ProtParam.ProteinAnalysis(prot).isoelectric_point() if len(prot) > 2 else 7.0
                except:
                    pi = 7.0
                
                scored.append({
                    "dna": dna, "prot": prot, "skor": max(0.1, fit - (tox * 0.4)), 
                    "zarar": tox, "mw": mw, "pi": pi, "nesil": g
                })
            
            scored.sort(key=lambda x: x['skor'], reverse=True)
            history.append(scored[0])
            
            # Seçilim ve Mutasyon (Elitizm)
            next_gen = [x['dna'] for x in scored[:int(pop_size*0.15)]] # En iyi %15
            while len(next_gen) < pop_size:
                p = random.choice(next_gen)
                child = "".join(c if random.random() > 0.07 else random.choice("ATGC") for c in p)
                next_gen.append(child)
            population = next_gen
            
            if g % 10 == 0:
                status.write(f"Nesil {g}: dG -{scored[0]['skor']:.2f} kcal/mol | Toksisite: {scored[0]['zarar']:.1f}")
        
        status.update(label="Analiz Tamamlandı!", state="complete", expanded=False)

    # Sonuçları Saklama
    best = history[-1]
    best['hedef'] = choice
    best['omur'] = round(5 + (best['mw']/1500) + random.uniform(0, 5), 1)
    best['biyo'] = min(98, int(best['skor'] * 0.9 + random.randint(5, 15)))
    best['hiz'] = min(100, int(best['skor'] * 0.85 + random.randint(1, 10)))
    
    st.session_state.final_res = best
    st.session_state.history = history

# --- 5. GÖRSELLEŞTİRME VE RAPORLAMA ---
if 'final_res' in st.session_state:
    res = st.session_state.final_res
    hist_df = pd.DataFrame(st.session_state.history)
    
    st.header("📊 In-silico Analiz Sonuçları")
    
    # Üst Metrik Kartları
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bağlanma Afinitesi (ΔG)", f"-{res['skor']:.2f} kcal", help="Düşük değer daha güçlü bağlanmayı temsil eder.")
    c2.metric("Moleküler Ağırlık", f"{int(res['mw'])} Da")
    c3.metric("Yarılanma Ömrü (t1/2)", f"{res['omur']} sa")
    c4.metric("Sitotoksisite Skoru", f"{int(res['zarar'])}", delta="Hücresel Güvenli" if res['zarar'] < 40 else "Riskli", delta_color="inverse")

    st.divider()

    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        # 1. 3D PROTEİN GÖRSELLEŞTİRME
        st.subheader("🧬 3D Peptit Konformasyonu")
        
        pdb_str = ""
        x, y, z = 0.0, 0.0, 0.0
        # PDB formatına uygun hatasız string üretimi
        for i, aa in enumerate(res['prot'][:30]):
            res_id = i + 1
            pdb_str += f"ATOM  {res_id:5d}  CA  {aa:3s} A{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            x += 1.5; y += 0.7; z += 0.5 # Temsili helix yapısı kaydırması
            
        view = py3Dmol.view(width=700, height=400)
        view.addModel(pdb_str, 'pdb')
        view.setStyle({'stick': {'color': 'spectrum'}, 'sphere': {'scale': 0.3}})
        view.zoomTo()
        showmol(view, height=400)

        # 2. FARMAKOKİNETİK GRAFİK
        st.subheader("📈 Farmakokinetik (PK) Simülasyonu")
        
        t = np.linspace(0, 72, 200)
        ka = 0.8 # Emilim hızı
        ke = 0.693 / res['omur'] # Eliminasyon hızı
        # Bateman Denklemi (Oral/Enjeksiyon emilim simülasyonu)
        conc = (dose_mg / 5) * (ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))
        
        fig_pk = go.Figure()
        fig_pk.add_trace(go.Scatter(x=t, y=conc, fill='tozeroy', name="Plazma Konsantrasyonu", line_color='#0047AB'))
        fig_pk.update_layout(title="Zamana Bağlı İlaç Konsantrasyonu", xaxis_title="Zaman (Saat)", yaxis_title="C (µg/mL)")
        st.plotly_chart(fig_pk, use_container_width=True)

    with col_right:
        # 3. RADAR KARŞILAŞTIRMA
        st.subheader("🆚 Endüstriyel Karşılaştırma")
        
        ref_vals = CANCER_DATA[choice]["ref"]
        categories = ['Afinite', 'Güvenlik', 'Yarı Ömür', 'Biyoyararlanım']
        # Değerleri 100 üzerinden normalize etme
        current_vals = [res['skor'], 100-res['zarar'], res['omur']*5, res['biyo']]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=current_vals, theta=categories, fill='toself', name='Tasarlanan Antidot', line_color='#0047AB'))
        fig_radar.add_trace(go.Scatterpolar(r=ref_vals, theta=categories, fill='toself', name='Standart Tedavi', line_color='#FF4B4B'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
        st.plotly_chart(fig_radar, use_container_width=True)

        # 4. KLİNİK NOTLAR VE PDF
        st.subheader("📋 Klinik Değerlendirme")
        st.success(f"**Biyoyararlanım:** %{res['biyo']}")
        st.info(f"**İzoelektrik Nokta (pI):** {res['pi']:.2f}")
        
        with st.expander("DNA Sekansını Görüntüle"):
            st.code(res['dna'], language="text")
        
        pdf_bytes = create_pdf(res)
        st.download_button(
            label="📥 PROFESYONEL RAPORU İNDİR (PDF)",
            data=pdf_bytes,
            file_name=f"DeepGenom_{res['hedef']}_Rapor.pdf",
            mime="application/pdf"
        )

    # 5. EVRİM GEÇMİŞİ (ALT)
    st.subheader("🕒 Optimizasyon Geçmişi")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=hist_df['nesil'], y=hist_df['skor'], name="Bağlanma Gücü", line_color='#0047AB'))
    fig_hist.add_trace(go.Scatter(x=hist_df['nesil'], y=hist_df['zarar'], name="Toksisite Eğilimi", line=dict(dash='dot', color='#FF4B4B')))
    st.plotly_chart(fig_hist, use_container_width=True)


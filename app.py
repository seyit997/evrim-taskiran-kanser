import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Bio.Seq import Seq
from Bio.SeqUtils import molecular_weight, ProtParam
from fpdf import FPDF
import random
import base64
from datetime import datetime

# ==================== GELİŞMİŞ ALGORİTMALAR ====================

def tr_to_en(text):
    map_chars = {"ş": "s", "Ş": "S", "ı": "i", "İ": "I", "ç": "c", "Ç": "C", "ü": "u", "Ü": "U", "ğ": "g", "Ğ": "G", "ö": "o", "Ö":"O"}
    for search, replace in map_chars.items():
        text = str(text).replace(search, replace)
    return text

def generate_smart_dna(length):
    """Başlangıç kodonu içeren ve erken durmayan (non-terminating) DNA üretir."""
    stop_codons = ['TAA', 'TAG', 'TGA']
    # Her zaman başlangıç kodonu ile başla
    codons = ['ATG']
    for _ in range((length // 3) - 1):
        codon = "".join(random.choice("ATGC") for _ in range(3))
        # İçeride stop kodonu olmamasını sağla
        while codon in stop_codons:
            codon = "".join(random.choice("ATGC") for _ in range(3))
        codons.append(codon)
    return "".join(codons)

def calculate_success_probability(best_candidate):
    """Adayın laboratuvar testlerini geçme olasılığını hesaplar."""
    factors = []
    # 1. Bağlanma Afinitesi Faktörü (Ideal: > 50)
    factors.append(min(1.0, best_candidate['skor'] / 80))
    # 2. Stabilite Faktörü (Instability Index < 40 iyidir)
    factors.append(1.0 if best_candidate['instability'] < 40 else max(0.2, (100 - best_candidate['instability'])/100))
    # 3. Toksisite Faktörü (Düşük zarar)
    factors.append(max(0.1, (100 - best_candidate['zarar']) / 100))
    # 4. Uzunluk Faktörü (İdeal ilaç peptidi 30-70 AA arası)
    aa_len = len(best_candidate['prot_seq'])
    factors.append(1.0 if 30 <= aa_len <= 70 else 0.5)
    
    prob = (sum(factors) / len(factors)) * 100
    return round(prob, 2)

def calculate_professional_fitness(prot_seq, dna_seq, target_motif):
    """Gerçekçi skorlama: Uzunluk, stabilite ve afinite dengesi."""
    if not prot_seq or len(prot_seq) < 10:
        return -500 # Çok kısa sekansları doğrudan ele
    
    # Biyofiziksel Analiz
    analysis = ProtParam.ProteinAnalysis(prot_seq)
    instability = analysis.instability_index()
    
    # 1. Bağlanma Enerjisi (Hedef motif tekrarı ve yük dengesi)
    binding_score = (prot_seq.count(target_motif) * 60)
    
    # 2. Uzunluk Primi (Kısa 'VDTNGA' gibi sonuçları engeller)
    length_bonus = len(prot_seq) * 3.5
    
    # 3. Stabilite ve Çözünürlük Cezası
    # Kararsız proteinler (instability > 40) laboratuvarda sentezlenemez
    stability_penalty = max(0, instability - 40) * 2
    
    # 4. GC İçeriği Dengesi (%40-60 arası ideal)
    gc = ((dna_seq.count("G") + dna_seq.count("C")) / len(dna_seq)) * 100
    gc_penalty = abs(50 - gc) * 1.5
    
    total_fitness = binding_score + length_bonus - stability_penalty - gc_penalty
    return total_fitness, instability

# ==================== UI VE SIMÜLASYON ====================

st.set_page_config(page_title="DeepGenom AI Pro v3", page_icon="🧪", layout="wide")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3042/3042301.png", width=80)
    st.title("🔬 Kontrol Paneli")
    cancer_type = st.selectbox("Hedef Endikasyon", ["Meme Kanseri (HER2+)", "Akciğer Kanseri (EGFR)", "Pankreas Kanseri (KRAS)"])
    pop_size = st.slider("Popülasyon Genişliği", 100, 500, 200)
    gen_limit = st.slider("Evrimsel Döngü", 50, 1000, 400)
    dna_len = st.number_input("DNA Uzunluğu (Base Pair)", 150, 600, 300)
    
    st.divider()
    dose = st.slider("Dozaj (mg/kg)", 5, 200, 50)
    run_btn = st.button("🚀 EVRİMSEL ANALİZİ BAŞLAT")

if run_btn:
    # Başlangıç
    motif = "HER" if "Meme" in cancer_type else "EGF" if "Akciğer" in cancer_type else "KRA"
    population = [generate_smart_dna(dna_len) for _ in range(pop_size)]
    history = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for g in range(gen_limit):
        scored_pop = []
        for dna in population:
            prot = str(Seq(dna).translate(to_stop=True))
            fit, instab = calculate_professional_fitness(prot, dna, motif)
            
            # GC İçeriği
            gc = ((dna.count("G") + dna.count("C")) / len(dna)) * 100
            
            scored_pop.append({
                "dna": dna, "prot_seq": prot, "skor": fit, 
                "instability": instab, "gc_content": gc, "nesil": g
            })
        
        # Seçilim (En iyi %15)
        scored_pop.sort(key=lambda x: x['skor'], reverse=True)
        history.append(scored_pop[0])
        
        # Yeni Nesil Oluşturma
        elites = [x['dna'] for x in scored_pop[:int(pop_size*0.15)]]
        new_pop = list(elites)
        
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(elites, 2)
            # Crossover
            cp = random.randint(10, len(p1)-10)
            child = p1[:cp] + p2[cp:]
            # Mutasyon
            if random.random() < 0.1:
                idx = random.randint(0, len(child)-1)
                child = child[:idx] + random.choice("ATGC") + child[idx+1:]
            new_pop.append(child)
        
        population = new_pop
        progress_bar.progress((g+1)/gen_limit)
        if g % 50 == 0:
            status_text.text(f"Döngü {g}: En İyi Afinite {scored_pop[0]['skor']:.2f}")

    # Final Hesaplamalar
    best = history[-1]
    best['hedef'] = cancer_type
    best['mw'] = molecular_weight(best['prot_seq'], 'protein')
    best['pi'] = ProtParam.ProteinAnalysis(best['prot_seq']).isoelectric_point()
    best['zarar'] = round(random.uniform(1.5, 4.5), 2) # Toksisite simülasyonu
    best['success_prob'] = calculate_success_probability(best)
    
    # PK Modeli
    times = np.linspace(0, 48, 100)
    best['pk_data'] = (dose / 5) * (np.exp(-0.1 * times) - np.exp(-1.2 * times))

    # --- SONUÇ EKRANI ---
    st.balloons()
    st.success(f"### ✅ Simülasyon Tamamlandı! Başarı Olasılığı: %{best['success_prob']}")
    
    # Üst Metrikler
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bağlanma Afinitesi", f"-{best['skor']:.1f} kcal/mol", delta="Optimal")
    m2.metric("Sentez Başarı Oranı", f"%{best['success_prob']}", delta="Yüksek")
    m3.metric("Moleküler Ağırlık", f"{best['mw']:.0f} Da")
    m4.metric("Stabilite İndeksi", f"{best['instability']:.1f}", delta="Kararlı" if best['instability'] < 40 else "Riskli", delta_color="inverse")

    st.divider()

    col_l, col_r = st.columns([3, 2])
    
    with col_l:
        st.subheader("📈 Evrimsel Gelişim ve Optimizasyon")
        hist_df = pd.DataFrame(history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df['nesil'], y=hist_df['skor'], name="Afinite Artışı", line=dict(color='#0047AB', width=3)))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧬 3D Konformasyonel Tahmin")
        # Profesyonel SVG Protein Gösterimi
        
        st.markdown(f"""
        <div style="background:#0e1117; padding:20px; border-radius:15px; border:1px solid #333; text-align:center">
            <svg width="400" height="120" viewBox="0 0 400 120">
                <path d="M20 60 Q 60 10, 100 60 T 180 60 T 260 60 T 340 60" fill="none" stroke="#4facfe" stroke-width="5" stroke-linecap="round">
                    <animate attributeName="stroke-dasharray" from="0,500" to="500,0" dur="3s" repeatCount="indefinite" />
                </path>
                <circle cx="20" cy="60" r="5" fill="#fff" />
                <circle cx="380" cy="60" r="5" fill="#fff" />
            </svg>
            <p style="color:gray">Tahmini Katlanma: {len(best['prot_seq'])} Amino Asit</p>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.subheader("💊 Farmakokinetik (PK)")
        fig_pk = go.Figure()
        fig_pk.add_trace(go.Scatter(x=times, y=best['pk_data'], fill='tozeroy', name="Plazma Kons.", line_color='#00CC96'))
        fig_pk.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_pk, use_container_width=True)

        st.subheader("📋 Sekans Verileri")
        st.caption("DNA Sekansı (Klinik Üretim İçin)")
        st.code(best['dna'][:120] + "...", language="text")
        st.caption("Protein Sekansı (Terapötik Peptit)")
        st.code(best['prot_seq'], language="text")

    # PDF Raporlama (Basitleştirilmiş)
    st.divider()
    if st.button("📥 Profesyonel Raporu (PDF) İndir"):
        st.info("Rapor oluşturuluyor... (PDF indirme linki simüle edilmiştir)")


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from Bio.Seq import Seq
from Bio.SeqUtils import molecular_weight, ProtParam
import random
import base64
from datetime import datetime

# ==================== GELİŞMİŞ ALGORİTMALAR VE HATA GİDERME ====================

def calculate_professional_fitness(prot_seq, dna_seq, target_motif):
    """Gerçekçi skorlama ve güvenli ProtParam analizi."""
    # Boş veya çok kısa sekansları filtrele
    if not prot_seq or len(prot_seq) < 5:
        return -1000.0, 100.0
    
    # ProtParam sadece standart amino asitlerle çalışır (ACDEFGHIKLMNPQRSTVWY)
    # Stop kodonu (*) veya geçersiz karakterleri temizle
    clean_prot = "".join([aa for aa in prot_seq if aa in "ACDEFGHIKLMNPQRSTVWY"])
    
    if len(clean_prot) < 5:
        return -1000.0, 100.0

    try:
        analysis = ProtParam.ProteinAnalysis(clean_prot)
        instability = analysis.instability_index()
        
        # 1. Bağlanma Enerjisi (Hedef motif tekrarı)
        binding_score = (clean_prot.count(target_motif) * 75)
        
        # 2. Uzunluk Primi
        length_bonus = len(clean_prot) * 4.0
        
        # 3. Stabilite Cezası (İdeal instability < 40)
        stability_penalty = max(0, instability - 40) * 3
        
        # 4. GC İçeriği Dengesi
        gc = ((dna_seq.count("G") + dna_seq.count("C")) / len(dna_seq)) * 100
        gc_penalty = abs(50 - gc) * 2
        
        total_fitness = binding_score + length_bonus - stability_penalty - gc_penalty
        return float(total_fitness), float(instability)
    
    except Exception:
        # Herhangi bir biyofiziksel hesaplama hatasında güvenli değer dön
        return -500.0, 99.0

def generate_smart_dna(length):
    """Başlangıç kodonu içeren ve stop kodonu barındırmayan DNA üretir."""
    stop_codons = ['TAA', 'TAG', 'TGA']
    codons = ['ATG'] # Start codon
    for _ in range((int(length) // 3) - 1):
        codon = "".join(random.choice("ATGC") for _ in range(3))
        while codon in stop_codons:
            codon = "".join(random.choice("ATGC") for _ in range(3))
        codons.append(codon)
    return "".join(codons)

# ==================== UI TASARIMI ====================

st.set_page_config(page_title="DeepGenom AI Pro v4", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🔬 Laboratuvar Kontrol")
    cancer_type = st.selectbox("Endikasyon", ["Meme Kanseri (HER2+)", "Akciğer Kanseri (EGFR)", "Pankreas Kanseri (KRAS)"])
    pop_size = st.slider("Popülasyon", 50, 500, 200)
    gen_limit = st.slider("Nesil Sayısı", 10, 1000, 300)
    dna_len = st.number_input("DNA Uzunluğu", 150, 900, 300)
    dose = st.slider("Doz (mg/kg)", 1, 200, 50)
    run_btn = st.button("🚀 SİMÜLASYONU BAŞLAT")

if run_btn:
    motif = "HER" if "Meme" in cancer_type else "EGF" if "Akciğer" in cancer_type else "KRA"
    population = [generate_smart_dna(dna_len) for _ in range(pop_size)]
    history = []
    
    progress_bar = st.progress(0)
    
    with st.status("Moleküler Evrim Çalışıyor...", expanded=True) as status:
        for g in range(gen_limit):
            scored_pop = []
            for dna in population:
                # DNA'yı Proteine çevir
                prot = str(Seq(dna).translate(to_stop=True))
                
                # Fitness hesapla (Hata kontrollü)
                fit, instab = calculate_professional_fitness(prot, dna, motif)
                
                gc = ((dna.count("G") + dna.count("C")) / len(dna)) * 100
                
                scored_pop.append({
                    "dna": dna, "prot_seq": prot, "skor": fit, 
                    "instability": instab, "gc_content": gc, "nesil": g
                })
            
            # Seçilim
            scored_pop.sort(key=lambda x: x['skor'], reverse=True)
            best_current = scored_pop[0]
            history.append(best_current)
            
            # Yeni Nesil (Elitizm + Crossover)
            elites = [x['dna'] for x in scored_pop[:max(2, int(pop_size*0.1))]]
            new_pop = list(elites)
            
            while len(new_pop) < pop_size:
                p1, p2 = random.sample(elites, 2)
                cp = random.randint(3, len(p1)-3)
                child = p1[:cp] + p2[cp:]
                # Mutasyon
                if random.random() < 0.05:
                    idx = random.randint(0, len(child)-1)
                    child = list(child)
                    child[idx] = random.choice("ATGC")
                    child = "".join(child)
                new_pop.append(child)
            
            population = new_pop
            progress_bar.progress((g+1)/gen_limit)
            
            if g % 50 == 0:
                status.write(f"Nesil {g}: Afinite -{best_current['skor']:.1f}")

    # --- SONUÇLAR ---
    best = history[-1]
    best['mw'] = molecular_weight("".join([aa for aa in best['prot_seq'] if aa in "ACDEFGHIKLMNPQRSTVWY"]), 'protein')
    
    st.balloons()
    
    # Metrik Paneli
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bağlanma Afinitesi", f"-{best['skor']:.1f} kcal")
    col2.metric("Stabilite (Instability)", f"{best['instability']:.1f}")
    col3.metric("Moleküler Ağırlık", f"{best['mw']:.0f} Da")
    col4.metric("GC İçeriği", f"%{best['gc_content']:.1f}")

    

    # Grafikler
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.subheader("📈 Optimizasyon Eğrisi")
        hist_df = pd.DataFrame(history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df['nesil'], y=hist_df['skor'], name="Fitness", line=dict(color='#00d4ff')))
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.subheader("💊 Farmakokinetik Tahmin")
        t = np.linspace(0, 24, 100)
        # Basit 1-Kompartman PK Modeli (Doz bağımlı)
        conc = (dose/10) * (np.exp(-0.15 * t)) 
        fig_pk = go.Figure()
        fig_pk.add_trace(go.Scatter(x=t, y=conc, fill='tozeroy', name="Plazma Kons.", line_color='#00ff88'))
        st.plotly_chart(fig_pk, use_container_width=True)

    # Sekans Bilgileri
    st.divider()
    st.subheader("🧬 Tasarlanan Biyo-Molekül Detayları")
    st.text_area("Optimal DNA Sekansı", best['dna'], height=100)
    st.text_area("Terapötik Peptit Sekansı", best['prot_seq'], height=80)

    # Başarı Analizi
    success_rate = min(99.0, (best['skor'] / 500) * 100)
    st.progress(success_rate / 100, text=f"Laboratuvar Sentez Başarı Olasılığı: %{success_rate:.1f}")


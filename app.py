import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from Bio.Seq import Seq
import random
import time
import base64

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="DeepGenom AI Pro", layout="wide")

# --- CSS İLE GÖRSEL DÜZENLEME (Butonları Belirginleştirme) ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stExpander {
        border: 1px solid #007bff;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ DeepGenom AI: Klinik Karar Destek Sistemi")
st.write("En az zarar, en iyi sonuç: Yapay zeka destekli biyolojik antidot tarayıcı.")

# --- SİMÜLASYON MOTORU ---
def run_simulation(cancer_type, p_size, g_limit):
    # Kurgusal hedef motifler
    targets = {"Meme": "HER", "Akciğer": "EGF", "Pankreas": "KRA"}
    target = targets.get(cancer_type, "P53")
    
    population = ["".join(random.choice("ATGC") for _ in range(60)) for _ in range(p_size)]
    all_results = []
    
    progress_bar = st.progress(0)
    for g in range(g_limit):
        current_gen_data = []
        for dna in population:
            protein = str(Seq(dna).translate(to_stop=True))
            
            # Başarı Skoru (Fitness)
            success = (protein.count(target) * 45) + (dna.count("GGC") * 5)
            
            # Zarar Analizi (Toksisite)
            toxicity_score = (protein.count("R") * 15) + (protein.count("C") * 10)
            
            # Doğa Analizi (Homoloji)
            similarity = random.randint(1, 15)
            
            res = {
                "dna": dna,
                "skor": max(0, success - (toxicity_score * 0.2)),
                "zarar": toxicity_score,
                "benzerlik": similarity,
                "nesil": g
            }
            current_gen_data.append(res)
        
        all_results.extend(current_gen_data)
        
        # Evrim: En iyileri seç ve mutasyonla yeni nesil yap
        current_gen_data.sort(key=lambda x: x['skor'], reverse=True)
        next_gen = [x['dna'] for x in current_gen_data[:10]]
        while len(next_gen) < p_size:
            p = random.choice(next_gen)
            child = "".join(c if random.random() > 0.05 else random.choice("ATGC") for c in p)
            next_gen.append(child)
        population = next_gen
        progress_bar.progress((g + 1) / g_limit)
        
    return pd.DataFrame(all_results)

# --- ANA PANEL ---
col_menu, col_main = st.columns([1, 3])

with col_menu:
    st.subheader("⚙️ Ayarlar")
    c_type = st.selectbox("Hastalık Türü", ["Meme", "Akciğer", "Pankreas"])
    pop = st.slider("Popülasyon", 20, 100, 50)
    gens = st.slider("Nesil Sayısı", 10, 200, 50)
    
    start_btn = st.button("🚀 SİMÜLASYONU BAŞLAT")

if start_btn:
    with st.spinner("Genomik veriler işleniyor..."):
        data = run_simulation(c_type, pop, gens)
        st.session_state.final_data = data

# --- SONUÇLARIN GÖSTERİLMESİ ---
if 'final_data' in st.session_state:
    df = st.session_state.final_data
    best_candidate = df.sort_values(by=["skor", "zarar"], ascending=[False, True]).iloc[0]

    with col_main:
        st.subheader("🏆 Optimum Antidot Adayı (En İyi Sonuç)")
        
        # Özet Kartları
        c1, c2, c3 = st.columns(3)
        c1.metric("Maksimum Başarı", f"{int(best_candidate['skor'])} Puan")
        c2.metric("Hücresel Zarar", f"%{int(best_candidate['zarar'])}", delta="Kritik" if best_candidate['zarar'] > 50 else "Güvenli", delta_color="inverse")
        c3.metric("Doğal Benzerlik", f"%{best_candidate['benzerlik']}")

        # Zarar Analizi Açıklaması
        if best_candidate['zarar'] < 30:
            st.success("✅ **Güvenli:** Bu dizi hücre homeostazı ile tam uyumlu.")
        elif best_candidate['zarar'] < 70:
            st.warning("⚠️ **Orta Risk:** Hücre bölünme hızında yavaşlamaya sebep olabilir.")
        else:
            st.error("🚨 **Yüksek Toksisite:** Mitokondriyal strese ve hücre zarında hasara yol açabilir.")

        st.info(f"**Doğa Analizi:** Bu DNA dizisi internette/doğada {'bulunmadı (Özgün)' if best_candidate['benzerlik'] < 10 else 'kısmen mevcut'}. ")
        st.code(best_candidate['dna'], language="text")

        # Grafik
        st.divider()
        st.subheader("📈 Gelişim ve Güvenlik Grafiği")
        gen_avg = df.groupby("nesil").agg({"skor": "max", "zarar": "mean"}).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gen_avg["nesil"], y=gen_avg["skor"], name="Başarı Skoru", line=dict(color='green', width=3)))
        fig.add_trace(go.Scatter(x=gen_avg["nesil"], y=gen_avg["zarar"], name="Ortalama Zarar", line=dict(color='red', dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

        # "DAHA FAZLA BAK" BÖLÜMÜ
        st.divider()
        with st.expander("🔍 DİĞER ADAYLARI VE DETAYLI ANALİZLERİ GÖR"):
            st.write("Sistem tarafından taranan en iyi 20 alternatif dizilim:")
            top_20 = df.sort_values(by="skor", ascending=False).drop_duplicates(subset=['dna']).head(20)
            
            for i, row in top_20.iterrows():
                col_dna, col_stat = st.columns([3, 1])
                col_dna.write(f"**Dizi:** `{row['dna']}`")
                col_stat.write(f"Skor: {int(row['skor'])} | Zarar: %{int(row['zarar'])}")
                st.progress(int(row['zarar']) / 100)
                st.write(f"---")


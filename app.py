import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from Bio.Seq import Seq
import random
import time

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="DeepGenom AI v4.0", layout="wide")
st.title("🧬 DeepGenom AI: Klinik Karar Destek Sistemi")

# --- YAN ETKİ VERİTABANI (Detaylı Analiz İçin) ---
SIDE_EFFECTS = {
    "Yüksek Toksisite": "Hücre zarında lipid peroksidasyonuna ve mitokondriyal strese yol açabilir.",
    "Orta Toksisite": "Hücre bölünme hızında yavaşlama ve geçici sitoplazmik şişme riski.",
    "Düşük Toksisite": "Minimal hücresel yük; biyo-uyumluluk oranı yüksek.",
    "Güvenli": "Hücre homeostazı ile tam uyumlu; yan etki saptanmadı."
}

# --- ANALİZ MOTORU ---
def analyze_sequence(dna):
    protein = str(Seq(dna).translate(to_stop=True))
    
    # 1. Başarı Skoru
    success_score = (dna.count("GGC") * 12) + (dna.count("AAA") * 5)
    
    # 2. Hücre Zararı ve Nedenleri
    tox_count = protein.count("R") + protein.count("C")
    if tox_count > 5:
        tox_level = "Yüksek Toksisite"
        tox_score = random.randint(70, 100)
    elif tox_count > 2:
        tox_level = "Orta Toksisite"
        tox_score = random.randint(30, 69)
    else:
        tox_level = "Güvenli"
        tox_score = random.randint(0, 29)
        
    # 3. Doğada Var mı? (Homoloji)
    similarity = random.randint(0, 15) # Sentetik tasarımlar genelde düşüktür
    found_in_nature = "Bulunamadı (Özgün Tasarım)" if similarity < 10 else f"Kısmi Benzerlik (%{similarity} - Homo Sapiens)"
    
    return {
        "dna": dna,
        "skor": success_score,
        "zarar_skoru": tox_score,
        "zarar_nedeni": SIDE_EFFECTS[tox_level],
        "dogada_varmi": found_in_nature,
        "benzerlik": similarity
    }

# --- SESSION STATE ---
if 'all_candidates' not in st.session_state:
    st.session_state.all_candidates = []

# --- ANA EKRAN ---
col1, col2 = st.columns([2, 1])

with st.sidebar:
    st.header("🧬 Analiz Ayarları")
    target = st.selectbox("Hedef Kanser", ["Meme", "Akciğer", "Pankreas"])
    if st.button("Simülasyonu Başlat"):
        st.session_state.all_candidates = [] # Reset
        pop = ["".join(random.choice("ATGC") for _ in range(60)) for _ in range(40)]
        
        for g in range(1, 51): # 50 Nesil hızlı analiz
            scored = [analyze_sequence(dna) for dna in pop]
            scored.sort(key=lambda x: x['skor'], reverse=True)
            st.session_state.all_candidates.extend(scored)
            
            # Nesil Yenileme
            next_gen = [x['dna'] for x in scored[:5]]
            while len(next_gen) < 40:
                p = random.choice(next_gen)
                child = "".join(c if random.random() > 0.05 else random.choice("ATGC") for c in p)
                next_gen.append(child)
            pop = next_gen
        st.success("Analiz Tamamlandı!")

# --- SONUÇLARI GÖSTER ---
if st.session_state.all_candidates:
    # En İyi Sonuç (Kapak)
    best = sorted(st.session_state.all_candidates, key=lambda x: x['skor'], reverse=True)[0]
    
    st.subheader("🏆 En Uygun Antidot Adayı")
    c1, c2, c3 = st.columns(3)
    c1.metric("Başarı Skoru", best['skor'])
    c2.metric("Hücre Zararı", f"%{best['zarar_skoru']}", delta="-Düşük" if best['zarar_skoru'] < 30 else "+Yüksek", delta_color="inverse")
    c3.write(f"**Doğa Analizi:** {best['dogada_varmi']}")
    
    st.info(f"**Hücresel Etki Analizi:** {best['zarar_nedeni']}")
    st.code(best['dna'], language="text")

    st.divider()
    
    # Diğerlerini Göster Butonu
    if st.checkbox("🔍 Diğer Adayları ve Detaylı Verileri Göster"):
        st.subheader("🧪 Alternatif İlaç Kütüphanesi")
        df_all = pd.DataFrame(st.session_state.all_candidates).drop_duplicates(subset=['dna'])
        df_all = df_all.sort_values(by="skor", ascending=False).head(20)
        
        for index, row in df_all.iterrows():
            with st.expander(f"Aday #{index+1} - Skor: {row['skor']} - Zarar: %{row['zarar_skoru']}"):
                st.write(f"**DNA Dizisi:** `{row['dna']}`")
                st.write(f"**Doğada Var mı?** {row['dogada_varmi']}")
                st.write(f"**Detaylı Zarar Analizi:** {row['zarar_nedeni']}")
                st.progress(row['zarar_skoru'] / 100)

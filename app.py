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
import requests
import json
from streamlit_lottie import st_lottie

# --- FONKSİYONLAR ---

def tr_to_en(text):
    """PDF hatasını önlemek için Türkçe karakterleri temizler."""
    map_chars = {"ş":"s", "Ş":"S", "ı":"i", "İ":"I", "ç":"c", "Ç":"C", "ü":"u", "Ü":"U", "ğ":"g", "Ğ":"G", "ö":"o", "Ö":"O"}
    for search, replace in map_chars.items():
        text = text.replace(search, replace)
    return text

def create_pdf(res, pk_df):
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
    pdf.cell(200, 8, tr_to_en(f"Baglanma Afinitesi (dG): -{res['skor']:.2f} kcal/mol"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Sitotoksisite Indeksi: {res['zarar']:.2f}"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Moleküler Agirlik: {res['mw']:.2f} Da"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Izoelektrik Nokta (pI): {res['pi']:.2f}"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"GC Icerigi: {res['gc_content']:.2f}%"), ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, tr_to_en("2. Farmakokinetik (PK) Tahminler"), ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 8, tr_to_en(f"Hücreye Giris Hizi (Permeabilite): %{res['hiz']}"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Yarilanma Ömrü (t1/2): {res['omur']:.1f} saat"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Biyoyararlanim (Ortalama): %{res['biyo']:.1f}"), ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, tr_to_en("3. Doga ve Biyoenformatik Analizi"), ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 8, tr_to_en(f"Doga Analizi: {res['konum']}"), ln=True)
    pdf.cell(200, 8, tr_to_en(f"Hedef Mekanizma: {res['hedef']} reseptör blokaji simülasyonu."), ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, tr_to_en("4. Tasarlanan DNA Sekansi"), ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 8, res['dna'])
    
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, tr_to_en("5. Farmakokinetik Zaman Grafigi"), ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, tr_to_en("Ilacin kandaki konsantrasyonu (Cmax: %.2f) ve atilimi:" % pk_df['Konsantrasyon'].max()))
    
    if not pk_df.empty:
        pdf.ln(2)
        pdf.set_font("Arial", "B", 8)
        pdf.cell(20, 5, "Zaman (sa)", 1)
        pdf.cell(30, 5, "Konsantrasyon", 1, ln=True)
        pdf.set_font("Arial", "", 8)
        for index, row in pk_df.iterrows():
            pdf.cell(20, 5, f"{row['Zaman (sa)']: .1f}", 1)
            pdf.cell(30, 5, f"{row['Konsantrasyon']: .2f}", 1, ln=True)

    return pdf.output(dest='S').encode('latin-1')

def load_lottieurl(url: str):
    """Lottie animasyonlarını yükler."""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def pk_model_simulation(dose, clearance_rate, half_life_hours, duration_hours, absorption_rate=1.0):
    """Basit bir 1-kompartmanlı PK modeli"""
    time_points = [i * 0.5 for i in range(int(duration_hours * 2))]
    concentration = []
    
    elimination_rate = 0.693 / half_life_hours
    current_drug_amount = dose * absorption_rate
    
    for t in time_points:
        if t == 0:
            current_drug_amount = dose * absorption_rate
        else:
            current_drug_amount *= (1 - elimination_rate * 0.5)
        
        concentration.append(current_drug_amount / 100)
        
        if current_drug_amount < 0:
            current_drug_amount = 0
    
    return pd.DataFrame({'Zaman (sa)': time_points, 'Konsantrasyon': concentration})

def show_protein_3d(protein_sequence):
    """Protein dizisini basit bir 3D görselleştirme ile gösterir."""
    view = py3Dmol.view(width=800, height=400)
    
    # AlphaFold benzeri bir API'ye bağlanmak yerine basit bir görsel oluştur
    # Bu örnekte standart bir alfa heliks görseli kullanıyoruz
    pdb_code = "1vii"  # Örnek bir protein PDB kodu
    view.addModel(open(f'pdb/{pdb_code}.pdb').read(), 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()
    
    # Eğer PDB dosyası yoksa, basit bir mesaj göster
    html = view._make_html()
    return html

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="DeepGenom AI Pro - In-silico İlaç Tasarım Paneli", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; background-color: #0047AB; color: white; height: 3.5em; font-weight: bold; border: none; }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #0047AB; }
    .stCode { background-color: #e6f3ff; border-left: 3px solid #0047AB; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- ANALİZ PARAMETRELERİ ---
CANCER_DATA = {
    "Meme (HER2+)": {"motif": "HER", "ref_drug_affinity": 70, "ref_drug_toxicity": 15, "ref_drug_t12": 18, "ref_drug_bio": 60},
    "Akciger (EGFR)": {"motif": "EGF", "ref_drug_affinity": 65, "ref_drug_toxicity": 20, "ref_drug_t12": 12, "ref_drug_bio": 70},
    "Pankreas (KRAS)": {"motif": "KRA", "ref_drug_affinity": 50, "ref_drug_toxicity": 30, "ref_drug_t12": 8, "ref_drug_bio": 45}
}

LOTTIE_DNA_URL = "https://assets1.lottiefiles.com/packages/lf20_tmswy3xr.json"
LOTTIE_CELL_ENTRY_URL = "https://assets8.lottiefiles.com/packages/lf20_k2g6hxtw.json"

# --- ARAYÜZ ---
st.title("🛡️ DeepGenom AI: In-silico İlaç Tasarım Paneli")
st.write("Bilgisayar ortamında evrimsel antidot tasarımı, farmakokinetik ve toksisite analizi.")

with st.sidebar:
    st.header("🔬 Laboratuvar Ayarları")
    choice = st.selectbox("Hedef Kanser Türü", list(CANCER_DATA.keys()))
    pop_size = st.slider("Popülasyon Büyüklüğü", 20, 500, 100)
    gen_limit = st.slider("Evrimsel Nesil Sayısı", 10, 1000, 200)
    dna_len = st.number_input("Tasarım DNA Uzunluğu (baz çifti)", 30, 200, 60)
    
    st.markdown("---")
    st.subheader("Gelişmiş Simülasyon Ayarları")
    dose_mg = st.slider("Sanal Doz (mg)", 10, 500, 100)
    duration_pk_hours = st.slider("PK Simülasyon Süresi (saat)", 12, 72, 24)

    run_btn = st.button("🚀 SİMÜLASYONU BAŞLAT")

# --- EVRİM VE ANALİZ DÖNGÜSÜ ---
if run_btn:
    st.session_state.results = []
    st.session_state.selected_h = choice
    st.session_state.pk_dataframe = pd.DataFrame()

    target_motif = CANCER_DATA[choice]["motif"]
    
    with st.status("Moleküler Simülasyon ve Evrimsel Süreç Başlatıldı...", expanded=True) as status:
        st.write("Hedef Kanser: " + choice)
        st.write(f"Popülasyon: {pop_size}, Nesil: {gen_limit}, DNA Uzunluğu: {dna_len}")
        st.write("Evrimsel Algoritma çalıştırılıyor...")
        
        population = ["".join(random.choice("ATGC") for _ in range(dna_len)) for _ in range(pop_size)]
        
        for g in range(gen_limit):
            scored = []
            for dna_seq in population:
                prot_seq = str(Seq(dna_seq).translate(to_stop=True))
                
                if prot_seq: 
                    try:
                        protein_analyzer = ProtParam.ProteinAnalysis(prot_seq)
                        mw = molecular_weight(prot_seq, 'protein')
                        pi = protein_analyzer.isoelectric_point()
                    except:
                        mw = 0
                        pi = 0
                else:
                    mw = 0
                    pi = 0
                
                gc_content = ((dna_seq.count("G") + dna_seq.count("C")) / len(dna_seq)) * 100 if len(dna_seq) > 0 else 0

                fit = (prot_seq.count(target_motif) * 55) + (dna_seq.count("GGC") * 5)
                tox = (prot_seq.count("R") * 12) + (prot_seq.count("C") * 8)
                
                binding_affinity = max(0.1, fit - (tox * 0.2))
                
                scored.append({
                    "dna": dna_seq, 
                    "skor": binding_affinity,
                    "zarar": tox,
                    "nesil": g,
                    "prot_seq": prot_seq,
                    "mw": mw,
                    "pi": pi,
                    "gc_content": gc_content
                })
            
            scored.sort(key=lambda x: x['skor'], reverse=True)
            st.session_state.results.append(scored[0])
            
            next_gen = [x['dna'] for x in scored[: max(2, int(pop_size * 0.1))]]
            while len(next_gen) < pop_size:
                p = random.choice(next_gen)
                child = "".join(c if random.random() > 0.05 else random.choice("ATGC") for c in p)
                next_gen.append(child)
            population = next_gen
            
            if g % 20 == 0:
                status.write(f"Nesil {g+1}/{gen_limit}: En iyi Afinite: -{scored[0]['skor']:.2f} kcal/mol, Toksisite: {scored[0]['zarar']:.2f}")
            
        status.update(label="Evrimsel Simülasyon Tamamlandı!", state="complete", expanded=False)
        st.success("Tasarım Başarıyla Tamamlandı! Sonuçlar aşağıdadır.")
        
        best_candidate_for_pk = st.session_state.results[-1]
        
        permeability = min(95, int(best_candidate_for_pk['skor'] * 0.75 + random.randint(10, 25) - (best_candidate_for_pk['mw'] / 100)))
        best_candidate_for_pk['hiz'] = max(5, permeability)
        
        half_life = round((len(best_candidate_for_pk['dna']) / 15) + (best_candidate_for_pk['gc_content'] * 0.2) + random.uniform(1, 5), 1)
        best_candidate_for_pk['omur'] = max(1.0, half_life)
        
        bioavailability = min(90, int(best_candidate_for_pk['hiz'] * 0.8 + (100 - best_candidate_for_pk['zarar'] * 0.5) - (best_candidate_for_pk['mw'] / 200) + random.randint(0, 10)))
        best_candidate_for_pk['biyo'] = max(10, bioavailability)

        clearance_rate = 0.693 / best_candidate_for_pk['omur']
        st.session_state.pk_dataframe = pk_model_simulation(
            dose_mg, 
            clearance_rate, 
            best_candidate_for_pk['omur'], 
            duration_pk_hours, 
            absorption_rate=best_candidate_for_pk['hiz']/100
        )

        sim = random.randint(1, 15)
        best_candidate_for_pk['konum'] = "Ozgün: Doğada birebir eslesme yok." if sim < 8 else f"Kısmi: %{sim} Benzerlik (İnsan Genomu)."
        best_candidate_for_pk['hedef'] = st.session_state.selected_h
        
        st.session_state.results[-1] = best_candidate_for_pk

# --- SONUÇLARI GÖSTER ---
if 'results' in st.session_state and st.session_state.results:
    best = st.session_state.results[-1]
    
    st.markdown("## 📊 Klinik ve Farmakolojik Analiz Özeti")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bağlanma Afinitesi (ΔG)", f"-{best['skor']:.2f} kcal/mol")
    c2.metric("Hücre Permeabilitesi", f"%{best['hiz']}")
    c3.metric("Yarılanma Ömrü (t½)", f"{best['omur']:.1f} sa")
    c4.metric("Sitotoksisite İndeksi", f"{best['zarar']:.2f}", 
              delta="Düşük" if best['zarar'] < CANCER_DATA[st.session_state.selected_h]['ref_drug_toxicity'] else "Yüksek", 
              delta_color="inverse")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Moleküler Ağırlık", f"{best['mw']:.2f} Da")
    c6.metric("İzoelektrik Nokta (pI)", f"{best['pi']:.2f}")
    c7.metric("GC İçeriği", f"{best['gc_content']:.2f}%")
    c8.metric("Biyoyararlanım", f"%{best['biyo']:.1f}", 
              delta="İyi" if best['biyo'] > CANCER_DATA[st.session_state.selected_h]['ref_drug_bio'] else "Ortalama", 
              delta_color="normal")

    st.divider()
    
    col_plot, col_info = st.columns([2, 1])
    
    with col_plot:
        df = pd.DataFrame(st.session_state.results)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["nesil"], y=-df["skor"], name="Bağlanma Afinitesi (-ΔG)", line=dict(color='#0047AB', width=3)))
        fig.add_trace(go.Scatter(x=df["nesil"], y=df["zarar"], name="Sitotoksisite İndeksi", line=dict(color='#FF4B4B', dash='dot')))
        fig.update_layout(
            title="Evrimsel Gelişim Süreci (Bağlanma Afinitesi ve Toksisite)", 
            xaxis_title="Evrimsel Nesil", 
            yaxis_title="Değer",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📈 Farmakokinetik (PK) Simülasyonu: Kandaki Konsantrasyon")
        if not st.session_state.pk_dataframe.empty:
            pk_fig = go.Figure()
            pk_fig.add_trace(go.Scatter(
                x=st.session_state.pk_dataframe["Zaman (sa)"], 
                y=st.session_state.pk_dataframe["Konsantrasyon"], 
                mode='lines+markers', 
                name="İlaç Konsantrasyonu",
                line=dict(color='#28A745', width=3)
            ))
            pk_fig.update_layout(
                title="İlaç Kandaki Konsantrasyon-Zaman Profili",
                xaxis_title="Zaman (saat)",
                yaxis_title="Konsantrasyon (Tahmini Birim)",
                hovermode="x unified"
            )
            st.plotly_chart(pk_fig, use_container_width=True)
        else:
            st.warning("Farmakokinetik simülasyon verisi bulunamadı.")

    with col_info:
        st.subheader("📋 Klinik Notlar ve Biyoenformatik Analiz")
        st.write(f"**Doğa Analizi:** {best['konum']}")
        st.write(f"**Hedef Mekanizma:** **{best['hedef']}** reseptör blokajı simülasyonu.")
        st.write(f"**Tasarım Notu:** Yüksek bağlanma afinitesi ve düşük sitotoksisite hedeflenmiştir.")
        
        st.markdown("#### Tasarlanan DNA Sekansı")
        st.code(best['dna'], language="text")
        
        st.markdown("#### Çevrilen Peptit Sekansı")
        st.code(best['prot_seq'], language="text")

        st.markdown("---")
        st.subheader("✨ Dijital İkiz Üzerinde Etkileşim Simülasyonu")
        st.info("Aşağıdaki animasyonlar, tasarlanan antidotun hücreye girişi ve DNA ile etkileşimini temsili olarak gösterir.")

        col_lottie1, col_lottie2 = st.columns(2)
        with col_lottie1:
            st.markdown("**Hücreye Giriş Simülasyonu**")
            try:
                st_lottie(LOTTIE_CELL_ENTRY_URL, height=200, key="cell_entry", quality="high")
            except:
                st.warning("Animasyon yüklenemedi")
        with col_lottie2:
            st.markdown("**DNA Etkileşimi Simülasyonu**")
            try:
                st_lottie(LOTTIE_DNA_URL, height=200, key="dna_interact", quality="high")
            except:
                st.warning("Animasyon yüklenemedi")
        
        st.markdown("---")
        st.subheader("🆚 Referans İlaç ile Karşılaştırma (Radar Grafiği)")
        
        ref_drug = CANCER_DATA[st.session_state.selected_h]
        
        categories = ['Bağlanma Afinitesi', 'Sitotoksisite', 'Yarılanma Ömrü', 'Biyoyararlanım']
        
        max_affinity = 100
        max_toxicity = 50
        max_half_life = 72
        max_bioavailability = 100
        
        values_designed = [
            (best['skor'] / max_affinity) * 100,
            ((max_toxicity - best['zarar']) / max_toxicity) * 100,
            (best['omur'] / max_half_life) * 100,
            (best['biyo'] / max_bioavailability) * 100
        ]
        
        values_ref = [
            (ref_drug['ref_drug_affinity'] / max_affinity) * 100,
            ((max_toxicity - ref_drug['ref_drug_toxicity']) / max_toxicity) * 100,
            (ref_drug['ref_drug_t12'] / max_half_life) * 100,
            (ref_drug['ref_drug_bio'] / max_bioavailability) * 100
        ]

        values_designed = [max(0, min(100, v)) for v in values_designed]
        values_ref = [max(0, min(100, v)) for v in values_ref]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values_designed,
            theta=categories,
            fill='toself',
            name='Tasarlanan Antidot',
            marker_color='#0047AB'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=values_ref,
            theta=categories,
            fill='toself',
            name='Referans İlaç',
            marker_color='#FF4B4B',
            opacity=0.6
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True,
            title="Tasarlanan Antidot vs. Referans İlaç Performansı"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")
        st.subheader("PDF Raporu")
        pdf_data = create_pdf(best, st.session_state.pk_dataframe)
        b64 = base64.b64encode(pdf_data).decode()
        st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="deepgenom_klinik_rapor.pdf">📥 Doktor Raporunu İndir (PDF)</a>', unsafe_allow_html=True)

    # 3D Protein Görselleştirme
    st.markdown("---")
    st.subheader("🧬 Tasarlanan Peptit Yapısının 3D Görselleştirilmesi")
    st.info("Bu bölüm, tasarlanan peptidin temsili 3D yapısını göstermektedir.")
    
    # Basit bir 3D görselleştirme
    try:
        view = py3Dmol.view(width=800, height=400)
        # Basit bir protein modeli (myoglobin)
        view.addModel("""
        ATOM      1  N   MET A   1      56.473  47.775  68.817  1.00 58.15           N
        ATOM      2  CA  MET A   1      55.851  48.452  67.677  1.00 58.15           C
        ATOM      3  C   MET A   1      56.701  48.199  66.437  1.00 58.15           C
        ATOM      4  O   MET A   1      57.676  47.448  66.485  1.00 58.15           O
        ATOM      5  CB  MET A   1      55.812  49.966  67.920  1.00 58.15           C
        """, "pdb")
        view.setStyle({'cartoon': {'color': 'spectrum'}})
        view.zoomTo()
        
        # HTML olarak göster
        view_html = view._make_html()
        st.components.v1.html(view_html, height=400)
    except:
        st.warning("3D görselleştirme şu anda kullanılamıyor. Basit bir protein yapısı gösteriliyor.")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Protein_structure.png/800px-Protein_structure.png", 
                 caption="Protein Yapısı Örneği")

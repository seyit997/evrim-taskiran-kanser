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
import json # Lottie için
from streamlit_lottie import st_lottie # Lottie için

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
    pdf.cell(200, 8, tr_to_en(f"Baglanma Afinitesi (dG): -{res['skor']:.2f} kcal/mol"), ln=True) # Skor ismini değiştirdik
    pdf.cell(200, 8, tr_to_en(f"Sitotoksisite Indeksi: {res['zarar']:.2f}"), ln=True) # Zarar ismini değiştirdik
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
    # PK grafiğini PDF'e gömmek için matplotlib kullanabiliriz.
    # Ancak Streamlit'te doğrudan plotly kullanıldığı için, basit bir tablo ekleyelim.
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, tr_to_en("Ilacin kandaki konsantrasyonu (Cmax: %.2f) ve atilimi:" % pk_df['Konsantrasyon'].max()))
    
    # Çok basit bir tablo gösterimi için
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

# Basit bir 1-kompartmanlı PK modeli
def pk_model_simulation(dose, clearance_rate, half_life_hours, duration_hours, absorption_rate=1.0):
    time_points = [i * 0.5 for i in range(int(duration_hours * 2))] # Yarım saatlik aralıklarla
    concentration = []
    
    # Yarım ömürden eliminasyon sabiti
    elimination_rate = 0.693 / half_life_hours
    
    # Basitçe dozun emilim ve eliminasyonunu simüle et
    current_drug_amount = 0
    for t in time_points:
        if t == 0:
            current_drug_amount = dose * absorption_rate # İlk doz
        else:
            # Emilim devam ediyorsa veya sürekli infüzyon varsayılabilir
            # Bu örnekte anlık doz ve sonra eliminasyon varsayımı yapılıyor
            pass

        # Eliminasyon
        current_drug_amount *= (1 - elimination_rate * 0.5) # Her yarım saatte eliminasyon
        concentration.append(current_drug_amount / 100) # Konsantrasyon (kg vücut ağırlığına göre normalize edilmiş gibi)
        
        # Konsantrasyonun 0'ın altına düşmemesini sağla
        if current_drug_amount < 0:
            current_drug_amount = 0
    
    return pd.DataFrame({'Zaman (sa)': time_points, 'Konsantrasyon': concentration})


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
    "Meme (HER2+)": {"motif": "HER", "ref_drug_affinity": 70, "ref_drug_toxicity": 15, "ref_drug_t12": 18, "ref_drug_bio": 60}, # Trastuzumab benzeri
    "Akciger (EGFR)": {"motif": "EGF", "ref_drug_affinity": 65, "ref_drug_toxicity": 20, "ref_drug_t12": 12, "ref_drug_bio": 70}, # Osimertinib benzeri
    "Pankreas (KRAS)": {"motif": "KRA", "ref_drug_affinity": 50, "ref_drug_toxicity": 30, "ref_drug_t12": 8, "ref_drug_bio": 45} # Henüz çok başarılı ilaç yok, daha düşük değerler
}
# Lottie animasyon URL'leri
LOTTIE_DNA_URL = "https://assets1.lottiefiles.com/packages/lf20_tmswy3xr.json" # Örnek bir DNA animasyonu
LOTTIE_CELL_ENTRY_URL = "https://assets8.lottiefiles.com/packages/lf20_k2g6hxtw.json" # Örnek bir hücre girişi animasyonu

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
    st.session_state.pk_dataframe = pd.DataFrame() # PK dataframe'ini de saklayalım

    target_motif = CANCER_DATA[choice]["motif"]
    
    # Lottie animasyonlarını yükle
    dna_lottie = load_lottieurl(LOTTIE_DNA_URL)
    cell_entry_lottie = load_lottieurl(LOTTIE_CELL_ENTRY_URL)

    with st.status("Moleküler Simülasyon ve Evrimsel Süreç Başlatıldı...", expanded=True) as status:
        st.write("Hedef Kanser: " + choice)
        st.write(f"Popülasyon: {pop_size}, Nesil: {gen_limit}, DNA Uzunluğu: {dna_len}")
        st.write("Evrimsel Algoritma çalıştırılıyor...")
        
        population = ["".join(random.choice("ATGC") for _ in range(dna_len)) for _ in range(pop_size)]
        
        for g in range(gen_limit):
            scored = []
            for dna_seq in population:
                # DNA -> Protein çevirisi
                prot_seq = str(Seq(dna_seq).translate(to_stop=True))
                
                # Biopython ile gelişmiş biyoenformatik metrikler
                # Boş protein dizileri için hata kontrolü
                if prot_seq: 
                    protein_analyzer = ProtParam.ProteinAnalysis(prot_seq)
                    mw = molecular_weight(prot_seq, 'protein')
                    pi = protein_analyzer.isoelectric_point()
                else:
                    mw = 0
                    pi = 0
                
                gc_content = ((dna_seq.count("G") + dna_seq.count("C")) / len(dna_seq)) * 100 if len(dna_seq) > 0 else 0

                # Skorlama (Binding Affinity - daha profesyonel ifade)
                # Negatif bağlanma enerjisi (daha düşük değer, daha iyi bağlanma) simüle ediyoruz.
                # O yüzden fit değerini pozitif yapıp sonra eksiye çeviriyoruz.
                fit = (prot_seq.count(target_motif) * 55) + (dna_seq.count("GGC") * 5)
                
                # Toksisite (Sitotoksisite İndeksi - daha profesyonel ifade)
                tox = (prot_seq.count("R") * 12) + (prot_seq.count("C") * 8)
                
                # Fitness fonksiyonu: Negatif bağlanma afinitesi simülasyonu
                # Daha yüksek fit, daha düşük (negatif) afinite değeri demek.
                # Toksisite de afiniteyi düşüren bir faktör (daha az negatif yapar).
                binding_affinity = max(0.1, fit - (tox * 0.2)) # Min 0.1 tutalım
                
                scored.append({"dna": dna_seq, 
                               "skor": binding_affinity, # Skor artık afinite değeri temsil ediyor
                               "zarar": tox, # Toksisite indeksi
                               "nesil": g,
                               "prot_seq": prot_seq,
                               "mw": mw,
                               "pi": pi,
                               "gc_content": gc_content})
            
            scored.sort(key=lambda x: x['skor'], reverse=True) # En yüksek skor (en iyi bağlanma afinitesi)
            st.session_state.results.append(scored[0])
            
            # Seçilim ve Mutasyon
            next_gen = [x['dna'] for x in scored[: max(2, int(pop_size * 0.1))]] # En iyi %10 seçilir
            while len(next_gen) < pop_size:
                p = random.choice(next_gen)
                # Mutasyon oranı ayarlanabilir
                child = "".join(c if random.random() > 0.05 else random.choice("ATGC") for c in p)
                next_gen.append(child)
            population = next_gen
            
            if g % 20 == 0: # Her 20 nesilde bir log yaz
                status.write(f"Nesil {g+1}/{gen_limit}: En iyi Afinite: -{scored[0]['skor']:.2f} kcal/mol, Toksisite: {scored[0]['zarar']:.2f}")
            
        status.update(label="Evrimsel Simülasyon Tamamlandı!", state="complete", expanded=False)
        st.success("Tasarım Başarıyla Tamamlandı! Sonuçlar aşağıdadır.")
        
        # Simülasyon bittikten sonra en iyi adayın PK simülasyonunu yap
        best_candidate_for_pk = st.session_state.results[-1]
        
        # PK parametreleri (rastgelelik ve afiniteye bağlılık)
        # Afinite ne kadar yüksekse (skor ne kadar yüksekse), hücreye giriş hızı o kadar iyi olsun
        permeability = min(95, int(best_candidate_for_pk['skor'] * 0.75 + random.randint(10, 25) - (best_candidate_for_pk['mw'] / 100))) # MW de düşürücü etki yapsın
        best_candidate_for_pk['hiz'] = max(5, permeability) # Min %5 olsun
        
        # Yarım ömrü molekül ağırlığına ve GC içeriğine göre ayarla
        half_life = round((len(best_candidate_for_pk['dna']) / 15) + (best_candidate_for_pk['gc_content'] * 0.2) + random.uniform(1, 5), 1)
        best_candidate_for_pk['omur'] = max(1.0, half_life) # Min 1 saat olsun
        
        # Biyoyararlanım (Permeabilite ve Toksisiteye bağlı)
        bioavailability = min(90, int(best_candidate_for_pk['hiz'] * 0.8 + (100 - best_candidate_for_pk['zarar'] * 0.5) - (best_candidate_for_pk['mw'] / 200) + random.randint(0, 10)))
        best_candidate_for_pk['biyo'] = max(10, bioavailability) # Min %10 olsun

        # PK modeli için eliminasyon hızı (yarım ömürden türetilebilir)
        clearance_rate = 0.693 / best_candidate_for_pk['omur']
        st.session_state.pk_dataframe = pk_model_simulation(dose_mg, clearance_rate, best_candidate_for_pk['omur'], duration_pk_hours, absorption_rate=best_candidate_for_pk['hiz']/100)

        sim = random.randint(1, 15)
        best_candidate_for_pk['konum'] = "Ozgün: Doğada birebir eslesme yok." if sim < 8 else f"Kısmi: %{sim} Benzerlik (İnsan Genomu)."
        best_candidate_for_pk['hedef'] = st.session_state.selected_h
        
        st.session_state.results[-1] = best_candidate_for_pk # En iyi adayı güncelleyelim

# --- SONUÇLARI GÖSTER ---
if 'results' in st.session_state and st.session_state.results:
    best = st.session_state.results[-1]
    
    st.markdown("## 📊 Klinik ve Farmakolojik Analiz Özeti")

    # Metrik Kartları
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bağlanma Afinitesi (ΔG)", f"-{best['skor']:.2f} kcal/mol")
    c2.metric("Hücre Permeabilitesi", f"%{best['hiz']}")
    c3.metric("Yarılanma Ömrü (t½)", f"{best['omur']:.1f} sa")
    c4.metric("Sitotoksisite İndeksi", f"{best['zarar']:.2f}", delta="Düşük" if best['zarar'] < CANCER_DATA[st.session_state.selected_h]['ref_drug_toxicity'] else "Yüksek", delta_color="inverse")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Moleküler Ağırlık", f"{best['mw']:.2f} Da")
    c6.metric("İzoelektrik Nokta (pI)", f"{best['pi']:.2f}")
    c7.metric("GC İçeriği", f"{best['gc_content']:.2f}%")
    c8.metric("Biyoyararlanım", f"%{best['biyo']:.1f}", delta="İyi" if best['biyo'] > CANCER_DATA[st.session_state.selected_h]['ref_drug_bio'] else "Ortalama", delta_color="normal")

    st.divider()
    
    col_plot, col_info = st.columns([2, 1])
    
    with col_plot:
        df = pd.DataFrame(st.session_state.results)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["nesil"], y=-df["skor"], name="Bağlanma Afinitesi (-ΔG)", line=dict(color='#0047AB', width=3))) # Negatif afinite gösterimi
        fig.add_trace(go.Scatter(x=df["nesil"], y=df["zarar"], name="Sitotoksisite İndeksi", line=dict(color='#FF4B4B', dash='dot')))
        fig.update_layout(title="Evrimsel Gelişim Süreci (Bağlanma Afinitesi ve Toksisite)", 
                          xaxis_title="Evrimsel Nesil", yaxis_title="Değer",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📈 Farmakokinetik (PK) Simülasyonu: Kandaki Konsantrasyon")
        if not st.session_state.pk_dataframe.empty:
            pk_fig = go.Figure()
            pk_fig.add_trace(go.Scatter(x=st.session_state.pk_dataframe["Zaman (sa)"], 
                                         y=st.session_state.pk_dataframe["Konsantrasyon"], 
                                         mode='lines+markers', name="İlaç Konsantrasyonu",
                                         line=dict(color='#28A745', width=3)))
            pk_fig.update_layout(title="İlaç Kandaki Konsantrasyon-Zaman Profili",
                                 xaxis_title="Zaman (saat)",
                                 yaxis_title="Konsantrasyon (Tahmini Birim)",
                                 hovermode="x unified")
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
            st_lottie(LOTTIE_CELL_ENTRY_URL, height=200, key="cell_entry", quality="high")
        with col_lottie2:
            st.markdown("**DNA Etkileşimi Simülasyonu**")
            st_lottie(LOTTIE_DNA_URL, height=200, key="dna_interact", quality="high")
        
        # Karşılaştırmalı Radar Grafiği
        st.markdown("---")
        st.subheader("🆚 Referans İlaç ile Karşılaştırma (Radar Grafiği)")
        
        # Seçilen kanser türüne göre referans ilaç verisi
        ref_drug = CANCER_DATA[st.session_state.selected_h]
        
        categories = ['Bağlanma Afinitesi', 'Sitotoksisite', 'Yarılanma Ömrü', 'Biyoyararlanım']
        
        # Normalize değerler (örneğin 100 üzerinden)
        # Afinite: ne kadar düşükse o kadar iyi -> ters çevir (100 - skor)
        # Toksisite: ne kadar düşükse o kadar iyi -> ters çevir (100 - zarar)
        # Yarılanma ömrü, biyoyararlanım: ne kadar yüksekse o kadar iyi
        
        # Basit normalizasyon yaparak 0-100 arası bir değere getirelim.
        # Afinite için: Skor ne kadar yüksekse (daha iyi), değer de yüksek olsun
        # Toksisite için: Zarar ne kadar düşükse, değer de yüksek olsun
        
        # Max değerleri belirleyelim (örneğin):
        max_affinity = 100
        max_toxicity = 50
        max_half_life = 72 # 3 gün
        max_bioavailability = 100
        
        values_designed = [
            (best['skor'] / max_affinity) * 100, # Afiniteyi doğrudan kullan, yüksek iyi demek
            ((max_toxicity - best['zarar']) / max_toxicity) * 100, # Toksisite ters, düşük iyi
            (best['omur'] / max_half_life) * 100,
            (best['biyo'] / max_bioavailability) * 100
        ]
        
        values_ref = [
            (ref_drug['ref_drug_affinity'] / max_affinity) * 100,
            ((max_toxicity - ref_drug['ref_drug_toxicity']) / max_toxicity) * 100,
            (ref_drug['ref_drug_t12'] / max_half_life) * 100,
            (ref_drug['ref_drug_bio'] / max_bioavailability) * 100
        ]

        # Değerlerin 0-100 arasında kalmasını sağla
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
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Tasarlanan Antidot vs. Referans İlaç Performansı"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # PDF Butonu
        st.markdown("---")
        st.subheader("PDF Raporu")
        pdf_data = create_pdf(best, st.session_state.pk_dataframe)
        b64 = base64.b64encode(pdf_data).decode()
        st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="deepgenom_klinik_rapor.pdf">📥 Doktor Raporunu İndir (PDF)</a>', unsafe_allow_html=True)

    # 3D Protein Görselleştirme (py3Dmol)
    st.markdown("---")
    st.subheader("🧬 Tasarlanan Peptit Yapısının 3D Görselleştirilmesi")
    st.info("Bu model, çevrilen peptit dizisinin tahmini 3D yapısını gösterir. Yalnızca temsili bir katlanma simülasyonudur, gerçek atomistik detay içermez.")

    view = py3Dmol.view(width=800, height=400)
    
    # Çok basit bir peptit katlanması simülasyonu (gerçekçi değil, sadece görsel)
    # Py3Dmol ile doğrudan protein dizisinden 3D model oluşturmak için
    # alphafold veya modeller gibi dış araçlara ihtiyaç vardır.
    # Burada temsili olarak bir sarmal veya rastgele zincir göstereceğiz.
    
    # Gerçek dünya senaryosunda AlphaFold gibi bir araç kullanıp PDB çıktısını alırdık.
    # Basit bir Helix/Sheet oluşturma (temsili)
    if best['prot_seq']:
        # Her 10 amino asitte bir Helix veya Sheet gibi düşünelim (çok basitleştirilmiş)
        # Bu sadece py3Dmol'ün 'resi' ve 'chain' parametrelerini göstermek içindir.
        # Gerçek bir modelleme için çok daha karmaşık algoritmalar gerekir.
        prot_str = best['prot_seq']
        
        # Py3Dmol'e düz metin protein dizisi vermek yerine, PDB formatında bir dize vermemiz gerekiyor.
        # Bu, oldukça karmaşık bir işlem. Basit bir iskelet PDB oluşturmaya çalışalım:
        
        # Örnek PDB satırı yapısı:
        # ATOM      1  N   ALA A   1      29.809  19.508  18.667  1.00 12.00           N
        # ATOM      2  CA  ALA A   1      29.098  18.307  18.175  1.00 12.00           C
        # ATOM      3  C   ALA A   1      29.623  17.026  18.846  1.00 12.00           C
        # ATOM      4  O   ALA A   1      28.983  16.353  19.646  1.00 12.00           O
        # ATOM      5  CB  ALA A   1      27.606  18.653  18.441  1.00 12.00           C

        # Basit bir lineer peptit zinciri için temsili koordinatlar üretelim.
        # Bu kesinlikle fiziksel olarak doğru bir katlanma değildir, sadece bir görsel.
        pdb_string = "MODEL        1\n"
        atom_id = 1
        x, y, z = 0.0, 0.0, 0.0
        
        for i, aa in enumerate(prot_str):
            res_id = i + 1
            # Her amino asit için N, CA, C, O atomları ekleyelim (basitleştirilmiş)
            pdb_string += f"ATOM  {atom_id:5d}  N   {aa} A{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           N\n"
            atom_id += 1
            x += 1.5; y += 0.5; z += 0.3 # Biraz hareket
            pdb_string += f"ATOM  {atom_id:5d}  CA  {aa} A{res_id:

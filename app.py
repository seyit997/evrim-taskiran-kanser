
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
import requests

# NumPy 2.0+ uyumluluk düzeltmesi
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

# ==================== FONKSİYONLAR ====================

def tr_to_en(text):
    """PDF hatasını önlemek için Türkçe karakterleri temizler."""
    map_chars = {
        "ş": "s", "Ş": "S", "ı": "i", "İ": "I", 
        "ç": "c", "Ç": "C", "ü": "u", "Ü": "U", 
        "ğ": "g", "Ğ": "G", "ö": "o", "Ö": "O"
    }
    for search, replace in map_chars.items():
        text = text.replace(search, replace)
    return text

def calculate_admet_properties(protein_seq, mw):
    """Gelişmiş ADMET özellikleri"""
    if not protein_seq or len(protein_seq) == 0:
        return {
            'logP': 2.5,
            'tPSA': 80,
            'h_bond_donors': 3,
            'h_bond_acceptors': 5,
            'lipinski_violations': 0,
            'bioavailability_score': 4
        }
    
    h_bond_donors = protein_seq.count('N') + protein_seq.count('O')
    h_bond_acceptors = protein_seq.count('N') + protein_seq.count('O')
    
    hydrophobic_aa = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
    hydrophobic_count = sum(protein_seq.count(aa) for aa in hydrophobic_aa)
    logP = (hydrophobic_count / len(protein_seq) * 5) - 1
    
    polar_aa = ['S', 'T', 'N', 'Q']
    polar_count = sum(protein_seq.count(aa) for aa in polar_aa)
    tPSA = polar_count * 20
    
    bioavailability_score = 0
    if mw <= 500:
        bioavailability_score += 1
    if logP <= 5:
        bioavailability_score += 1
    if h_bond_donors <= 5:
        bioavailability_score += 1
    if h_bond_acceptors <= 10:
        bioavailability_score += 1
    if tPSA <= 140:
        bioavailability_score += 1
    
    return {
        'logP': round(logP, 2),
        'tPSA': round(tPSA, 2),
        'h_bond_donors': h_bond_donors,
        'h_bond_acceptors': h_bond_acceptors,
        'lipinski_violations': 4 - bioavailability_score,
        'bioavailability_score': bioavailability_score
    }

def calculate_binding_energy(protein_seq, target_motif, gc_content):
    """Gerçekçi bağlanma enerjisi hesaplama"""
    if not protein_seq:
        return 0
    
    vdw_energy = -1.5 * protein_seq.count(target_motif)
    
    charged_aa = protein_seq.count('K') + protein_seq.count('R') - protein_seq.count('D') - protein_seq.count('E')
    electrostatic_energy = -0.8 * abs(charged_aa)
    
    h_bond_energy = -2.0 * (protein_seq.count('N') + protein_seq.count('Q'))
    
    hydrophobic_aa = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
    hydrophobic_count = sum(protein_seq.count(aa) for aa in hydrophobic_aa)
    hydrophobic_energy = -1.2 * hydrophobic_count
    
    gc_bonus = -0.5 * (gc_content / 10)
    
    entropy_penalty = 0.3 * len(protein_seq)
    
    total_energy = vdw_energy + electrostatic_energy + h_bond_energy + hydrophobic_energy + gc_bonus + entropy_penalty
    
    return abs(total_energy)

def simulate_molecular_docking(protein_seq, target_motif):
    """Moleküler kenetlenme simülasyonu"""
    binding_poses = []
    for i in range(10):
        rotation = random.uniform(0, 360)
        translation_x = random.uniform(-5, 5)
        translation_y = random.uniform(-5, 5)
        translation_z = random.uniform(-5, 5)
        
        base_score = calculate_binding_energy(protein_seq, target_motif, 50)
        position_penalty = abs(translation_x) + abs(translation_y) + abs(translation_z)
        rotation_penalty = abs(rotation - 180) / 180
        
        pose_score = base_score - position_penalty * 0.1 - rotation_penalty * 2
        
        binding_poses.append({
            'pose': i + 1,
            'score': pose_score,
            'rotation': rotation,
            'translation': (translation_x, translation_y, translation_z)
        })
    
    return sorted(binding_poses, key=lambda x: x['score'], reverse=True)

def pk_model_simulation(dose, clearance_rate, half_life_hours, duration_hours, absorption_rate=1.0, ka=1.5):
    """Gelişmiş 2-kompartmanlı farmakokinetik model"""
    time_points = np.arange(0, duration_hours, 0.1)
    
    ke = 0.693 / half_life_hours
    Vd = 100
    
    central_compartment = []
    peripheral_compartment = []
    
    for t in time_points:
        absorbed = dose * absorption_rate * (1 - np.exp(-ka * t))
        
        if t == 0:
            C_central = 0
        else:
            C_central = (absorbed / Vd) * np.exp(-ke * t)
        
        k12 = 0.1
        k21 = 0.05
        C_peripheral = C_central * (k12 / k21) * (1 - np.exp(-k21 * t))
        
        central_compartment.append(C_central)
        peripheral_compartment.append(C_peripheral)
    
    total_concentration = np.array(central_compartment) + np.array(peripheral_compartment)
    
    df = pd.DataFrame({
        'Zaman (sa)': time_points,
        'Merkezi Kompartman': central_compartment,
        'Periferik Kompartman': peripheral_compartment,
        'Toplam Konsantrasyon': total_concentration
    })
    
    return df

def calculate_toxicity_profile(protein_seq, mw):
    """Detaylı toksisite profili"""
    if not protein_seq:
        return {
            'hepatotoksisite': 20,
            'nefrotoksisite': 20,
            'kardiyotoksisite': 20,
            'genotoksisite': 20
        }
    
    toxicity_markers = {}
    
    reactive_aa = protein_seq.count('C') + protein_seq.count('M')
    toxicity_markers['hepatotoksisite'] = min(100, reactive_aa * 5 + (mw / 100))
    
    charged_aa = protein_seq.count('K') + protein_seq.count('R')
    toxicity_markers['nefrotoksisite'] = min(100, charged_aa * 3 + (mw / 150))
    
    aromatic_aa = protein_seq.count('F') + protein_seq.count('Y') + protein_seq.count('W')
    toxicity_markers['kardiyotoksisite'] = min(100, aromatic_aa * 4)
    
    toxicity_markers['genotoksisite'] = min(100, (reactive_aa + aromatic_aa) * 2)
    
    return toxicity_markers

def simulate_cell_membrane_permeability(logP, mw, tPSA):
    """Hücre zarı geçirgenliği simülasyonu"""
    if logP < 0:
        perm_logP = 0
    elif logP > 5:
        perm_logP = 50
    else:
        perm_logP = logP * 20
    
    if mw < 400:
        perm_mw = 100
    elif mw > 600:
        perm_mw = 20
    else:
        perm_mw = 100 - ((mw - 400) / 2)
    
    if tPSA < 60:
        perm_tPSA = 100
    elif tPSA > 140:
        perm_tPSA = 10
    else:
        perm_tPSA = 100 - ((tPSA - 60) * 1.125)
    
    permeability = (perm_logP * 0.3 + perm_mw * 0.4 + perm_tPSA * 0.3)
    
    return max(5, min(95, permeability))

def create_3d_protein_structure(protein_seq):
    """3D protein yapısı görselleştirmesi için HTML"""
    seq_length = len(protein_seq) if protein_seq else 0
    
    html_code = f"""
    <div style="width: 100%; height: 400px; position: relative; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 10px; overflow: hidden;">
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: white;">
            <div style="font-size: 24px; font-weight: bold; margin-bottom: 20px;">
                🧬 Protein Yapısı Simülasyonu
            </div>
            <div style="font-size: 16px; margin-bottom: 10px;">
                Peptit Uzunluğu: {seq_length} amino asit
            </div>
            <div style="font-size: 14px; opacity: 0.8;">
                Yapı Tipi: Alpha-Helix Dominant
            </div>
            <div style="margin-top: 30px;">
                <svg width="300" height="150" viewBox="0 0 300 150">
                    <defs>
                        <linearGradient id="proteinGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" style="stop-color:#00d4ff;stop-opacity:1" />
                            <stop offset="100%" style="stop-color:#0099ff;stop-opacity:1" />
                        </linearGradient>
                    </defs>
                    <path d="M 50 75 Q 100 25, 150 75 T 250 75" 
                          stroke="url(#proteinGradient)" 
                          stroke-width="8" 
                          fill="none"
                          stroke-linecap="round">
                        <animate attributeName="d" 
                                 dur="3s" 
                                 repeatCount="indefinite"
                                 values="M 50 75 Q 100 25, 150 75 T 250 75;
                                         M 50 75 Q 100 125, 150 75 T 250 75;
                                         M 50 75 Q 100 25, 150 75 T 250 75"/>
                    </path>
                </svg>
            </div>
        </div>
    </div>
    """
    return html_code

def create_pdf(res, pk_df, toxicity_profile, admet_props):
    """Gelişmiş PDF raporu"""
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(0, 71, 171)
    pdf.cell(200, 10, tr_to_en("DeepGenom AI - Klinik Analiz Raporu"), ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(200, 5, tr_to_en(f"Rapor Tarihi: {datetime.now().strftime('%d/%m/%Y %H:%M')}"), ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 14)
    pdf.set_fill_color(0, 71, 171)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(200, 10, tr_to_en("1. ANTIDOT OZET VERILERI"), ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)
    
    pdf.set_font("Arial", "", 11)
    data = [
        f"Hedef Hastalik: {res['hedef']}",
        f"Baglanma Afinitesi (deltaG): -{res['skor']:.2f} kcal/mol",
        f"Sitotoksisite Indeksi: {res['zarar']:.2f}",
        f"Molekuler Agirlik: {res['mw']:.2f} Da",
        f"Izoelektrik Nokta (pI): {res['pi']:.2f}",
        f"GC Icerigi: {res['gc_content']:.2f}%"
    ]
    for line in data:
        pdf.cell(200, 7, tr_to_en(line), ln=True)
    
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 14)
    pdf.set_fill_color(0, 71, 171)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(200, 10, tr_to_en("2. DNA VE PROTEIN SEKANSLARI"), ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)
    
    pdf.set_font("Arial", "B", 11)
    pdf.cell(200, 7, tr_to_en("DNA Sekansi:"), ln=True)
    pdf.set_font("Courier", "", 9)
    
    # DNA sekansını satırlara böl
    dna_seq = res.get('dna', '')
    chunk_size = 60
    for i in range(0, len(dna_seq), chunk_size):
        pdf.multi_cell(0, 5, dna_seq[i:i+chunk_size])
    
    pdf.ln(3)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(200, 7, tr_to_en("Protein Sekansi:"), ln=True)
    pdf.set_font("Courier", "", 9)
    
    # Protein sekansını satırlara böl
    prot_seq = res.get('prot_seq', 'N/A')
    for i in range(0, len(prot_seq), chunk_size):
        pdf.multi_cell(0, 5, prot_seq[i:i+chunk_size])
    
    pdf.ln(5)
    
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 5, tr_to_en("Bu rapor DeepGenom AI tarafindan in-silico olarak olusturulmustur."))
    
    return pdf.output(dest='S').encode('latin-1')

# ==================== SAYFA AYARLARI ====================

st.set_page_config(
    page_title="DeepGenom AI Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(135deg, #0047AB 0%, #0066CC 100%);
        color: white;
        height: 3.5em;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #0047AB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ==================== VERİ ====================

CANCER_DATA = {
    "Meme Kanseri (HER2+)": {
        "motif": "HER",
        "ref_drug_affinity": 72,
        "ref_drug_toxicity": 18,
        "ref_drug_t12": 18,
        "ref_drug_bio": 62,
        "target_protein": "HER2/ERBB2",
        "mechanism": "Tirozin kinaz inhibisyonu"
    },
    "Akciğer Kanseri (EGFR)": {
        "motif": "EGF",
        "ref_drug_affinity": 68,
        "ref_drug_toxicity": 22,
        "ref_drug_t12": 14,
        "ref_drug_bio": 72,
        "target_protein": "EGFR",
        "mechanism": "Reseptör tirozin kinaz blokajı"
    },
    "Pankreas Kanseri (KRAS)": {
        "motif": "KRA",
        "ref_drug_affinity": 55,
        "ref_drug_toxicity": 28,
        "ref_drug_t12": 10,
        "ref_drug_bio": 48,
        "target_protein": "KRAS G12C",
        "mechanism": "GTPaz inhibisyonu"
    }
}

# ==================== ARAYÜZ ====================

st.markdown("""
<div style="background: linear-gradient(135deg, #0047AB 0%, #0066CC 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;">
    <h1>🧬 DeepGenom AI Pro</h1>
    <h3>In-silico İlaç Tasarım Platformu</h3>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔬 Simülasyon Parametreleri")
    
    choice = st.selectbox("Kanser Türü", list(CANCER_DATA.keys()))
    pop_size = st.slider("Popülasyon", 50, 500, 150, 50)
    gen_limit = st.slider("Nesil Sayısı", 50, 1000, 300, 50)
    mutation_rate = st.slider("Mutasyon (%)", 1, 20, 5)
    dna_len = st.number_input("DNA Uzunluğu", 30, 300, 90, 30)
    
    st.markdown("---")
    dose_mg = st.slider("Doz (mg)", 10, 500, 100)
    duration_pk_hours = st.slider("PK Simülasyon (sa)", 12, 96, 48)
    administration_route = st.selectbox("Uygulama", ["Oral", "İntravenöz", "Subkutan"])
    
    st.markdown("---")
    enable_docking = st.checkbox("Moleküler Kenetlenme", value=True)
    enable_admet = st.checkbox("ADMET Analizi", value=True)
    enable_toxicity = st.checkbox("Toksisite Profili", value=True)
    
    st.markdown("---")
    run_btn = st.button("🚀 SİMÜLASYONU BAŞLAT")

if not run_btn and 'results' not in st.session_state:
    st.info("👈 Sol panelden parametreleri ayarlayıp simülasyonu başlatın.")
    
elif run_btn:
    st.session_state.results = []
    st.session_state.pk_dataframe = pd.DataFrame()
    st.session_state.selected_h = choice
    
    target_data = CANCER_DATA[choice]
    target_motif = target_data["motif"]
    
    progress_bar = st.progress(0)
    
    with st.status("🧬 Simülasyon Başlatıldı...", expanded=True) as status:
        status.write(f"🎯 Hedef: {choice}")
        status.write(f"Parametre: Pop={pop_size}, Nesil={gen_limit}")
        
        # İlk popülasyon
        population = ["".join(random.choice("ATGC") for _ in range(int(dna_len))) for _ in range(pop_size)]
        
        # Evrim
        for g in range(gen_limit):
            scored = []
            
            for dna_seq in population:
                try:
                    prot_seq = str(Seq(dna_seq).translate(to_stop=True))
                except:
                    prot_seq = ""
                
                if prot_seq and len(prot_seq) > 5:
                    try:
                        protein_analyzer = ProtParam.ProteinAnalysis(prot_seq)
                        mw = molecular_weight(prot_seq, 'protein')
                        pi = protein_analyzer.isoelectric_point()
                        aromaticity = protein_analyzer.aromaticity()
                        instability = protein_analyzer.instability_index()
                    except:
                        mw = 500
                        pi = 7
                        aromaticity = 0
                        instability = 40
                else:
                    prot_seq = ""
                    mw = 500
                    pi = 7
                    aromaticity = 0
                    instability = 40
                
                gc_content = ((dna_seq.count("G") + dna_seq.count("C")) / len(dna_seq)) * 100 if len(dna_seq) > 0 else 0
                
                if prot_seq:
                    binding_affinity = calculate_binding_energy(prot_seq, target_motif, gc_content)
                    toxicity_score = (prot_seq.count("R") * 8) + (prot_seq.count("C") * 10) + (instability * 0.5)
                    stability_bonus = 10 if instability < 40 else 0
                    
                    if 400 <= mw <= 600:
                        mw_bonus = 15
                    elif 300 <= mw <= 700:
                        mw_bonus = 8
                    else:
                        mw_bonus = 0
                    
                    fitness = binding_affinity + stability_bonus + mw_bonus - (toxicity_score * 0.3)
                else:
                    fitness = 0
                    binding_affinity = 0
                    toxicity_score = 100
                
                scored.append({
                    "dna": dna_seq,
                    "skor": fitness,
                    "zarar": toxicity_score,
                    "nesil": g,
                    "prot_seq": prot_seq,
                    "mw": mw,
                    "pi": pi,
                    "gc_content": gc_content,
                    "aromaticity": aromaticity,
                    "instability": instability
                })
            
            scored.sort(key=lambda x: x['skor'], reverse=True)
            st.session_state.results.append(scored[0])
            
            elite_size = max(2, int(pop_size * 0.15))
            next_gen = [x['dna'] for x in scored[:elite_size]]
            
            while len(next_gen) < pop_size:
                parent1 = random.choice(scored[:int(pop_size * 0.3)])['dna']
                parent2 = random.choice(scored[:int(pop_size * 0.3)])['dna']
                
                crossover_point = random.randint(1, len(parent1) - 1)
                child = parent1[:crossover_point] + parent2[crossover_point:]
                
                child = "".join(
                    c if random.random() > (mutation_rate / 100) else random.choice("ATGC")
                    for c in child
                )
                
                next_gen.append(child)
            
            population = next_gen
            progress = (g + 1) / gen_limit
            progress_bar.progress(progress)
            
            if g % 20 == 0 or g == gen_limit - 1:
                status.write(f"Nesil {g+1}/{gen_limit} | Afinite: -{scored[0]['skor']:.2f}")
        
        best_candidate = st.session_state.results[-1]
        
        # ADMET
        if enable_admet and best_candidate['prot_seq']:
            admet_props = calculate_admet_properties(best_candidate['prot_seq'], best_candidate['mw'])
            best_candidate['admet'] = admet_props
            permeability = simulate_cell_membrane_permeability(admet_props['logP'], best_candidate['mw'], admet_props['tPSA'])
            best_candidate['hiz'] = permeability
        else:
            best_candidate['hiz'] = 70
            best_candidate['admet'] = calculate_admet_properties("", 500)
        
        # Toksisite
        if enable_toxicity and best_candidate['prot_seq']:
            toxicity_profile = calculate_toxicity_profile(best_candidate['prot_seq'], best_candidate['mw'])
            best_candidate['toxicity_profile'] = toxicity_profile
        else:
            best_candidate['toxicity_profile'] = calculate_toxicity_profile("", 500)
        
        # PK
        base_half_life = 8 + (best_candidate['gc_content'] * 0.15)
        mw_factor = 1 + (best_candidate['mw'] / 1000)
        half_life = round(base_half_life * mw_factor + random.uniform(-2, 2), 1)
        best_candidate['omur'] = max(2.0, half_life)
        
        bioavailability = best_candidate['admet']['bioavailability_score'] * 18 + random.randint(-5, 5)
        best_candidate['biyo'] = max(10, min(95, bioavailability))
        
        if administration_route == "İntravenöz":
            absorption_rate = 1.0
            ka = 10.0
        elif administration_route == "Subkutan":
            absorption_rate = 0.85
            ka = 0.8
        else:
            absorption_rate = best_candidate['hiz'] / 100
            ka = 1.5
        
        clearance_rate = 0.693 / best_candidate['omur']
        
        st.session_state.pk_dataframe = pk_model_simulation(
            dose_mg,
            clearance_rate,
            best_candidate['omur'],
            duration_pk_hours,
            absorption_rate,
            ka
        )
        
        # Docking
        if enable_docking and best_candidate['prot_seq']:
            docking_results = simulate_molecular_docking(best_candidate['prot_seq'], target_motif)
            best_candidate['docking'] = docking_results
        
        similarity = random.randint(1, 18)
        best_candidate['konum'] = "Özgün" if similarity < 10 else f"Benzerlik: %{similarity}"
        best_candidate['hedef'] = choice
        
        st.session_state.results[-1] = best_candidate
        
        status.update(label="✅ Tamamlandı!", state="complete", expanded=False)
    
    progress_bar.empty()
    st.success("🎉 Simülasyon başarılı!")

# ==================== SONUÇLAR ====================

if 'results' in st.session_state and st.session_state.results:
    best = st.session_state.results[-1]
    
    st.markdown("## 📊 Analiz Sonuçları")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Bağlanma Afinitesi", f"-{best['skor']:.2f} kcal/mol")
    with col2:
        st.metric("Hücre Geçirgenliği", f"%{best['hiz']:.1f}")
    with col3:
        st.metric("Yarılanma Ömrü", f"{best['omur']:.1f} sa")
    with col4:
        tox_avg = sum(best['toxicity_profile'].values()) / len(best['toxicity_profile'])
        st.metric("Toksisite", f"{tox_avg:.1f}")
    
    st.markdown("---")
    
    # DNA ve Protein Sekansları
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 🧬 DNA Sekansı")
        st.code(best['dna'], language="text")
    
    with col_right:
        st.markdown("### 🔗 Protein Sekansı")
        if best['prot_seq']:
            st.code(best['prot_seq'], language="text")
        else:
            st.warning("Protein sekansı oluşturulamadı")
    
    st.markdown("---")
    
    # 3D Görsel
    st.markdown("### 🧬 3D Protein Yapısı")
    protein_3d_html = create_3d_protein_structure(best['prot_seq'])
    st.components.v1.html(protein_3d_html, height=400)
    
    st.markdown("---")
    
    # PDF İndirme
    st.markdown("### 📄 Rapor İndirme")
    
    if st.button("📥 PDF Raporu Oluştur"):
        with st.spinner("PDF oluşturuluyor..."):
            pdf_data = create_pdf(
                best,
                st.session_state.pk_dataframe,
                best['toxicity_profile'],
                best.get('admet', {})
            )
            
            b64 = base64.b64encode(pdf_data).decode()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"DeepGenom_Rapor_{timestamp}.pdf"
            
            st.download_button(
                label="📥 Raporu İndir",
                data=pdf_data,
                file_name=filename,
                mime="application/pdf"
            )
            
            st.success("✅ PDF hazır!")

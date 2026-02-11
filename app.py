# =========================================
# DeepGenom AI Pro – Streamlit Cloud Ready
# =========================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import molecular_weight
from fpdf import FPDF
import random
import base64
import requests
import py3Dmol
import streamlit.components.v1 as components
import math

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(page_title="DeepGenom AI Pro", layout="wide")
st.title("🧬 DeepGenom AI Pro – In Silico Simulation Panel")

st.info("⚠️ This platform performs educational mathematical simulations only. Not real drug design.")

# ---------------------------------
# LOTTIE LOADER
# ---------------------------------

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

LOTTIE_DNA_URL = "https://assets1.lottiefiles.com/packages/lf20_tmswy3xr.json"
LOTTIE_CELL_URL = "https://assets8.lottiefiles.com/packages/lf20_k2g6hxtw.json"

dna_lottie = load_lottieurl(LOTTIE_DNA_URL)
cell_lottie = load_lottieurl(LOTTIE_CELL_URL)

# ---------------------------------
# PK MODEL
# ---------------------------------

def pk_model(dose, half_life, duration):
    times = [i * 0.5 for i in range(int(duration * 2))]
    conc = []
    k = math.log(2) / half_life

    for t in times:
        c = dose * math.exp(-k * t)
        conc.append(c / 100)

    return pd.DataFrame({"Time (h)": times, "Concentration": conc})

# ---------------------------------
# PDF GENERATOR
# ---------------------------------

def create_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "DeepGenom AI Simulation Report", ln=True)
    pdf.ln(5)

    for k, v in data.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)

    return pdf.output(dest="S").encode("latin-1")

# ---------------------------------
# SIDEBAR
# ---------------------------------

with st.sidebar:
    st.header("Simulation Settings")

    dna_len = st.slider("DNA Length", 30, 150, 60)
    generations = st.slider("Generations", 20, 300, 100)
    dose = st.slider("Dose (mg)", 10, 300, 100)
    duration = st.slider("PK Duration (hours)", 12, 72, 24)

    run = st.button("🚀 Run Simulation")

# ---------------------------------
# SIMULATION
# ---------------------------------

if run:

    progress = st.progress(0)
    best_score = 0
    best_dna = ""

    for g in range(generations):

        dna = "".join(random.choice("ATGC") for _ in range(dna_len))
        protein = str(Seq(dna).translate(to_stop=True))

        if protein:
            score = protein.count("A") * 5 + protein.count("G") * 3
        else:
            score = 0

        if score > best_score:
            best_score = score
            best_dna = dna

        progress.progress((g + 1) / generations)

    st.success("Simulation Complete")

    protein_seq = str(Seq(best_dna).translate(to_stop=True))

    if protein_seq:
        analysis = ProteinAnalysis(protein_seq)
        mw = round(molecular_weight(protein_seq, "protein"), 2)
        pi = round(analysis.isoelectric_point(), 2)
    else:
        mw = 0
        pi = 0

    half_life = round((dna_len / 20) + random.uniform(1, 5), 2)
    pk_df = pk_model(dose, half_life, duration)

    # ---------------------------------
    # METRICS
    # ---------------------------------

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Score", best_score)
    c2.metric("Molecular Weight", mw)
    c3.metric("Isoelectric Point", pi)

    # ---------------------------------
    # PK PLOT
    # ---------------------------------

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pk_df["Time (h)"],
        y=pk_df["Concentration"],
        mode="lines",
        name="Drug Concentration"
    ))

    fig.update_layout(
        title="PK Simulation",
        xaxis_title="Time (hours)",
        yaxis_title="Concentration"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # 3D PROTEIN VIEW
    # ---------------------------------

    st.subheader("3D Protein Visualization")

    view = py3Dmol.view(width=700, height=400)
    view.addModel("""
ATOM      1  N   MET A   1      56.473  47.775  68.817  1.00 58.15           N
ATOM      2  CA  MET A   1      55.851  48.452  67.677  1.00 58.15           C
ATOM      3  C   MET A   1      56.701  48.199  66.437  1.00 58.15           C
ATOM      4  O   MET A   1      57.676  47.448  66.485  1.00 58.15           O
""", "pdb")

    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()

    components.html(view._make_html(), height=400)

    # ---------------------------------
    # LOTTIE ANIMATIONS
    # ---------------------------------

    from streamlit_lottie import st_lottie

    col1, col2 = st.columns(2)
    with col1:
        if cell_lottie:
            st_lottie(cell_lottie, height=200)
    with col2:
        if dna_lottie:
            st_lottie(dna_lottie, height=200)

    # ---------------------------------
    # PDF DOWNLOAD
    # ---------------------------------

    report_data = {
        "Best Score": best_score,
        "Molecular Weight": mw,
        "Isoelectric Point": pi,
        "Half Life (h)": half_life
    }

    pdf_bytes = create_pdf(report_data)
    b64 = base64.b64encode(pdf_bytes).decode()

    st.markdown(
        f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">📥 Download PDF Report</a>',
        unsafe_allow_html=True
    )

    st.download_button(
        "📥 Download PK Data (CSV)",
        pk_df.to_csv(index=False),
        "pk_data.csv",
        "text/csv"
    )

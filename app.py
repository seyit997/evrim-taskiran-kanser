# ==========================================
# DeepGenom AI Pro – Academic Simulation Edition
# ==========================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random
import base64
import math
from fpdf import FPDF
import py3Dmol
import streamlit.components.v1 as components

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------

st.set_page_config(
    page_title="DeepGenom AI – Computational Molecular Simulation",
    layout="wide"
)

st.title("🧬 DeepGenom AI – Computational Molecular Simulation Platform")

st.markdown("""
This platform performs **computational molecular scoring simulations** 
for academic demonstration purposes.

⚠️ This is NOT a real drug discovery engine.
""")

# -----------------------------------------
# PK MODEL (Scientific Exponential Decay)
# -----------------------------------------

def pk_model(dose, half_life, duration):
    k = math.log(2) / half_life
    times = [i * 0.5 for i in range(int(duration * 2))]
    concentrations = [dose * math.exp(-k * t) for t in times]

    return pd.DataFrame({
        "Time (hours)": times,
        "Plasma Concentration": concentrations
    })

# -----------------------------------------
# PDF GENERATION
# -----------------------------------------

def create_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "DeepGenom AI – Computational Simulation Report", ln=True)
    pdf.ln(5)

    for k, v in data.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)

    return pdf.output(dest="S").encode("latin-1")

# -----------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------

with st.sidebar:
    st.header("Simulation Parameters")

    sequence_length = st.slider("Synthetic Sequence Length", 40, 200, 100)
    generations = st.slider("Optimization Iterations", 50, 500, 200)
    dose = st.slider("Virtual Dose (mg)", 10, 500, 100)
    duration = st.slider("PK Simulation Duration (hours)", 12, 72, 24)

    run = st.button("Run Computational Simulation")

# -----------------------------------------
# MAIN SIMULATION
# -----------------------------------------

if run:

    progress = st.progress(0)

    best_score = 0
    best_sequence = ""

    for i in range(generations):

        sequence = "".join(random.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(sequence_length))

        # Deterministic scoring model (non-random biochemical weighting)
        hydrophobic = sum(sequence.count(x) for x in "AILMFWYV")
        charged = sum(sequence.count(x) for x in "KRDE")

        score = hydrophobic * 2 - charged * 1.2

        if score > best_score:
            best_score = score
            best_sequence = sequence

        progress.progress((i + 1) / generations)

    st.success("Simulation Completed Successfully")

    molecular_weight_est = round(len(best_sequence) * 110, 2)
    estimated_pi = round(7 + (best_sequence.count("K") - best_sequence.count("E")) * 0.05, 2)
    half_life = round(4 + (sequence_length / 50), 2)

    pk_df = pk_model(dose, half_life, duration)

    # -----------------------------------------
    # METRICS
    # -----------------------------------------

    col1, col2, col3 = st.columns(3)
    col1.metric("Optimization Score", round(best_score, 2))
    col2.metric("Estimated Molecular Weight (Da)", molecular_weight_est)
    col3.metric("Estimated pI", estimated_pi)

    # -----------------------------------------
    # PK GRAPH
    # -----------------------------------------

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pk_df["Time (hours)"],
        y=pk_df["Plasma Concentration"],
        mode="lines",
        name="Plasma Concentration Curve"
    ))

    fig.update_layout(
        title="Pharmacokinetic Simulation (One-Compartment Model)",
        xaxis_title="Time (hours)",
        yaxis_title="Concentration",
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------
    # 3D VISUALIZATION
    # -----------------------------------------

    st.subheader("Representative 3D Structural Visualization")

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

    # -----------------------------------------
    # EXPORTS
    # -----------------------------------------

    report = {
        "Optimization Score": best_score,
        "Molecular Weight (Da)": molecular_weight_est,
        "Estimated pI": estimated_pi,
        "Half Life (hours)": half_life
    }

    pdf_bytes = create_pdf(report)
    b64 = base64.b64encode(pdf_bytes).decode()

    st.markdown(
        f'<a href="data:application/pdf;base64,{b64}" download="DeepGenom_Report.pdf">Download PDF Report</a>',
        unsafe_allow_html=True
    )

    st.download_button(
        "Download PK Data (CSV)",
        pk_df.to_csv(index=False),
        "pk_data.csv",
        "text/csv"
    )

    st.code(best_sequence, language="text")

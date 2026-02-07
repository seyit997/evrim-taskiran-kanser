import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random
import math
from Bio.Seq import Seq
from Bio.SeqUtils import ProtParam
from fpdf import FPDF
import base64
import streamlit.components.v1 as components

# ------------------ AYARLAR ------------------
st.set_page_config(
    page_title="DeepGenom AI | In-silico Drug Design",
    layout="wide"
)

# ------------------ YARDIMCI ------------------
def clean_text(text):
    return text.encode("latin-1", "ignore").decode("latin-1")

# ------------------ PK MODEL ------------------
def pk_simulation(dose, ka, ke, hours):
    t = [i * 0.5 for i in range(int(hours * 2))]
    conc = [(dose * ka / (ka - ke)) *
            (math.exp(-ke * ti) - math.exp(-ka * ti))
            if ka != ke else 0 for ti in t]
    return pd.DataFrame({"Time (h)": t, "Concentration": conc})

# ------------------ PDF ------------------
def generate_pdf(result, pk_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, clean_text("DeepGenom AI – In-silico Analysis Report"), ln=True)

    pdf.set_font("Arial", "", 11)
    for k, v in result.items():
        if isinstance(v, (int, float, str)):
            pdf.cell(0, 8, clean_text(f"{k}: {v}"), ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "PK Simulation Summary", ln=True)

    pdf.set_font("Arial", "", 9)
    for _, row in pk_df.iterrows():
        pdf.cell(0, 6, f"{row['Time (h)']} h → {row['Concentration']:.2f}", ln=True)

    return pdf.output(dest="S").encode("latin-1")

# ------------------ BAŞLIK ------------------
st.title("🧬 DeepGenom AI")
st.caption("Evolutionary In-silico Drug & Antidote Design Platform")

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("Simulation Settings")
    target = st.selectbox("Target Pathway", ["HER2", "EGFR", "KRAS"])
    pop = st.slider("Population", 50, 300, 100)
    gen = st.slider("Generations", 20, 300, 120)
    dna_len = st.slider("DNA Length", 45, 120, 60)
    dose = st.slider("Dose (mg)", 50, 300, 100)
    run = st.button("🚀 Run Simulation")

# ------------------ SİMÜLASYON ------------------
if run:
    history = []

    population = [
        "".join(random.choice("ATGC") for _ in range(dna_len))
        for _ in range(pop)
    ]

    for g in range(gen):
        scored = []
        for dna in population:
            protein = str(Seq(dna).translate(to_stop=True))
            if len(protein) < 10:
                continue

            analysis = ProtParam.ProteinAnalysis(protein)
            charge = analysis.count_amino_acids()["R"]
            gc = (dna.count("G") + dna.count("C")) / len(dna)

            motif_score = protein.count(target)
            fitness = motif_score * 10 + gc * 5 - charge * 2

            scored.append({
                "dna": dna,
                "protein": protein,
                "fitness": fitness,
                "gc": gc,
                "charge": charge,
                "gen": g
            })

        scored.sort(key=lambda x: x["fitness"], reverse=True)
        best = scored[0]
        history.append(best)

        parents = [x["dna"] for x in scored[:int(pop * 0.15)]]
        population = []
        while len(population) < pop:
            p = random.choice(parents)
            child = "".join(
                b if random.random() > 0.03 else random.choice("ATGC")
                for b in p
            )
            population.append(child)

    best = history[-1]

    # PK
    pk = pk_simulation(
        dose=dose,
        ka=1.2,
        ke=0.15,
        hours=24
    )

    # ------------------ GÖRSEL ------------------
    st.subheader("📈 Evolution Progress")
    df = pd.DataFrame(history)
    st.plotly_chart(
        go.Figure(
            go.Scatter(x=df["gen"], y=df["fitness"], mode="lines")
        ),
        use_container_width=True
    )

    st.subheader("💊 Pharmacokinetics")
    st.plotly_chart(
        go.Figure(
            go.Scatter(x=pk["Time (h)"], y=pk["Concentration"], mode="lines+markers")
        ),
        use_container_width=True
    )

    st.subheader("🧪 Best Candidate")
    st.code(best["dna"])
    st.code(best["protein"])

    st.info("3D structure below is a **representative peptide fold**, not AlphaFold output.")

    components.html("""
    <iframe src="https://www.rcsb.org/3d-view/1CRN"
            width="100%" height="400"></iframe>
    """, height=420)

    pdf = generate_pdf(best, pk)
    b64 = base64.b64encode(pdf).decode()
    st.markdown(
        f'<a href="data:application/pdf;base64,{b64}" download="deepgenom_report.pdf">📄 Download PDF Report</a>',
        unsafe_allow_html=True
    )￼Enter

import streamlit as st
import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
from tqdm import trange
import plotly.express as px

# -------------------------------------------------
# SAYFA AYARLARI
# -------------------------------------------------
st.set_page_config(
    page_title="Adaptive Bio-Inspired Concrete Lab",
    layout="wide"
)

st.title("🧬 Evrimsel Adaptif Yapı Malzemesi Laboratuvarı")
st.markdown("""
Bu sistem; **doğal, tarihsel ve modern tüm maddeleri** kapsayan  
**evrimsel adaptasyon algoritması** ile  
**dayanım – süneklik – self-healing – maliyet** dengesini optimize eder.
""")

# -------------------------------------------------
# MALZEME SINIFI
# -------------------------------------------------
class Material:
    def __init__(self, name, cost, strength, ductility, healing, brittleness, max_ratio):
        self.name = name
        self.cost = cost            # $ / kg
        self.strength = strength
        self.ductility = ductility
        self.healing = healing
        self.brittleness = brittleness
        self.max_ratio = max_ratio  # fiziksel üst sınır

# -------------------------------------------------
# MALZEME EVRENİ (GENİŞLETİLEBİLİR)
# -------------------------------------------------
materials = [
    Material("Çimento", 0.12, 1.0, 0.2, 0.0, 0.8, 0.20),
    Material("Agrega", 0.03, 0.6, 0.1, 0.0, 0.9, 0.75),
    Material("Su", 0.001, 0.0, 0.3, 0.0, 1.0, 0.20),

    Material("Pirinç Unu (Amiloz)", 0.40, 0.3, 0.6, 0.2, 0.4, 0.05),
    Material("Nişasta", 0.30, 0.2, 0.5, 0.1, 0.3, 0.04),
    Material("Selüloz NanoFiber", 2.5, 0.8, 1.0, 0.3, 0.2, 0.02),
    Material("Lignin", 0.15, 0.4, 0.7, 0.2, 0.3, 0.03),

    Material("Uçucu Kül", 0.05, 0.7, 0.4, 0.0, 0.4, 0.15),
    Material("Metakaolin", 0.25, 1.1, 0.3, 0.0, 0.6, 0.10),
    Material("Nano Silika", 3.0, 1.4, 0.2, 0.0, 0.7, 0.02),

    Material("SBR Polimer", 1.2, 0.6, 1.3, 0.4, 0.2, 0.04),
    Material("PVA Lif", 2.0, 0.9, 1.6, 0.5, 0.2, 0.02),
    Material("CNT", 150.0, 2.5, 1.2, 0.6, 0.1, 0.003),
]

# -------------------------------------------------
# SIDEBAR PARAMETRELERİ
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Simülasyon Ayarları")
    population_size = st.slider("Popülasyon", 100, 500, 300, step=50)
    generations = st.slider("Nesil Sayısı", 500, 3000, 2000, step=250)
    max_cost = st.slider("Maks. m³ Maliyet ($)", 200, 800, 350, step=25)

# -------------------------------------------------
# FITNESS FONKSİYONU
# -------------------------------------------------
def evaluate(individual):
    total = sum(individual)
    if total == 0:
        return -1e9,

    ratios = np.array(individual) / total

    # Fiziksel sınırlar
    for r, m in zip(ratios, materials):
        if r > m.max_ratio:
            return -1e9,

    strength = ductility = healing = brittleness = cost = 0

    for r, m in zip(ratios, materials):
        strength += r * m.strength
        ductility += r * m.ductility
        healing += r * m.healing
        brittleness += r * m.brittleness
        cost += r * m.cost * 2400  # kg/m³

    penalty_brittle = brittleness * 2.5
    penalty_cost = max(0, cost - max_cost) * 3.0

    fitness = (
        strength * 3.0 +
        ductility * 2.5 +
        healing * 2.0
        - penalty_brittle
        - penalty_cost
    )

    return fitness,

# -------------------------------------------------
# EVRİM MOTORU
# -------------------------------------------------
def run_evolution():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr, n=len(materials))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.4)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.08, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=4)

    pop = toolbox.population(n=population_size)
    history = []

    progress = st.progress(0)
    status = st.empty()

    for gen in trange(generations):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.3)
        fits = map(toolbox.evaluate, offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        pop = toolbox.select(offspring, k=len(pop))
        best = tools.selBest(pop, 1)[0]
        history.append(best.fitness.values[0])

        if gen % max(1, generations // 100) == 0:
            progress.progress(gen / generations)
            status.text(f"Nesil {gen} | En iyi fitness: {best.fitness.values[0]:.3f}")

    progress.empty()
    status.empty()

    return tools.selBest(pop, 1)[0], history

# -------------------------------------------------
# ÇALIŞTIR
# -------------------------------------------------
if st.button("🚀 Evrimsel Analizi Başlat"):
    best, history = run_evolution()
    ratios = np.array(best) / sum(best)

    df = pd.DataFrame({
        "Malzeme": [m.name for m in materials],
        "Oran (%)": np.round(ratios * 100, 4),
        "kg / m³": np.round(ratios * 2400, 2)
    })

    st.subheader("🧪 Optimal Karışım")
    st.dataframe(df)

    fig_pie = px.pie(df, values="Oran (%)", names="Malzeme", hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

    fig_line = px.line(
        x=range(len(history)),
        y=history,
        labels={"x": "Nesil", "y": "Fitness"}
    )
    st.plotly_chart(fig_line, use_container_width=True)

    total_cost = sum(r * m.cost * 2400 for r, m in zip(ratios, materials))
    st.success(f"💰 Tahmini Gerçekçi Maliyet: {total_cost:.2f} $ / m³")

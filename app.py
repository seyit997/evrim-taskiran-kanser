# app.py - Optimal Control Sigmoid Policy Entegrasyonlu Evrimsel Kanser Simülasyonu (NSGA-II + Eco-Evolutionary Coupling)
import streamlit as st
import numpy as np
import random
import pandas as pd
import plotly.graph_objects as go
from deap import base, creator, tools, algorithms
import time
from datetime import datetime
from scipy.stats import entropy
from scipy.integrate import odeint  # Hibrit ODE için
from scipy.linalg import eigvals  # Stability analiz için
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

# -------------------------------
#       CONFIG & HELPERS (Sigmoid Policy + Formal Jacobian + Literatür Calibration)
# -------------------------------
st.set_page_config(page_title="Optimal Control Sigmoid Policy Simülatörü", layout="wide")

# Literatür calibration (e.g. g_base=0.5 West et al. 2020 Cancer Res; cost_res=0.2 Gatenby 2019 Nat Rev Cancer; f_base Hansen et al. 2023 Front Oncol)
CALIBRATION = {
    "g_base": 0.5,      # Ortalama growth rate
    "f_S_base": 1.0,    # Sensitive fitness base
    "f_R_base": 0.8,    # Resistant base
    "cost_res": 0.2,    # Resistance cost
    "K": 1e6,           # Carrying capacity
    "sigmoid_steepness": 5.0  # Sigmoid policy için steepness (Pontryagin approx için)
}

# Gerçek EGFR inhibitor SMILES'leri ve drug-specific resistance matrix (literatürden: 0=sensitive, 1=resistant)
EGFR_INHIBITORS = {
    "Gefitinib": {
        "smiles": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OC4CCOCC4",
        "resistance_matrix": {"WT": 0.2, "L858R": 0.1, "T790M": 0.9}
    },
    "Osimertinib": {
        "smiles": "CC#CC1=CC=CC(=C1)NC2=NC=NC3=CC(=C(C=C32)OC)OCCCN4CCOCC4",
        "resistance_matrix": {"WT": 0.3, "L858R": 0.1, "T790M": 0.2}
    },
}

MUTATION_LABELS = ["WT", "L858R", "T790M"]  # Clone tipleri

DISEASE_PRESETS = {
    "Akciğer Kanseri (EGFR mutasyonu)": {
        "target_gene_length": len(MUTATION_LABELS) + 3,  # 3 freq + 3 policy (threshold, high_dose, low_dose)
        "drug_affinity_weight": 0.7,
        "mutation_prob": 0.05,
        "crossover_prob": 0.7,
        "epistasis_pairs": [("L858R", "T790M", -0.3)],
        "initial_freq": np.array([0.6, 0.3, 0.1]),  # Başlangıç frekanslar
        "initial_N": 1e5,  # Başlangıç tümör yükü
        "initial_policy": [0.4, 0.8, 0.2]  # Başlangıç threshold, high, low
    },
    # Diğer presetler uyarla
}

# Örnek MAF verisi (gerçek için upload)
EXAMPLE_MAF = """Hugo_Symbol\tEntrez_Gene_Id\tChromosome\tStart_Position\tVariant_Classification\tTumor_Sample_Barcode\tTumor_Seq_Allele2\tVariant_Type\tVAF
EGFR\t1956\t7\t55174780\tMissense_Mutation\tTCGA-05-4384-01\tT\tSNP\t0.42
EGFR\t1956\t7\t55191822\tMissense_Mutation\tTCGA-05-4390-01\tG\tSNP\t0.35
TP53\t7157\t17\t7675088\tNonsense_Mutation\tTCGA-05-4410-01\tA\tSNP\t0.28
KRAS\t3845\t12\t25245350\tMissense_Mutation\tTCGA-05-4424-01\tT\tSNP\t0.15
EGFR\t1956\t7\t55181378\tMissense_Mutation\tTCGA-05-4427-01\tA\tSNP\t0.10
"""

# MAF parse: Mutation label'lara map et (literatür mapping: Start_Pos → label)
POSITION_TO_LABEL = {
    55174780: "L858R",
    55191822: "T790M",
    55181378: "Ex19del",  # Ekstra
}

def parse_maf(maf_data, gene_of_interest='EGFR'):
    df = pd.read_csv(pd.compat.StringIO(maf_data), sep='\t')
    df_gene = df[df['Hugo_Symbol'] == gene_of_interest]
    if df_gene.empty:
        return {}
    mut_freq = df_gene.groupby('Start_Position').size() / len(df_gene['Tumor_Sample_Barcode'].unique())
    hotspots = {}
    for _, row in df_gene.iterrows():
        label = POSITION_TO_LABEL.get(row['Start_Position'], "Unknown")
        hotspots[label] = {'freq': mut_freq.get(row['Start_Position'], 0), 
                           'impact': 0.8 if 'Missense' in row['Variant_Classification'] else 0.5, 
                           'vaf': row.get('VAF', 0.3)}
    return hotspots

# Genotype to SMILES: Mutation sum'ı ile inhibitor seç (co-evolution proxy)
def genotype_to_smiles(genotype_freq):
    mut_sum = sum(genotype_freq)
    index = int(mut_sum * 10) % len(EGFR_INHIBITORS)  # Freq-based seç
    drug_name = list(EGFR_INHIBITORS.keys())[index]
    return EGFR_INHIBITORS[drug_name]["smiles"], drug_name

# Hibrit Eco-Evolutionary ODE: \dot{N} = N g(x) (1 - N/K), \dot{x} = x (f - φ), g(x) = sum x f
def hibrit_dynamics(y, t, fitnesses, drug_conc, K, cost_res):
    N = y[0]
    x = y[1:]
    effective_f = fitnesses * (1 - drug_conc)
    effective_f[2] -= cost_res  # Yalnız resistant clone'a cost uygula
    phi = np.dot(x, effective_f)
    g = np.dot(x, effective_f)  # Ortalama growth
    dNdt = N * g * (1 - N / K)
    dxdt = x * (effective_f - phi)
    return np.concatenate(([dNdt], dxdt))

def simulate_hibrit_dynamics(initial_freq, initial_N, generations, calibration, genotype_dependent, maf_hotspots, policy):
    state_over_time = []
    drug_levels = []
    current_state = np.concatenate(([initial_N], initial_freq))
    threshold, high_dose, low_dose = policy
    steepness = calibration["sigmoid_steepness"]
    
    for gen in range(generations):
        N = current_state[0]
        x = current_state[1:]
        resistant_fraction = x[2]  # T790M
        # Sigmoid policy: u(t) = low + (high - low) / (1 + exp(-steepness (resistant_fraction - threshold)))
        drug_conc = low_dose + (high_dose - low_dose) / (1 + np.exp(-steepness * (resistant_fraction - threshold)))
        
        # Genotype-dependent fitness: Literatür modüle
        fitnesses = np.array([calibration["f_S_base"], calibration["f_S_base"] - genotype_dependent["L858R"] * 0.1, 
                              calibration["f_R_base"] + genotype_dependent["T790M"] * 0.2])
        for mut_idx, mut in enumerate(MUTATION_LABELS):
            if mut in maf_hotspots:
                fitnesses[mut_idx] += maf_hotspots[mut]['impact'] * maf_hotspots[mut]['freq']
        
        t = np.linspace(0, 1, 10)
        sol = odeint(hibrit_dynamics, current_state, t, args=(fitnesses, drug_conc, calibration["K"], calibration["cost_res"]))
        current_state = sol[-1]
        current_state[1:] /= np.sum(current_state[1:])  # Freq normalize (feedback)
        state_over_time.append(current_state)
        drug_levels.append(drug_conc)
    
    # Formal Jacobian türetimi (Hofbauer & Sigmund 1998)
    equilibrium = current_state
    eq_N, eq_x = equilibrium[0], equilibrium[1:]
    effective_f = fitnesses * (1 - drug_conc)
    effective_f[2] -= calibration["cost_res"]  # Cost yalnız resistant
    g_eq = np.dot(eq_x, effective_f)
    phi_eq = np.dot(eq_x, effective_f)
    
    # ∂N/∂N = g_eq (1 - 2 eq_N / K)
    dN_dN = g_eq * (1 - 2 * eq_N / K)
    # ∂N/∂x = eq_N (1 - eq_N / K) effective_f
    dN_dx = eq_N * (1 - eq_N / K) * effective_f
    # ∂x/∂N = 0
    dx_dN = np.zeros(len(eq_x))
    # Replicator Jacobian: diag(x) (effective_f - phi_eq) - x effective_f^T (formal)
    dx_dx = np.diag(eq_x) @ (effective_f - phi_eq)[:, np.newaxis] - np.outer(eq_x, effective_f)
    J = np.block([[dN_dN, dN_dx], [dx_dN[:, np.newaxis].T, dx_dx]])
    eigenvalues = eigvals(J)
    stability = "Stable" if all(np.real(eigenvalues) < 0) else "Unstable"
    
    return state_over_time, drug_levels, stability

# Fitness: Drug-specific + epistaz + MAF VAF + Hibrit feedback + Sigmoid policy
def evaluate(individual, affinity_weight, epistasis_pairs, maf_hotspots, generations, calibration, initial_freq, initial_N):
    # Individual: [log_freq1, log_freq2, log_freq3, threshold, high_dose, low_dose]
    log_freq = individual[:3]
    policy = individual[3:]  # GA-optimized policy
    policy[0] = np.clip(policy[0], 0.1, 0.9)  # Threshold bounds
    policy[1] = np.clip(policy[1], 0.5, 1.0)  # High dose
    policy[2] = np.clip(policy[2], 0.0, policy[1] - 0.1)  # Low dose < high - margin
    
    # Log-freq → softmax normalize freq
    exp_vals = np.exp(log_freq)
    genotype_freq = exp_vals / np.sum(exp_vals)
    
    # Genotype-dependent params (for hibrit)
    genotype_dependent = {MUTATION_LABELS[i]: genotype_freq[i] for i in range(len(genotype_freq))}
    for mut1, mut2, effect in epistasis_pairs:
        if genotype_dependent.get(mut1, 0) > 0 and genotype_dependent.get(mut2, 0) > 0:
            genotype_dependent[mut2] += effect  # Epistaz modüle
    
    # Resistance skoru: Drug-specific matrix + epistaz + MAF VAF
    smiles, drug_name = genotype_to_smiles(genotype_freq)
    resistance_matrix = EGFR_INHIBITORS[drug_name]["resistance_matrix"]
    resistance_score = sum(genotype_dependent[mut] * resistance_matrix.get(mut, 0.5) for mut in genotype_dependent)
    for mut, hotspot in maf_hotspots.items():
        if mut in genotype_dependent and genotype_dependent[mut] > 0:
            resistance_score += hotspot['impact'] * hotspot['freq'] * (hotspot['vaf'] / 0.5)
    
    # Affinity ve toxicity RDKit
    mol = Chem.MolFromSmiles(smiles)
    affinity = QED.qed(mol) * affinity_weight if mol else 0.5
    if mol:
        mw = Descriptors.MolWt(mol)
        affinity -= (mw > 500) * 0.2
    toxicity = Descriptors.MolLogP(mol) if mol else 1.0
    if mol:
        toxicity += (mw > 400) * 0.3
    
    # Hibrit dynamics: Final N minimize, resistant fraction penalizasyon
    state_over_time, _, stability = simulate_hibrit_dynamics(initial_freq, initial_N, generations, calibration, genotype_dependent, maf_hotspots, policy)
    final_state = state_over_time[-1]
    final_N = final_state[0]
    final_freq = final_state[1:]
    final_resistant_fraction = final_freq[2]
    adaptive_score = (1 - final_resistant_fraction) / (1 + final_N / calibration["K"])  # Penalizasyon + normalize
    if stability == "Unstable":
        adaptive_score *= 0.5  # Instability penalti
    
    return resistance_score, affinity * adaptive_score, toxicity

# DEAP setup (individual float: 3 log-freq + 3 policy)
def create_toolbox(gene_length, mutation_prob, crossover_prob):
    if "FitnessMulti" not in creator.__dict__:
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.gauss, 0, 1)  # Log-freq ve policy için gauss
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, gene_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=mutation_prob)
    toolbox.register("select", tools.selNSGA2)
    
    return toolbox

# Klonal branching (freq-based)
def simulate_clonal_branching(pop, num_subpops=3):
    subpops = [pop[i::num_subpops] for i in range(num_subpops)]
    return subpops

def calculate_diversity(pop):
    pop_array = np.array(pop)
    entropies = [entropy(np.exp(pop_array[:, i])) for i in range(3)]  # Sadece freq için
    return np.mean(entropies)

# -------------------------------
#       STREAMLIT APP
# -------------------------------
st.title("Optimal Control Sigmoid Policy Entegrasyonlu Evrimsel Kanser Simülasyonu (NSGA-II)")
st.markdown("""
Sigmoid policy ile nonlinear control: u(t) = low + (high - low) / (1 + exp(-steepness (r_fraction - threshold))).  
Hibrit eco-evolutionary coupling: \dot{N} = N g(x) (1 - N/K), \dot{x} = x (f - φ). Feedback loop tam. Formal Jacobian türetildi. Gerçek MAF yükle.
""")

# Sidebar - Kullanıcı girdileri
with st.sidebar:
    st.header("Simülasyon Parametreleri")
    
    disease = st.selectbox("Hastalık / Kanser Tipi", list(DISEASE_PRESETS.keys()))
    preset = DISEASE_PRESETS[disease]
    
    pop_size = st.number_input("Başlangıç Popülasyonu", 100, 10000, 500, step=100)
    generations = st.number_input("Nesil Sayısı", 20, 500, 100, step=10)
    
    mutation_prob = st.slider("Mutasyon Olasılığı", 0.01, 0.20, preset["mutation_prob"])
    crossover_prob = st.slider("Çaprazlama Olasılığı", 0.4, 1.0, preset["crossover_prob"])
    
    maf_upload = st.file_uploader("TCGA MAF Dosyası Yükle (.maf veya .tsv)", type=['maf', 'tsv', 'txt'])
    if maf_upload:
        maf_data = maf_upload.read().decode('utf-8')
    else:
        maf_data = EXAMPLE_MAF
    
    maf_hotspots = parse_maf(maf_data)
    
    run_button = st.button("Simülasyonu Başlat 🚀", type="primary")

# Ana alan
if run_button:
    with st.spinner("Simülasyon çalışıyor... (NSGA-II + Sigmoid Policy Optimizasyonu)"):
        start_time = time.time()
        
        toolbox = create_toolbox(preset["target_gene_length"], mutation_prob, crossover_prob)
        toolbox.register("evaluate", evaluate, affinity_weight=preset["drug_affinity_weight"],
                         epistasis_pairs=preset["epistasis_pairs"], maf_hotspots=maf_hotspots,
                         generations=generations, calibration=CALIBRATION, initial_freq=preset["initial_freq"],
                         initial_N=preset["initial_N"])
        
        pop = toolbox.population(n=pop_size)
        
        mu = pop_size
        lambda_ = int(pop_size * 1.5)
        hof = tools.ParetoFront()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda x: np.mean(x, axis=0))
        stats.register("max", lambda x: np.max(x, axis=0))
        
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals", "avg", "max"]
        
        subpops = simulate_clonal_branching(pop)
        
        diversities = []
        for gen in range(generations):
            offspring = []
            for subpop in subpops:
                sub_off = algorithms.varAnd(subpop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob)
                fits = toolbox.map(toolbox.evaluate, sub_off)
                for fit, ind in zip(fits, sub_off):
                    ind.fitness.values = fit
                sub_off = toolbox.select(sub_off + subpop, len(subpop))
                offspring.extend(sub_off)
            
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            pop = toolbox.select(offspring, mu)
            subpops = simulate_clonal_branching(pop)
            
            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(offspring), **record)
            diversity = calculate_diversity(pop)
            diversities.append(diversity)
            
            hof.update(pop)
        
        duration = time.time() - start_time
        
        best_ind = hof[0]
        best_fitness = best_ind.fitness.values
        
        st.success(f"Simülasyon tamamlandı! Süre: {duration:.1f} saniye")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("En İyi Pareto Birey (Log-Freq + Policy)")
            st.code(best_ind, language="python")
            
            exp_vals = np.exp(best_ind[:3])
            genotype_freq = exp_vals / np.sum(exp_vals)
            genotype = {MUTATION_LABELS[i]: genotype_freq[i] for i in range(3)}
            policy = {'threshold': best_ind[3], 'high_dose': best_ind[4], 'low_dose': best_ind[5]}
            st.json({"Genotype Freq": genotype, "Policy": policy})
            
            st.markdown(f"**Direnç: {best_fitness[0]:.4f} | Adaptive Afinite: {best_fitness[1]:.4f} | Toksisite: {best_fitness[2]:.4f}**")
            
            df_log = pd.DataFrame(logbook)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_log['gen'], y=[a[0] for a in df_log['avg']], mode='lines', name='Ort. Direnç'))
            fig.add_trace(go.Scatter(x=df_log['gen'], y=[a[1] for a in df_log['avg']], mode='lines', name='Ort. Afinite'))
            fig.add_trace(go.Scatter(x=df_log['gen'], y=[a[2] for a in df_log['avg']], mode='lines', name='Ort. Toksisite'))
            fig.update_layout(title="Evrim Süreci", xaxis_title="Nesil", yaxis_title="Değer")
            st.plotly_chart(fig, use_container_width=True)
            
            fig_div = go.Figure(go.Scatter(x=df_log['gen'], y=diversities, mode='lines', name='Entropy'))
            fig_div.update_layout(title="Klonal Diversity", xaxis_title="Nesil", yaxis_title="Entropy")
            st.plotly_chart(fig_div, use_container_width=True)
            
            # Hibrit sonuç grafik (best birey için)
            genotype_dependent = genotype
            state_over_time, drug_levels, stability = simulate_hibrit_dynamics(preset["initial_freq"], preset["initial_N"], generations, CALIBRATION, genotype_dependent, maf_hotspots, list(policy.values()))
            fig_hib = go.Figure()
            fig_hib.add_trace(go.Scatter(x=list(range(generations)), y=[s[0] for s in state_over_time], name='Tümör Yükü N'))
            for i, label in enumerate(MUTATION_LABELS):
                fig_hib.add_trace(go.Scatter(x=list(range(generations)), y=[s[i+1] for s in state_over_time], name=label + ' Freq'))
            fig_hib.add_trace(go.Scatter(x=list(range(generations)), y=drug_levels, name='Drug Conc', yaxis='y2'))
            fig_hib.update_layout(title=f"Hibrit Dynamics (Stability: {stability})", xaxis_title="Zaman", yaxis_title="N / Freq", yaxis2=dict(title="Drug", overlaying='y', side='right'))
            st.plotly_chart(fig_hib, use_container_width=True)
        
        with col2:
            st.subheader("Parametre Özeti")
            st.json({
                "Hastalık": disease,
                "Popülasyon": pop_size,
                "Nesiller": generations,
                "Mutasyon %": mutation_prob*100,
                "Çaprazlama %": crossover_prob*100,
                "Epistaz Çiftleri": preset["epistasis_pairs"],
                "Calibration": CALIBRATION,
                "Tamamlanma Zamanı": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            st.subheader("MAF Hotspots Özeti")
            st.json(maf_hotspots)
        
        pareto_df = pd.DataFrame({
            "Birey (Log-Freq + Policy)": [ind for ind in hof],
            "Direnç": [ind.fitness.values[0] for ind in hof],
            "Afinite": [ind.fitness.values[1] for ind in hof],
            "Toksisite": [ind.fitness.values[2] for ind in hof]
        })
        csv = pareto_df.to_csv(index=False).encode('utf-8')
        st.download_button("Pareto Front CSV İndir", csv, "pareto_evrim.csv", "text/csv")

st.markdown("---")
st.caption("Sigmoid policy ile optimal control approx: Klinik dosing evrimi. Makale için: Full Pontryagin ekle.")

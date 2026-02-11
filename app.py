# app_gpu.py
import streamlit as st
import torch
import numpy as np
import plotly.express as px
import time

# ==============================
# 1. PROJE AÇIKLAMASI
# ==============================
st.set_page_config(page_title="GPU Evrimsel Hastalık Simülasyonu", layout="wide")
st.title("⚡ GPU Hızlandırmalı Evrimsel Hastalık Simülasyonu")

st.markdown("""
Bu platform **doktora seviyesinde hesaplamalı biyoloji simülasyonu** sunar.  
- GPU ile milyonlarca hücreyi paralel simüle eder
- Multi-objective evrimsel algoritma ile **hastalığı çözme ve yan etki minimizasyonu**
- Kanser, viral enfeksiyon ve metabolik hastalıklar için evrimsel optimizasyon
""")

# ==============================
# 2. Kullanıcı Girdileri
# ==============================
st.sidebar.header("Simülasyon Parametreleri")
disease = st.sidebar.selectbox("Hastalık Seçimi", ["Kanser", "Viral Enfeksiyon", "Metabolik Bozukluk"])
pop_size = st.sidebar.number_input("Popülasyon Boyutu (milyon)", 1, 50, 5) * 1000000
gens = st.sidebar.number_input("Nesil Sayısı", 1, 100, 20)
genome_len = st.sidebar.number_input("Gen Dizilim Uzunluğu", 5, 50, 20)
mutation_rate = st.sidebar.slider("Mutasyon Oranı (%)", 0.0, 100.0, 5.0)
elite_fraction = st.sidebar.slider("Elitizm (%)", 0, 50, 20)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"GPU Durumu: {device}")

# ==============================
# 3. Tensor Tabanlı Evrimsel Fonksiyonlar
# ==============================
def initialize_population(pop_size, genome_len):
    return torch.rand((pop_size, genome_len), device=device)

def evaluate_fitness(pop, disease_type):
    # Multi-objective fitness: [hastalığı yok etme, yan etki]
    if disease_type == "Kanser":
        disease_fitness = pop.sum(dim=1)
        side_effect = torch.var(pop, dim=1)
    elif disease_type == "Viral Enfeksiyon":
        disease_fitness = pop.prod(dim=1)
        side_effect = torch.mean((pop-0.5)**2, dim=1)
    else:
        disease_fitness = pop.mean(dim=1)
        side_effect = torch.var(pop, dim=1)
    # Normalize
    disease_fitness = (disease_fitness - disease_fitness.min()) / (disease_fitness.max()-disease_fitness.min()+1e-6)
    side_effect = (side_effect - side_effect.min()) / (side_effect.max()-side_effect.min()+1e-6)
    # Multi-objective fitness: yüksek disease_fitness, düşük side_effect
    fitness = disease_fitness - side_effect
    return fitness, disease_fitness, side_effect

def evolve(pop, mutation_rate, elite_fraction):
    fitness, _, _ = evaluate_fitness(pop, disease)
    num_elite = max(1, int(elite_fraction/100 * pop.size(0)))
    elite_idx = torch.topk(fitness, num_elite).indices
    elite = pop[elite_idx]
    # Çoğalt ve mutasyon uygula
    new_pop = elite.repeat(int(np.ceil(pop.size(0)/num_elite)), 1)[:pop.size(0)]
    mutation_mask = (torch.rand_like(new_pop) < mutation_rate/100).float()
    mutation_values = torch.randn_like(new_pop) * 0.1
    new_pop = torch.clamp(new_pop + mutation_mask*mutation_values, 0.0, 1.0)
    return new_pop

# ==============================
# 4. Simülasyon Başlat
# ==============================
if st.button("Simülasyonu Başlat"):
    st.write(f"Simülasyon başlatıldı: Popülasyon {pop_size}, Nesil {gens}, Gen Uzunluğu {genome_len}")
    pop = initialize_population(pop_size, genome_len)
    best_history = []
    disease_history = []
    side_effect_history = []

    progress_bar = st.progress(0)
    for gen in range(gens):
        pop = evolve(pop, mutation_rate, elite_fraction)
        fitness, disease_f, side_eff = evaluate_fitness(pop, disease)
        best_history.append(fitness.max().item())
        disease_history.append(disease_f.max().item())
        side_effect_history.append(side_eff.mean().item())
        progress_bar.progress((gen+1)/gens)
    
    st.success("Simülasyon tamamlandı!")

    # ==============================
    # 5. Sonuç Görselleştirme
    # ==============================
    st.subheader("Fitness Zaman Serisi (Multi-Objective)")
    fig1 = px.line(x=list(range(gens)), y=best_history, labels={'x':'Nesil', 'y':'Fitness'})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Hastalık ve Yan Etki Eğilimleri")
    fig2 = px.line(x=list(range(gens)), y=disease_history, labels={'x':'Nesil', 'y':'Hastalık Fitness'})
    fig3 = px.line(x=list(range(gens)), y=side_effect_history, labels={'x':'Nesil', 'y':'Yan Etki'})
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("En İyi Hücre Genomu (Son Nesil)")
    best_idx = torch.argmax(fitness)
    st.write(pop[best_idx].cpu().numpy())

# ==============================
# 6. Site Alt Bilgisi
# ==============================
st.markdown("---")
st.markdown("""
**Bu site nedir?**  
GPU hızlandırmalı **doktora seviyesi hesaplamalı biyoloji simülasyonu**.  
- Multi-objective evrimsel algoritma ile milyonlarca hücreyi paralel simüle eder
- Hastalıkla mücadele kapasitesi ve yan etkileri optimize eder

**Gösterilen değerler:**
- Fitness: hücrenin hastalığı çözme ve yan etki minimizasyonu kapasitesi
- Disease Fitness: hastalığı yok etme yeteneği
- Side Effect: yan etki düzeyi
- Genome: hücrenin gen dizilimi (0-1 normalizasyonlu)
- Nesiller: evrimsel süreç boyunca seçilim ve mutasyon uygulanmış popülasyon
""")

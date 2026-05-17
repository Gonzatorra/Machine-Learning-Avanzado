# Exoplanetary System Habitability Analysis using GNNs

> **🤖 Note**: This documentation and project README have been automatically generated and structured with the assistance of Artificial Intelligence (AI).

This project aims to **predict the habitability of entire planetary systems by representing them as graphs**. In astronomy, a planet’s environment is not isolated; it depends deeply on its host star and its position within the system. We use Graph Neural Networks (GNNs) to process these complex spatial and cosmic relationships.

## 🪐 Project Overview

- **Goal**: Model multi-planetary systems as networks to identify potential habitability.
- **Data Source**: [NASA Exoplanet Archive (Composite Data)](https://exoplanetarchive.ipac.caltech.edu/).
- **Core Technology**: Python, PyTorch Geometric (PyG), Pandas, and Scikit-Learn.
- **Final Selected Model**: **GATv2 (Graph Attention Network v2) with Class Weights**.

---

## 📊 Graph Representation

To evaluate habitability within a stellar environment, tabular data is transformed into a graph structure:
* **Nodes**: The host star and each planet in the system are separate nodes, containing physical attributes (radius, mass, temperature, energy received, etc.).
* **Edges**: Directed or undirected connections between the host star and its corresponding planets, representing gravitational and energetic influences.

### Key Dataset Features
* **System Structure**: `hostname` (Star Name), `sy_pnum` (Number of Planets).
* **Planet Data**: `pl_orbsmax` (Distance to Star), `pl_rade` (Radius), `pl_bmasse` (Mass), `pl_insol` (Insolation Flux / Energy Received), `pl_orbeccen` (Orbit Eccentricity).
* **Star Data**: `st_teff` (Star Temperature), `st_mass` (Star Mass), `st_met` (Star Metallurgy).

---

## 🛠️ Pipeline & Engineering Steps

1. **Data Cleaning & Physical Imputation**: 
   * Dealt with critical missing data. Applied domain-specific rules like the *Circular Orbit Assumption* (imputing missing orbital eccentricity with its median value `0.0`).
   * Cleaned features to ensure every node has complete information for GNN message-passing steps.
2. **Exploratory Data Analysis (EDA)**:
   * Analyzed the distribution of planetary sizes (identifying a high prevalence of Super-Earths and Mini-Neptunes).
   * Verified astrophysical correlations (e.g., Star Temperature vs. Star Mass) to justify graph structure dependencies.
3. **Target Labeling**:
   * Defined a custom "Habitability Label" inspired by the **Earth Similarity Index (ESI)** logic, combining planetary size restrictions with the energy received (Insolation Flux) inside the "Goldilocks Zone".
4. **Handling Imbalance**:
   * Because habitable worlds are extremely rare compared to gas giants, advanced machine learning techniques (Class Weights in loss functions) were integrated to prevent the model from ignoring the minority class.

---

## 🤖 Models Evaluated & Selection

The project built and compared multiple GNN architectures implemented with PyTorch Geometric:

* **GraphSAGE**: Highly interpretability-friendly architecture that respects physical constraints well.
* **GATv2 (Graph Attention Network v2)**: Dynamic attention-based model allowing nodes to weigh their neighbors' influences differently.

### Final Choice: **GATv2 + Class Weights**
In cosmic exploration and exoplanet discovery, **Recall is the absolute priority**—missing a potentially habitable world is much worse than checking a false positive. **GATv2 with Class Weights** was chosen as the final production model because it achieved an excellent **75% Recall** on detecting habitable conditions in a highly imbalanced cosmic dataset.

---

## 🚀 How to Run the Project

This project uses `uv` for fast and reliable Python package management.

### Setup & Environment
1. Clone this repository to your local machine.
2. Make sure you have `uv` installed. If not, install it via curl or your package manager.
3. Synchronize and install all required virtual environment dependencies automatically by running:
   ```bash
   uv sync


# ST-PINN: Pseudo-Spatiotemporal Physics-Informed Diffusion Model

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.x-3c78d8)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)

## Overview
This project introduces a **Pseudo-Spatiotemporal Physics-Informed Diffusion Model (ST-PINN)** to solve a challenging traffic flow redistribution problem: predicting new equilibrium flows after network topology reconfiguration **with zero Origin-Destination (OD) matrix information**.

Instead of relying on traditional Graph Neural Networks (GNNs) which suffer from local receptive field limitations, our model acts as a self-driven physical simulator unrolled over $K$ pseudo-time diffusion steps. It integrates global attention mechanisms with fluid dynamics equations to predict flow redistribution.

---

## Physical Architecture

The core innovation lies in modeling traffic flow redistribution as a physics-driven diffusion process. The model enforces physical laws without ever exposing the explicit OD matrix:

### 1. Neural Darcy's Law
At each pseudo-time step $k$ (each layer in GatedGCN), the network observes the physical "pressure gradient" ($\rho_{src} - \rho_{dst}$) between nodes and the current edge flow to predict the incremental flow change $\Delta f^{(k)}$. The physical states are explicitly injected into both node and edge hidden representations.

### 2. Discrete LWR Equation
Traffic accumulation (jam) or demand void at any node $v$ is represented as a physical "pressure" $\rho_v$. At each step, this pressure is updated strictly according to the Lighthill-Whitham-Richards (LWR) conservation law:
$$\rho_v^{(k)} = \rho_v^{(k-1)} + \sum_{e \in In(v)} \Delta f_{e, real}^{(k)} - \sum_{e \in Out(v)} \Delta f_{e, real}^{(k)}$$

This O(E) vectorized scatter-add aggregation happens in real physical space (veh/hr) to guarantee mass conservation.

### 3. Implicit Demand Virtual Routing Layer
When a network experiences disconnections, flow routing changes globally. Traditional GNNs require deep stacking to propagate these signals, leading to over-smoothing. We employ Multi-Head Self-Attention to dynamically establish **Implicit Virtual Links** among all nodes. These global virtual edges allow the network to instantly infer implicit OD relationships and transfer demand globally in a single hop.

---

## Model Structure

The architecture is built as a self-driven recurrent cell unrolled over $K$ steps.

```text
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1 & 2: Edge Alignment & Initial State Projection (k=0)    │
│   - O(1) Vectorized Hash-based Edge Alignment (G vs G')         │
│   - Projects old flows → f_scaled^(0) & initial shockwave ρ^(0) │
├─────────────────────────────────────────────────────────────────┤
│ Phase 3: Pseudo-Time Diffusion Loop (k = 1 ... K)               │
│   DiffusionCell (Neural Darcy's Law):                           │
│     - Dual-stream: GatedGCN (Local) + Self-Attention (Global)   │
│     → Predicts Δf_scaled^(k)                                    │
│   Physics Update:                                               │
│     - Flow: f^(k) = f^(k-1) + Δf^(k)                            │
│     - LWR:  ρ^(k) = ρ^(k-1) + scatter(Δf_real, in - out)        │
├─────────────────────────────────────────────────────────────────┤
│ Phase 4: Output & Physics-Informed Loss                         │
│   Terminal Equilibrium Flows: f_scaled^(K)                      │
│   Terminal Pressure (should → 0): ρ_v^(K)                       │
└─────────────────────────────────────────────────────────────────┘

```

### Physics-Informed Loss Function

The model is optimized using a two-part loss function:

$$ L_{total} = L_{Data} + \lambda_{eq} \cdot L_{Eq} $$

1. **Data-fitting Loss ($L_{Data}$)**: Standard regression loss in normalized space.

$$ L_{Data} = \text{MSE}(f_{scaled}^{(K)}, y_{scaled}) $$


2. **Terminal Equilibrium Constraint ($L_{Eq}$)**: A structurally grounded regularizer acting on the terminal pressure field. By forcing $\rho_v^{(K)}$ to zero, the predicted flows naturally satisfy the implicit OD demand without needing the true OD matrix.

$$ L_{Eq} = \text{MSE}\left(\frac{\rho_v^{(K)}}{\sigma}, 0\right) $$

*Note: Division by the flow standard deviation $\sigma$ normalizes the real-space pressure back to dimensionless $O(1)$ space, perfectly matching the gradient scale of $L_{Data}$ to prevent gradient explosion.*

---

## Results & Ablation Study

Our ST-PINN model achieves State-of-the-Art performance across multiple metrics. The ablation study explicitly demonstrates the necessity of the physics-informed mechanisms and the implicit virtual routing layer.

| Model / Ablation Variant | R2 (↑) | MSE (↓) | RMSE (↓) | Global WMAPE (%) (↓) | New Edges WMAPE (%) (↓) | 95% Worst-case WMAPE (%) (↓) | Test Time per Graph (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Ours (Physics-Informed Diffusion)** | **0.9038** | **0.0971** | **0.3116** | 24.80 | **40.36** | **37.67** | 2.78 |
| Ours (Previous: GNN+PINN) | 0.8854 | 0.1157 | 0.3401 |**24.29** | 41.05 | 39.54| 2.30 |
| Node-Centric GNN | 0.8803 | 0.1209 | 0.3477 | 24.83 | 43.40 | 41.66| **0.98** |
| w/o Capacity | 0.8716 | 0.1297 | 0.3601| 25.80 | 42.09 | 41.32 | 2.19 |
| w/o Virtual Links | 0.8661 | 0.1353 | 0.3678| 26.50 | 44.49 | 42.44| 2.67 |
| w/o Free Flow Time (FFT) | 0.7498 | 0.2527 | 0.5027 | 37.15 | 58.06 | 57.03 | 2.26 |

---

## Installation

```bash
# 1. Create and activate the environment
conda create -n st-pinn python=3.9 -y
conda activate st-pinn

# 2. Install PyTorch
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 3. Install PyTorch Geometric and scatter
pip install torch_geometric
pip install torch_scatter torch_sparse -f [https://data.pyg.org/whl/torch-2.0.0+cu118.html](https://data.pyg.org/whl/torch-2.0.0+cu118.html)

# 4. Install remaining dependencies
pip install -e .
pip install scikit-learn networkx tqdm wandb

```

## Usage

### 1. Generate Dataset

Generate the network pairs $(G, G')$ and solve the SUE for ground truth generation.

```bash
cd create_sioux_data
python solve_network_pairs.py
python build_network_pairs_dataset.py --output_dir processed_data/pyg_dataset
cd ..

```

### 2. Train the Model

Train the full ST-PINN model using the provided configuration.

```bash
python main.py --cfg configs/GatedGCN/network-pairs-topology.yaml

```

```

```

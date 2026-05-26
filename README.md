# Federated Learning

This repository documents an exploratory journey through **Federated Learning (FL)**, starting from a foundational algorithm implementation and progressively focusing on **federated unlearning** and related **reconstruction attacks**.

The work follows a research-driven path, each folder building on the previous one.



## Repository Structure

| Folder | Description |
|---|---|
| `FedAvg/` | Baseline implementation of Federated Averaging |
| `FUIA_Client_Unlearning/` | FUIA attack in a Client Unlearning scenario (MNIST) |
| `FUIA_Sample_Unlearning/` | FUIA attack in a Sample Unlearning scenario (MNIST) |
| `FUIA_VGG-16_CelebA/` | FUIA attack in a Sample Unlearning scenario (VGG-16 + CelebA) |
| `FUIA_PUF_Client_Unlearning/` | FUIA attack combined with the PUF unlearning algorithm |
| `FUIA_Realistic_Settings_Client_Unlearning/` | FUIA attack in a more realistic Client Unlearning scenario (MNIST) |



## Experimental Progression

### 1. `FedAvg` — The Baseline
Implementation of **Federated Averaging**, the foundational FL optimization algorithm, following the paper:

> *"Communication-Efficient Learning of Deep Networks from Decentralized Data"* — McMahan et al.

This serves as the starting point and reference baseline for all subsequent experiments.



### 2. `FUIA_Client_Unlearning` — First Attack Implementation
First implementation of the **FUIA (Federated Unlearning Inversion Attack)**, following the paper:

> *"Model Inversion Attack Against Federated Unlearning"* — Zhou et al., IEEE TIFS 2026

Scenario: **Client Unlearning** — an entire client is removed from the federation. Dataset: **MNIST**.



### 3. `FUIA_Sample_Unlearning` — Sample Unlearning
FUIA attack applied to a **Sample Unlearning** scenario, where only specific data samples (rather than a full client) are forgotten. Dataset: **MNIST**.



### 4. `FUIA_VGG-16_CelebA` — Scaling Up
The Sample Unlearning scenario is scaled up to a more complex setting: **VGG-16** architecture on the **CelebA** dataset, to evaluate the attack with a more complex and realistic dataset.



### 5. `FUIA_PUF_Client_Unlearning` — Combining Algorithms
Experimental fusion of the **PUF (Pseudo-gradients Updates for Federated Unlearning)** algorithm with the FUIA attack, following the paper:

> *"Federated Unlearning Made Practical: Seamless Integration via Negated Pseudo-Gradients"*

This explores whether PUF's unlearning mechanism affects the effectiveness of the inversion attack.



### 6. `FUIA_Realistic_Settings_Client_Unlearning` — Back to the Basics, Realistically
A return to the **Client Unlearning** scenario with MNIST, but in a more **realistic federation setup** — testing how the attack behaves under conditions closer to real-world deployments.



## Common Requirements

Each folder contains its own `README.md` with specific setup and execution instructions. In general, all experiments require:

- Python 3.8+
- PyTorch
- A virtual environment (recommended)

Some experiments optionally support **WandB** for experiment tracking. When prompted at runtime, enter `3` to skip WandB logging.



## References

- H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, *"Communication-Efficient Learning of Deep Networks from Decentralized Data"*, in Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), 2017, pp. 1273–1282.

- L. Zhou, Y. Zhu, and R. Liu, *"Model Inversion Attack Against Federated Unlearning"*, IEEE Transactions on Information Forensics and Security, vol. 21, pp. 2342–2357, 2026.

- A. Mora, C. Mazzocca, R. Montanari, and P. Bellavista, *"Federated Unlearning Made Practical: Seamless Integration via Negated Pseudo-Gradients"*, Journal of LaTeX Class Files, vol. 14, no. 8, 2021.



## Author

**Ottone Piazzi** — [github.com/ottonepiazzi](https://github.com/ottonepiazzi)

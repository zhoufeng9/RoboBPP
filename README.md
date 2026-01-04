# RoboBPP: Benchmarking Robotic Online Bin Packing with Physics-based Simulation

**RoboBPP** is a comprehensive benchmark for **Robotic Online 3D Bin Packing**, designed to bridge the gap between algorithmic packing performance and **physical feasibility in real robotic scenarios**.  
The benchmark integrates **real-world industrial datasets**, **standardized evaluation metrics**, and a **physics-based simulation environment** to support fair, reproducible, and realistic comparisons.

---

## 📄 Paper

**RoboBPP: Benchmarking Robotic Online Bin Packing with Physics-based Simulation**  
📌 *Under review at International Journal of Robotics Research (IJRR)*  

- arXiv: https://arxiv.org/abs/2512.04415

If you find this project useful, please consider citing our paper (see [Citation](#citation)).

---

## 🌐 Project Website

🔗 **Project Homepage:**  
https://robot-bin-packing-benchmark.github.io/

The website provides:
- Detailed benchmark descriptions  
- Dataset statistics and visualizations  
- Leaderboards across different settings  
- Documentation of evaluation metrics  
- Interactive demonstrations of the packing process  

---

## 📦 Datasets

RoboBPP includes **three real-world industrial datasets**, covering diverse packing characteristics and challenges:

- **Repetitive Dataset** – Manufacturer-oriented supply data  
- **Diverse Dataset** – Office supplies vendor data  
- **Wood Board Dataset** – Long and slender industrial materials  

📥 **All datasets can be downloaded from the project homepage:**  
https://robot-bin-packing-benchmark.github.io/download.html

---

## 🧩 Code Structure

The repository is organized into three main directories: `code`, `test`, and `experiment`, reflecting the full pipeline from algorithm implementation to evaluation and analysis.

### `code/`
This directory contains the **training and implementation code** for all packing algorithms reproduced in this benchmark.

- `heuristics/`  
  Implements classical **heuristic-based online bin packing methods**, reproduced by us according to their original papers.

- `learning_based/`  
  Contains **learning-based approaches**, including model definitions and training pipelines, implemented following the corresponding publications.

All algorithms in this folder are **our own re-implementations** to ensure consistency and fair comparison across methods.

---

### `test/`
This directory provides the **testing and evaluation code** for running all algorithms in the **physics-based simulation environment**.

- Unified testing interfaces for both heuristic and learning-based methods  
- Integration with the PyBullet-based simulator  
- Scripts for executing online packing under different benchmark settings  
- Includes **pre-trained models for selected learning-based methods** to facilitate reproducibility

---

### `experiment/`
Experiments validating the effectiveness and rationality of the proposed benchmark metrics

---

## 🚀 Getting Started

```bash
git clone https://github.com/your-repo/RoboBPP.git
cd RoboBPP
```

# RoboBPP: Benchmarking Robotic Online Bin Packing with Physics-based Simulation

📦🤖 **RoboBPP** is a comprehensive benchmark for **Robotic Online 3D Bin Packing**, designed to bridge the gap between algorithmic packing performance and **physical feasibility in real robotic scenarios**.  
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

## 📊 Benchmark Overview

RoboBPP focuses on **online bin packing under robotic constraints**, where items arrive sequentially and decisions must be made without future knowledge.

### Key Features
- **Online 3D Bin Packing** setting  
- **Physics-based validation** of packing results  
- **Industrial-scale real datasets**  
- **Unified evaluation protocol**  
- **Support for both heuristic and learning-based methods**

---

## 📦 Datasets

RoboBPP includes **three real-world industrial datasets**, covering diverse packing characteristics and challenges:

- **PG Dataset** – Manufacturer-oriented supply data  
- **DELI Dataset** – Consumer-oriented demand data  
- **OPAI Dataset** – Long and slender industrial materials  

📥 **All datasets can be downloaded from the project homepage:**  
https://robot-bin-packing-benchmark.github.io/

Each dataset includes:
- Sequential item arrival order  
- Exact 3D dimensions  
- Realistic distribution patterns observed in industry  

---

## 🧪 Evaluation Protocol

The benchmark evaluates methods from both **geometric efficiency** and **robotic executability** perspectives.

### Metrics include:
- Packing height / volume utilization  
- Packing success rate  
- Stability and physical feasibility  
- Online decision quality  

Multiple test settings are provided to ensure robustness and fairness.

---

## 🤖 Physics-based Simulation

To validate robotic feasibility, RoboBPP integrates a **physics-based simulation environment** built on **PyBullet**, enabling:

- Collision-aware placement  
- Gravity and stability checking  
- Visualization of the packing process  
- Optional robotic arm execution simulation  

This avoids overly optimistic results that are infeasible in real-world deployment.

---

## 🧩 Code Structure

All code required to **run experiments, reproduce results, and visualize packing processes** is fully open-sourced in this repository.

Typical components include:
- Dataset loaders  
- Packing algorithms (heuristic & learning-based)  
- Evaluation scripts  
- Physics simulation environment  
- Visualization and logging tools  

Detailed usage instructions can be found in the corresponding subdirectories.

---

## 🚀 Getting Started

```bash
git clone https://github.com/your-repo/RoboBPP.git
cd RoboBPP



learning based:
```bash
AR2L
PCT
PackE
CDRL
TapNet++
```

heuristics:
```bash
 HeightmapMin
 LSAH
 MACS
 RANDOM
 OnlineBPH
 DBL
 BR
 SDFPack
```

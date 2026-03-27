# 🛰️ DDM: Landslide Risk Predictor (MoE 5-Sensor Fusion)

## 📌 Project Overview
The **DDM-Landslide Predictor** is an AI-driven Early Warning System (EWS) that utilizes a **Mixture of Experts (MoE)** architecture to predict landslide probabilities. By fusing 5 distinct data streams, the model identifies high-risk zones with precision.

### 🔬 The Mixture of Experts (MoE) Architecture
Unlike standard models, our system uses **5 specialized neural networks (Experts)**:
1. **SAR Expert:** Processes Sentinel-1 (Radar) for ground displacement.
2. **Optical Expert:** Processes Sentinel-2 for vegetation/land cover changes.
3. **Terrain Expert (NEW):** Processes DEM (Digital Elevation Models) for slope/aspect.
4. **Hydrology Expert:** Processes Rainfall (.nc) data for trigger events.
5. **Saturation Expert:** Processes Soil Moisture for ground stability.

A **Gating Network** dynamically weights these experts based on the input, ensuring that in heavy rain, the Hydrology Expert is prioritized, while in clear weather, the Optical Expert takes the lead.

---

## 🚀 Getting Started

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```
### 2. running
```bash
streamlit run app.py
```

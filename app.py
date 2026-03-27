import streamlit as st
import torch
import torch.nn as nn
import os
import rasterio
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import tempfile
import datetime

# ==========================================
# 1. MODEL ARCHITECTURE (UNCHANGED)
# ==========================================
class DomainExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, x):
        return self.net(x)

class LandslideMoE(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.hidden_dim = 64
        self.expert_s1 = DomainExpert(input_dim, self.hidden_dim)
        self.expert_s2 = DomainExpert(input_dim, self.hidden_dim)
        self.expert_rain = DomainExpert(input_dim, self.hidden_dim)
        self.expert_soil = DomainExpert(input_dim, self.hidden_dim)
        
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, s1, s2, rain, soil):
        expert_outputs = torch.stack([
            self.expert_s1(s1), self.expert_s2(s2), 
            self.expert_rain(rain), self.expert_soil(soil)
        ], dim=1)
        gate_weights = self.gating_network(torch.cat([s1, s2, rain, soil], dim=1)).unsqueeze(2)
        return self.classifier(torch.sum(expert_outputs * gate_weights, dim=1))

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_first_file(folder, ext):
    if not os.path.exists(folder): return None
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(ext):
                return os.path.join(root, f)
    return None

def generate_replication_adaptation_report(location, risk_prob, model_version="v2.0-5Expert"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Adaptation Logic
    if risk_prob > 0.65:
        level, action = "URGENT", "Immediate evacuation of Tier-1 zones and deployment of physical debris barriers."
    else:
        level, action = "MONITORING", "Reforestation of slopes using deep-rooted vegetation and routine drainage clearing."

    report = f"""
============================================================
REPLICATION & ADAPTATION STRATEGY REPORT
============================================================
Target Location : {location}
Model Engine    : {model_version} Mixture of Experts
Date Generated  : {timestamp}

------------------------------------------------------------
PART A: REPLICATION PROTOCOL (Technical Setup)
------------------------------------------------------------
To replicate these results in a new geographic region:
1. Data Acquisition: 
   - Sentinel-1 (GRD, VV+VH) & Sentinel-2 (L2A) via Copernicus.
   - SRTM 30m Digital Elevation Model (DEM) for slope analysis.
2. Pre-processing:
   - Normalize all rasters to a range.
   - Flatten to 1024-dimensional feature vectors.
3. Model Loading:
   - Initialize 5-Expert MoE Architecture.
   - Load 'landslide_moE_v2_5expert.pth' weights.

------------------------------------------------------------
PART B: ADAPTATION MEASURES (Action Plan)
------------------------------------------------------------
Risk Probability : {risk_prob*100:.2f}%
Adaptation Tier  : {level}

Recommended Strategic Actions:
- [Physical] {action}
- [Systemic] Update local GIS zoning maps to reflect new {risk_prob*100:.0f}% risk threshold.
- [Sensor] Install ground-based tiltmeters to validate MoE displacement predictions.

============================================================
END OF REPORT - PROJECT DDM-LANDSLIDE
============================================================
"""
    return report

def generate_detailed_report(location_name, risk_prob, data_sources):
    """Generates a detailed markdown/text report for the analysis."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Determine risk category
    if risk_prob >= 0.70:
        risk_level = "CRITICAL / HIGH RISK"
        recommendations = "- IMMEDIATE ACTION REQUIRED: Consider issuing evacuation warnings.\n- Deploy ground response teams and field sensors.\n- Alert local disaster management authorities."
    elif risk_prob >= 0.40:
        risk_level = "ELEVATED / MEDIUM RISK"
        recommendations = "- Heightened monitoring recommended.\n- Review structural integrity of local infrastructure.\n- Prepare standby emergency response protocols."
    else:
        risk_level = "LOW RISK"
        recommendations = "- Routine monitoring advised.\n- No immediate structural or geological threat detected from current telemetry."

    # Build the report string
    report = f"""====================================================
TERRAIN & LANDSLIDE RISK ASSESSMENT REPORT
====================================================
Date & Time Generated : {timestamp}
Analysis Target       : {location_name}

----------------------------------------------------
1. RISK ANALYSIS SUMMARY
----------------------------------------------------
Calculated Risk Probability : {risk_prob * 100:.2f}%
Risk Classification         : {risk_level}

----------------------------------------------------
2. SATELLITE & TELEMETRY DATA UTILIZED
----------------------------------------------------
"""
    for source, status in data_sources.items():
        report += f"- {source:<18}: {status}\n"

    report += f"""
----------------------------------------------------
3. MODEL INFERENCE DETAILS
----------------------------------------------------
Architecture   : Domain-Decoupled Mixture of Experts (MoE)
Tensor Profile : 1024-dimensional flattened geospatial vectors
Active Experts : 
  - Sentinel-1 (SAR / Terrain Displacement)
  - Sentinel-2 (Multispectral Optical)
  - Soil Moisture Saturation
  - Rainfall / Climate NetCDF

----------------------------------------------------
4. ACTIONABLE RECOMMENDATIONS
----------------------------------------------------
{recommendations}

====================================================
Generated by MoE Landslide Predictor AI
====================================================
"""
    return report

def process_array(data_array, fixed_length=1024):
    # Force conversion to numpy array first
    data_array = np.asarray(data_array) 
    flat_data = data_array.astype(np.float32).flatten()
    # ... (rest of your existing process_array code)
    
    # 1. Filter out the zero-padding/black borders
    valid_data = flat_data[flat_data != 0] 
    if len(valid_data) == 0:
        valid_data = np.zeros(fixed_length) # Fallback if image is empty
        
    # 2. NORMALIZATION (The Key Fix)
    # Scale data to range [1] so the model doesn't saturate
    v_min, v_max = valid_data.min(), valid_data.max()
    if v_max > v_min:
        valid_data = (valid_data - v_min) / (v_max - v_min)
    
    # 3. Select 1024 pixels from the center of the terrain
    fixed_size_array = np.zeros(fixed_length)
    length = min(len(valid_data), fixed_length)
    mid = len(valid_data) // 2
    start = max(0, mid - (length // 2))
    end = start + length
    
    fixed_size_array[:end-start] = valid_data[start:end]
    return torch.tensor(np.nan_to_num(fixed_size_array), dtype=torch.float32).unsqueeze(0)
def plot_satellite(path1, path2=None, title1="Optical", title2="Soil"):
    """Renders satellite imagery with explicit axis handling to prevent array errors."""
    
    if path2:
        # Explicitly unpack the two axes into ax1 and ax2
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        with rasterio.open(path1) as src1:
            ax1.imshow(src1.read(1), cmap='viridis')
            ax1.set_title(title1)
            ax1.axis('off')
            
        with rasterio.open(path2) as src2:
            ax2.imshow(src2.read(1), cmap='Blues')
            ax2.set_title(title2)
            ax2.axis('off')
            
    else:
        # Create a single axis and unpack it into ax1
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
        
        with rasterio.open(path1) as src:
            ax1.imshow(src.read(1), cmap='viridis')
            ax1.set_title(title1)
            ax1.axis('off')
            
    st.pyplot(fig)

def load_geodata(folder_path):
    # Find files
    s1_p = get_first_file(os.path.join(folder_path, "Sentinel-1"), '.tif') or get_first_file(folder_path, '.tif')
    s2_p = get_first_file(os.path.join(folder_path, "Sentinel-2"), '.tif') or get_first_file(folder_path, '.tif')
    rain_p = get_first_file(folder_path, '.nc')
    soil_p = get_first_file(os.path.join(folder_path, "Soil_moisture"), '.tif') or get_first_file(folder_path, '.tif')

    # Process to tensors (using np.array to avoid attribute errors)
    s1_t = process_array(rasterio.open(s1_p).read()) if s1_p else torch.zeros(1, 1024)
    s2_t = process_array(rasterio.open(s2_p).read()) if s2_p else torch.zeros(1, 1024)
    soil_t = process_array(rasterio.open(soil_p).read()) if soil_p else torch.zeros(1, 1024)
    
    rain_t = torch.zeros(1, 1024)
    if rain_p:
        try:
            ds = xr.open_dataset(rain_p)
            var_name = list(ds.data_vars)
            # Convert xarray to numpy explicitly to fix .flatten() issues
            rain_t = process_array(np.array(ds[var_name]))
        except: pass

    # Dictionary keys MUST match the UI calls (s2 and soil)
    return s1_t, s2_t, rain_t, soil_t, {"s2": s2_p, "soil": soil_p}

# ==========================================
# 3. STREAMLIT UI (NO PLOTTING)
# ==========================================
st.set_page_config(page_title="MoE Landslide Predictor", layout="wide")
st.title("🏔️ Domain-Decoupled MoE Landslide Predictor")

@st.cache_resource
def load_model():
    model = LandslideMoE(input_dim=1024)
    model.load_state_dict(torch.load("landslide_moE_final.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
# Create the selection menu in the sidebar
# Create the selection menu in the sidebar
analysis_mode = st.sidebar.selectbox(
    "Select Analysis View:",
    [
        "Wayanad T1 (Past)", 
        "Wayanad T2 (Present)", 
        "Wayanad Comparison (T1 vs T2)", 
        "Puthumala Event (2019)",
        "Custom Data Upload"   # <--- ADDED THIS LINE
    ]
)

# Shared dictionary for the pre-loaded local datasets
local_data_used = {
    "Sentinel-1 (SAR)": "Local Pre-loaded Dataset",
    "Sentinel-2 (Opt)": "Local Pre-loaded Dataset",
    "Soil Moisture": "Local Pre-loaded Dataset",
    "Rainfall Data": "Local Pre-loaded Dataset (or Synthesized)"
}

if st.button("Analyze", type="primary"):
    rep_adapt_report = "Analysis not yet complete."
    pred = 0.0
    
    with st.spinner("Processing 5-Sensor Fusion..."):
        if analysis_mode == "Wayanad T1 (Past)":
            s1, s2, r, soil, paths = load_geodata("./Wayanad_T1")
            pred = model(s1, s2, r, soil).item()
            st.metric("Landslide Risk (T1)", f"{pred*100:.2f}%")
            plot_satellite(paths['s2'], paths['soil'], "Optical (T1)", "Soil Moisture (T1)")
            
            # Display Report in Browser
            detailed_report = generate_detailed_report("Wayanad T1 (Past)", pred, local_data_used)
            with st.expander("📄 View Detailed Analysis Report", expanded=True):
                st.text(detailed_report)

        # Scenario 2: Wayanad T2
        elif analysis_mode == "Wayanad T2 (Present)":
            s1, s2, r, soil, paths = load_geodata("./Wayanad_T2")
            pred = model(s1, s2, r, soil).item()
            st.metric("Landslide Risk (T2)", f"{pred*100:.2f}%")
            plot_satellite(paths['s2'], paths['soil'], "Optical (T2)", "Soil Moisture (T2)")
            
            # Display Report in Browser
            detailed_report = generate_detailed_report("Wayanad T2 (Present)", pred, local_data_used)
            with st.expander("📄 View Detailed Analysis Report", expanded=True):
                st.text(detailed_report)

        # Scenario 3: Comparison (T1 vs T2)
        elif analysis_mode == "Wayanad Comparison (T1 vs T2)":
            s1_1, s2_1, r_1, soil_1, p1 = load_geodata("./Wayanad_T1")
            s1_2, s2_2, r_2, soil_2, p2 = load_geodata("./Wayanad_T2")
            
            p_t1 = model(s1_1, s2_1, r_1, soil_1).item()
            p_t2 = model(s1_2, s2_2, r_2, soil_2).item()
            
            col1, col2 = st.columns(2)
            col1.metric("Past Risk (T1)", f"{p_t1*100:.2f}%")
            col2.metric("Current Risk (T2)", f"{p_t2*100:.2f}%", delta=f"{(p_t2-p_t1)*100:.2f}%")
            
            st.write("### Temporal Comparison (Optical Data)")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            with rasterio.open(p1['s2']) as s: 
                ax1.imshow(s.read(1), cmap='viridis')
                ax1.set_title("T1 (Past)")
                ax1.axis('off')
                
            with rasterio.open(p2['s2']) as s: 
                ax2.imshow(s.read(1), cmap='viridis')
                ax2.set_title("T2 (Present)")
                ax2.axis('off')
                
            st.pyplot(fig)
            
            # Display Report in Browser (Basing it on the current T2 risk)
            detailed_report = generate_detailed_report("Wayanad Comparison (Current Threat Level)", p_t2, local_data_used)
            with st.expander("📄 View Detailed Analysis Report", expanded=True):
                st.text(detailed_report)

        # Scenario 4: Puthumala
        elif analysis_mode == "Puthumala Event (2019)":
            s1, s2, r, soil, paths = load_geodata("./Puthumala_2019")
            pred = model(s1, s2, r, soil).item()
            st.subheader("Event Reconstruction: Puthumala 2019")
            st.metric("Disaster Probability", f"{pred*100:.2f}%")
            plot_satellite(paths['s2'], paths['soil'], "Optical View", "Soil Saturation")
            
            # Display Report in Browser
            detailed_report = generate_detailed_report("Puthumala Event (2019)", pred, local_data_used)
            with st.expander("📄 View Detailed Analysis Report", expanded=True):
                st.text(detailed_report)

        # ==========================================
        # NEW SCENARIO 5: CUSTOM UPLOAD
        # ==========================================
        elif analysis_mode == "Custom Data Upload":
            st.subheader("📤 Upload Custom Satellite Data")
            
            st.info("""
            **Required Data Format:**
            * **Sentinel-1 (SAR):** Single `.tif` format
            * **Sentinel-2 (Optical):** Multiple `.tif` band files (Upload all together)
            * **Soil Moisture:** Multiple `.tif` files (Upload all together)
            * **Rainfall (Optional):** `.nc` format (NetCDF climate data)
            """)
            
            with st.form("upload_form"):
                s1_file = st.file_uploader("Upload Sentinel-1 (.tif)", type=["tif", "tiff"])
                s2_files = st.file_uploader("Upload Sentinel-2 bands (.tif)", type=["tif", "tiff"], accept_multiple_files=True)
                soil_files = st.file_uploader("Upload Soil Moisture (.tif)", type=["tif", "tiff"], accept_multiple_files=True)
                rain_file = st.file_uploader("Upload Rainfall (.nc)", type=["nc"])
                
                submitted = st.form_submit_button("⚙️ Predict Landslide Risk", type="primary")
            
            if submitted:
                if not (s1_file and len(s2_files) > 0 and len(soil_files) > 0):
                    st.warning("⚠️ Please upload at least Sentinel-1, Sentinel-2 (at least one band), and Soil Moisture (at least one file) to proceed.")
                else:
                    with st.spinner("Processing custom data through MoE..."):
                        with tempfile.TemporaryDirectory() as temp_dir:
                            def save_file(uploaded_file, filename):
                                path = os.path.join(temp_dir, filename)
                                with open(path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                return path
                            
                            # Save files to temp directory
                            s1_path = save_file(s1_file, "s1.tif")
                            s2_path = save_file(s2_files, "s2_band.tif") 
                            soil_path = save_file(soil_files, "soil.tif")
                            
                            # Process tensors
                            s1_t = process_array(rasterio.open(s1_path).read())
                            s2_t = process_array(rasterio.open(s2_path).read())
                            soil_t = process_array(rasterio.open(soil_path).read())
                            
                            if rain_file is not None:
                                rain_path = save_file(rain_file, "rain.nc")
                                ds = xr.open_dataset(rain_path)
                                var_name = list(ds.data_vars)
                                rain_t = process_array(np.array(ds[var_name]))
                            else:
                                rain_t = torch.zeros(1, 1024)
                            
                            # Run Model Prediction
                            with torch.no_grad():
                                pred = model(s1_t, s2_t, rain_t, soil_t).item()
                            
                            # Display Results
                            st.divider()
                            st.metric("Custom Location Risk Probability", f"{pred*100:.2f}%")
                            if pred > 0.5:
                                st.error("⚠️ **HIGH RISK DETECTED FOR UPLOADED REGION**")
                            else:
                                st.success("✅ **LOW RISK DETECTED**")
                                
                            st.write("### Input Visualizations")
                            plot_satellite(s2_path, soil_path, "Uploaded Optical (Band)", "Uploaded Soil Moisture")
                            
                            # Display Report in Browser
                            custom_data_used = {
                                "Sentinel-1 (SAR)": s1_file.name if s1_file else "Not Provided",
                                "Sentinel-2 (Opt)": s2_files.name if s2_files else "Not Provided",
                                "Soil Moisture": soil_files.name if soil_files else "Not Provided",
                                "Rainfall Data": rain_file.name if rain_file else "Synthesized (Zero Tensor)"
                            }
                            
                            detailed_report = generate_detailed_report("Custom User Upload", pred, custom_data_used)
                            with st.expander("📄 View Detailed Analysis Report", expanded=True):
                                st.text(detailed_report)                        
                        
        # After your prediction logic:
        rep_adapt_report = generate_replication_adaptation_report(analysis_mode, pred)

    # 3. DISPLAY & SUBMIT (Separately)
    st.divider()
    st.subheader("📄 Replication & Adaptation Brief")
    
    # Show it in the browser
    st.text_area("Report Preview", rep_adapt_report, height=300)
    
    # PROVIDE THE SEPARATE FILE FOR SUBMISSION
    st.download_button(
        label="Download Report",
        data=rep_adapt_report,
        file_name=f"Replication_Adaptation_{analysis_mode}.txt",
        mime="text/plain",
        type="primary"
    )
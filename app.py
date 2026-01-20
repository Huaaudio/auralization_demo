import streamlit as st
import numpy as np
import scipy.signal as signal
import plotly.graph_objs as go
import soundfile as sf
import io
import tempfile, os

# --- INTEGRATION: Import your metrics class ---
# Ensure test_metrics.py is in the same directory
try:
    from test_metrics import SQmetrics
    from solve_plate_FE_STL_Diffuse_ERA import main as fe_solver
    from mosqito.utils import load
    METRICS_AVAILABLE = True
except ImportError as e:
    METRICS_AVAILABLE = False
    st.error(f"Could not import 'SQmetrics': {e}. Ensure project root is on PYTHONPATH and 'mosqito' is installed.")

# --- CONFIGURATION & CONSTANTS ---
st.set_page_config(page_title="Metamaterial Auralization", layout="wide")
SAMPLE_RATE = 48000
RHO_0 = 1.225       # Air density [kg/m^3]
C_0 = 343.0         # Speed of sound [m/s]
Z_0 = RHO_0 * C_0   # Impedance of air

# --- UPDATED DEFAULTS (Matched to setup_panel.py) ---
# Aluminum Properties
DEFAULT_RHO = 2700.0  # 
DEFAULT_H = 2.0       #  (2mm)
DEFAULT_E = 70e9      # 
DEFAULT_NU = 0.3      #

# --- 1. PHYSICS MODELS (Exact implementation of Eq 1-10) ---

def calculate_single_wall_tau(f, rho, h):
    """Eq (1): Sound transmission coefficient for single panel."""
    omega = 2 * np.pi * f
    omega = np.maximum(omega, 1e-9) 
    numerator = 2 * Z_0 * omega
    denominator = 1j * 2 * Z_0 * omega - (rho * h * omega**2)
    return numerator / denominator

def calculate_double_wall_tau(f, rho1, h1, rho2, h2, s, eta=0.0):
    """Eq (3): Sound transmission coefficient for double panel."""
    omega = 2 * np.pi * f
    omega = np.maximum(omega, 1e-9)
    s_complex = s * (1 + 1j * eta)
    term1 = -(rho1 * h1 * omega**2) + s_complex + 1j * omega * Z_0
    term2 = -(rho2 * h2 * omega**2) + s_complex + 1j * omega * Z_0
    numerator = 2 * Z_0 * omega * s_complex
    denominator = (term1 * term2) - (s_complex**2)
    return numerator / denominator

def calculate_metamaterial_rho_eq(f, rho_host, h_host, S_unit, m_res, f_res, c_res, n_res=1):
    """Eq (5) & (6): Equivalent mass density for metamaterial partition."""
    omega = 2 * np.pi * f
    omega_res = 2 * np.pi * f_res
    k_res = m_res * (omega_res**2) # Eq (8)
    denom = k_res + 1j * omega * c_res - (omega**2 * m_res)
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    num = m_res * (k_res + 1j * omega * c_res)
    term_res = (1 / (S_unit * h_host)) * (n_res * (num / denom))
    return rho_host + term_res

def get_transmission_loss(tau):
    """Eq (2): Convert transmission coefficient to STL (dB)."""
    tau_mag = np.abs(tau)
    tau_mag = np.maximum(tau_mag, 1e-12)
    return -20 * np.log10(tau_mag)

# --- 2. SIGNAL PROCESSING ---

def generate_noise(type, duration, sr=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration))
    if type == "Pink Noise":
        uneven = np.random.randn(len(t))
        X = np.fft.rfft(uneven)
        X = X / np.sqrt(np.arange(len(X)) + 1.0)
        return np.fft.irfft(X), t
    else: # Traffic / White Noise proxy
        return np.random.normal(0, 0.5, len(t)), t

def load_audio_file(uploaded_file, target_sr=SAMPLE_RATE):
    """Load WAV, downmix to mono, and resample to target_sr if needed."""
    if isinstance(uploaded_file, (str, bytes, os.PathLike)):
        audio, sr_in = load(uploaded_file)
    else:
        # Streamlit UploadedFile -> temp wav path for mosqito.utils.load
        uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            audio, sr_in = load(tmp_path)
        finally:
            os.remove(tmp_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr_in != target_sr:
        num_samples = int(len(audio) * target_sr / sr_in)
        audio = signal.resample(audio, num_samples)
        sr_in = target_sr
    return audio.astype(np.float32), sr_in

def apply_transfer_function(audio_signal, tau_complex, sim_freqs, sr=SAMPLE_RATE):
    """Eq (11): Filters input audio using transmission coefficient with energy preservation."""
    n = len(audio_signal)
    spectrum = np.fft.rfft(audio_signal)
    freqs = np.fft.rfftfreq(n, d=1/sr)

    # Interpolate complex Tau to match audio FFT frequencies
    tau_real = np.interp(freqs, sim_freqs, np.real(tau_complex))
    tau_imag = np.interp(freqs, sim_freqs, np.imag(tau_complex))
    tau_interp = tau_real + 1j * tau_imag

    # Eq (11): Frequency Domain Multiplication
    filtered_spectrum = spectrum * tau_interp
    filtered_signal = np.fft.irfft(filtered_spectrum)

    # Energy preservation factor (RMS match)
    num = np.sum(np.square(audio_signal, dtype=np.float64))
    den = np.sum(np.square(filtered_signal, dtype=np.float64))
    if den > 1e-18:
        epf = np.sqrt(num / den)
        filtered_signal *= epf

    return filtered_signal.astype(np.float32)

# --- 3. UI LAYOUT ---

def main():
    st.title("🎧 Acoustic Metamaterial Auralization Workbench")
    st.markdown("Comparative analysis of analytical and numerical models with psychoacoustic metrics.")

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("1. Input Excitation")
        noise_type = st.selectbox("Signal Source", ["Traffic Noise (White)", "Pink Noise", "Upload WAV"])
        duration = st.slider("Duration (s)", 1.0, 5.0, 2.0)
        uploaded_file = None
        if noise_type == "Upload WAV":
            uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
        
        st.divider()
        st.header("2. Partition Configuration")
        model_mode = st.radio("Model Type", ["Single Partition", "Double Partition"])
        
        st.subheader("Host Panel Parameters")
        rho_1 = st.number_input(r"Density $\rho_1$ (kg/m³)", value=DEFAULT_RHO) 
        h_1 = st.number_input(r"Thickness $h_1$ (mm)", value=DEFAULT_H) / 1000.0
        
        rho_2, h_2, s_stiffness = None, None, None
        if model_mode == "Double Partition":
            st.markdown("**Panel 2 & Core**")
            rho_2 = st.number_input(r"Density $\rho_2$ (kg/m³)", value=DEFAULT_RHO)
            h_2 = st.number_input(r"Thickness $h_2$ (mm)", value=DEFAULT_H) / 1000.0
            s_stiffness = st.number_input("Interlayer Stiffness $s$ (Pa/m)", value=75000.0) 
            
        st.divider()
        st.header("3. Metamaterial Design")
        enable_meta = st.checkbox("Enable Local Resonators", value=True)
        
        m_ratio = 0.0
        f_res = 1000.0
        c_res = 0.005
        
        if enable_meta:
            f_res = st.slider("Resonance Frequency $f_{res}$ (Hz)", 100, 2000, 950)
            m_ratio = st.slider("Mass Ratio (%)", 0, 50, 40) / 100.0
            c_res = st.slider("Damping $c_{res}$ (Ns/m)", 0.001, 1.0, 0.005)
            S_cell = 0.04 * 0.038 
            m_panel_mass = rho_1 * h_1 * S_cell
            m_res = m_panel_mass * m_ratio

    # --- SIMULATION & CALCULATION ---
    freqs = np.arange(10, 4000, 1)
    
    # Calculate Models
    tau_meta = None
    if model_mode == "Single Partition":
        tau_analytical = calculate_single_wall_tau(freqs, rho_1, h_1)
        if enable_meta:
            rho_eq = calculate_metamaterial_rho_eq(freqs, rho_1, h_1, S_cell, m_res, f_res, c_res)
            tau_meta = calculate_single_wall_tau(freqs, rho_eq, h_1)
    else: # Double Partition
        tau_analytical = calculate_double_wall_tau(freqs, rho_1, h_1, rho_2, h_2, s_stiffness)
        if enable_meta:
            rho_eq = calculate_metamaterial_rho_eq(freqs, rho_1, h_1, S_cell, m_res, f_res, c_res)
            tau_meta = calculate_double_wall_tau(freqs, rho_eq, h_1, rho_2, h_2, s_stiffness)

    current_tau = tau_meta if enable_meta else tau_analytical
    tl_analytical = get_transmission_loss(tau_analytical)
    tl_meta = get_transmission_loss(tau_meta) if enable_meta and tau_meta is not None else None

    # 2. Numerical Calculation (Triggered)
    tl_meta_numerical = None
    
    # UI Container for Numerical Ops
    st.sidebar.divider()
    st.sidebar.header("4. Numerical Verification")
    
    # Check if .mat file exists
    mat_file_path = "TestPlateForJiahua.mat" # Ensure this is in folder
    if not os.path.exists(mat_file_path):
        st.sidebar.warning("Numerical features disabled due to limitations of online demo environment.")
    else:
        if st.sidebar.button("▶ Run Numerical STL Simulation"):
            with st.spinner("Running Finite Element Simulation..."):
                # Store to session state to avoid rerun
                fe_result = fe_solver(freqs, f_res, m_ratio, c_res, mat_file_path, fig_flag=False)
                st.session_state['numerical_result'] = {
                    'freqs': fe_result[0],
                    'tl_meta_numerical': fe_result[1],
                    'tau_meta_numerical': fe_result[2]
                }
            st.sidebar.success("Numerical simulation completed.")

    # Retrieve numerical results if available
    if 'numerical_result' in st.session_state:
        num_res = st.session_state['numerical_result']
        freqs_num = num_res['freqs']
        tl_meta_numerical = num_res['tl_meta_numerical']
        tau_meta_numerical = num_res['tau_meta_numerical']

    # --- DASHBOARD UI ---
    col_viz, col_metrics = st.columns([2, 1])
    
    with col_viz:
        st.subheader("Transmission Loss Analysis")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freqs, y=tl_analytical, name='Without Metamaterial', line=dict(color='blue', width=3)))
        if enable_meta and tl_meta is not None:
            fig.add_trace(go.Scatter(x=freqs, y=tl_meta, name='With Metamaterial (Analytical)', line=dict(color='orange', width=3)))
            fig.add_vrect(x0=f_res*0.9, x1=f_res*1.1, fillcolor="yellow", opacity=0.1, annotation_text="Resonance")
        # Numerical Result
        if tl_meta_numerical is not None:
            fig.add_trace(go.Scatter(x=freqs_num, y=tl_meta_numerical, name='With Metamaterial (Numerical)', line=dict(color='green', width=3, dash='dash')))
        fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="STL (dB)", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # --- AURALIZATION CONTROL ---
        st.subheader("🔊 Auralization & Metrics")
        
        if st.button("▶ Run Auralization & Compute Metrics"):
            if not METRICS_AVAILABLE:
                st.error("Cannot run: test_metrics.py not found.")
            else:
                with st.spinner("Processing Audio & Calculating Psychoacoustics..."):
                    # 1. Source audio: noise or uploaded WAV
                    if noise_type == "Upload WAV":
                        if uploaded_file is None:
                            st.warning("Please upload a WAV file to run auralization.")
                            st.stop()
                        raw_audio, sr = load_audio_file(uploaded_file, target_sr=SAMPLE_RATE)
                    else:
                        raw_audio, _ = generate_noise(noise_type, duration, sr=SAMPLE_RATE)
                        sr = SAMPLE_RATE
                    filtered_audio = apply_transfer_function(raw_audio, current_tau, freqs, sr=sr)
                    # Add filtered audio without metamaterial for comparison
                    filtered_audio_no_meta = apply_transfer_function(raw_audio, tau_analytical, freqs, sr=sr)
                    filtered_audio_numerical = apply_transfer_function(raw_audio, tau_meta_numerical, freqs_num, sr=sr) if tl_meta_numerical is not None else filtered_audio
                    # 2. Normalize for playback/analysis
                    st.session_state['audio_bare_panel'] = filtered_audio_no_meta
                    st.session_state['audio_result'] = filtered_audio
                    st.session_state['audio_raw_norm'] = raw_audio
                    st.session_state['audio_numerical'] = filtered_audio_numerical
                    
                    # 3. Compute Metrics using SQmetrics
                    sq = SQmetrics()
                    metrics_orig = sq.compute_metrics(raw_audio, sr)
                    metrics_meta = sq.compute_metrics(filtered_audio, sr)
                    metrics_meta_numerical = sq.compute_metrics(filtered_audio_numerical, sr) if tl_meta_numerical is not None else None
                    st.session_state['metrics_orig'] = metrics_orig
                    st.session_state['metrics_meta'] = metrics_meta
                    st.session_state['metrics_numerical'] = metrics_meta_numerical if metrics_meta_numerical is not None else None
                    st.session_state['audio_sr'] = sr
                    st.session_state['has_run'] = True

        # Display Audio Player if run
        if st.session_state.get('has_run'):
            col_raw_audio, col_bare_panel_audio, col_meta_audio, col_num_audio = st.columns(4)
            with col_raw_audio:
                st.caption("Original Audio")
                st.audio(st.session_state['audio_raw_norm'], sample_rate=st.session_state.get('audio_sr', SAMPLE_RATE), format='audio/wav')
            with col_bare_panel_audio:
                st.caption("Auralized (Without Metamaterial)")
                st.audio(st.session_state['audio_bare_panel'], sample_rate=st.session_state.get('audio_sr', SAMPLE_RATE), format='audio/wav')
            with col_meta_audio:
                st.caption("Auralized (Metamaterial, Analytical)")
                st.audio(st.session_state['audio_result'], sample_rate=st.session_state.get('audio_sr', SAMPLE_RATE), format='audio/wav')
            with col_num_audio:
                st.caption("Auralized (Metamaterial, Numerical)")
                st.audio(st.session_state['audio_numerical'], sample_rate=st.session_state.get('audio_sr', SAMPLE_RATE), format='audio/wav')

    # --- METRICS DISPLAY ---
    with col_metrics:
        st.subheader("Psychoacoustic Metrics")
        
        if st.session_state.get('has_run'):
            m_orig = st.session_state['metrics_orig']
            m_meta = st.session_state['metrics_meta']
            m_num = st.session_state['metrics_numerical'] if 'metrics_numerical' in st.session_state else None

            def display_metric(label, key, jnd_val=None):
                val_orig = float(m_orig[key])
                val_meta = float(m_meta[key])
                if m_num is not None:
                    val_meta_numerical = float(m_num[key])
                delta = val_meta - val_orig if m_num is None else val_meta_numerical - val_meta
                
                # Determine JND status if provided
                status_color = "black"
                status_text = ""
                if jnd_val:
                    if abs(delta) > jnd_val:
                        status_text = "Perceptible Change (>JND)"
                        status_color = "red" if delta > 0 else "green" # Green if reduced (usually good for noise)
                    else:
                        status_text = "Imperceptible Change (<JND)"
                        status_color = "gray"

                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                    <h5 style="margin:0">{label}</h5>
                    <div style="display:flex; justify-content:space-between; margin-top:10px;">
                        <div><small>Original</small><br><strong>{val_orig:.2f}</strong></div>
                        <div><small>Metamaterial</small><br><strong>{val_meta:.2f}</strong></div>
                        {f'<div><small>Numerical</small><br><strong>{val_meta_numerical:.2f}</strong></div>' if m_num is not None else 'N/A'}
                    </div>
                    <hr style="margin:5px 0">
                    <div style="color:{status_color}; font-size:0.9em;">
                        Δ {delta:+.2f} <br> {status_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Display Cards using keys from your SQmetrics class
            display_metric("Loudness (Zwicker)", 'loudness_zwst', jnd_val=1 if m_meta is None else float(m_meta['loudness_zwst'])*0.1)
            display_metric("Sharpness (DIN)", 'sharpness_din_st', jnd_val=0.3)
            display_metric("Roughness", 'roughness_ecma', jnd_val=0.1)
            display_metric("Tonality (TNR)", 'tnr_ecma_st', jnd_val=0.1)
            
        else:
            st.info("Click 'Run Auralization' to calculate metrics.")

if __name__ == "__main__":
    main()
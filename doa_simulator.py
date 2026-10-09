"""
Depth of Anaesthesia (DoA) teaching simulator — prototype
Educational use only. Not a clinical monitor. BIS is illustrative, not the proprietary BIS algorithm.
Run: streamlit run doa_simulator.py
"""
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="DoA Teaching Simulator", layout="wide")
st.title("Depth of Anaesthesia — Teaching Simulator")
st.caption("SIMULATION ONLY • Synthetic EEG and illustrative indices • Not for clinical decision-making")

if "t" not in st.session_state:
    st.session_state.t = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "drug_events" not in st.session_state:
    st.session_state.drug_events = []
if "state" not in st.session_state:
    st.session_state.state = {"propofol": 0.0, "dexmedetomidine": 0.0,
                              "ketamine": 0.0, "sevoflurane": 0.0, "isoflurane": 0.0}
if "noise" not in st.session_state:
    st.session_state.noise = 0.05

DRUGS = {
    "Propofol": ("propofol", "Illustrative effect-site level (0–10)"),
    "Dexmedetomidine": ("dexmedetomidine", "Illustrative effect-site level (0–10)"),
    "Ketamine": ("ketamine", "Illustrative effect-site level (0–10)"),
    "Sevoflurane": ("sevoflurane", "Illustrative effect-site level (0–10)"),
    "Isoflurane": ("isoflurane", "Illustrative effect-site level (0–10)"),
}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def current_metrics(state, noise):
    p = state["propofol"]
    d = state["dexmedetomidine"]
    k = state["ketamine"]
    s = state["sevoflurane"]
    i = state["isoflurane"]
    volatile = 0.5 * (s + i)
    hypnotic = p + 0.75*d + 0.7*volatile
    # Teaching-only heuristics. Ketamine may increase fast activity and make BIS less predictable.
    bis = 96 - 8.0*hypnotic - 1.4*max(hypnotic-5, 0)**2 + 1.8*k
    bis += np.random.normal(0, 1.3 + noise*8)
    bis = float(np.clip(bis, 5, 98))
    # Synthetic spectral descriptors (Hz), deliberately approximate.
    sef95 = float(np.clip(28 - 1.5*hypnotic + 0.55*k - 0.3*d + np.random.normal(0, 0.6), 1, 30))
    mf = float(np.clip(14 - 0.85*hypnotic + 0.35*k - 0.2*d + np.random.normal(0, 0.4), 0.5, 25))
    sqi = int(np.clip(98 - noise*85 - np.random.uniform(0, 3), 0, 100))
    suppression = float(np.clip((hypnotic-7.5)/2.5, 0, 0.8))
    return {"BIS*": round(bis), "SEF95 (Hz)": round(sef95, 1),
            "MF (Hz)": round(mf, 1), "SQI (/100)": sqi,
            "Suppression tendency*": round(suppression*100)}

def synthetic_dsa(state, t):
    # Illustrative power distribution, not measured or validated EEG.
    freqs = np.linspace(0.5, 30, 120)
    times = np.linspace(max(0, t-60), t, 160)
    p, d, k = state["propofol"], state["dexmedetomidine"], state["ketamine"]
    vol = 0.5*(state["sevoflurane"] + state["isoflurane"])
    hyp = p + 0.75*d + 0.7*vol
    Z = np.zeros((len(freqs), len(times)))
    for j, tt in enumerate(times):
        phase = tt/60
        slow_amp = 0.25 + 0.08*hyp
        alpha_amp = 0.15 + 0.08*p + 0.025*vol
        spindle_amp = 0.05 + 0.10*d
        beta_amp = 0.08 + 0.045*k
        slow = slow_amp*np.exp(-0.5*((freqs-(1.5+0.15*np.sin(phase)))/1.3)**2)
        alpha = alpha_amp*np.exp(-0.5*((freqs-(10+0.3*np.sin(phase*1.7)))/2.0)**2)
        spindle = spindle_amp*np.exp(-0.5*((freqs-12.5)/1.8)**2)
        beta = beta_amp*np.exp(-0.5*((freqs-22)/4.0)**2)
        Z[:, j] = slow + alpha + spindle + beta + 0.008*np.random.random(len(freqs))
        if hyp > 7:
            # intermittent low-power periods to suggest suppression; not real burst-suppression data
            burst_phase = (tt % 12)
            if burst_phase > 7:
                Z[:, j] *= 0.12
    return freqs, times, Z

with st.sidebar:
    st.header("Virtual drug controls")
    st.write("Set illustrative effect-site levels; these are not clinical doses or concentrations.")
    for label, (key, desc) in DRUGS.items():
        st.session_state.state[key] = st.slider(label, 0.0, 10.0,
            float(st.session_state.state[key]), 0.5, help=desc)
    st.session_state.noise = st.slider("Signal artefact / noise", 0.0, 1.0, float(st.session_state.noise), 0.05)
    c1, c2 = st.columns(2)
    if c1.button("Advance 5 s", use_container_width=True):
        st.session_state.t += 5
        m = current_metrics(st.session_state.state, st.session_state.noise)
        st.session_state.history.append({"Time (s)": st.session_state.t, **m})
    if c2.button("Reset", use_container_width=True):
        st.session_state.t = 0
        st.session_state.history = []
        st.session_state.drug_events = []
        st.session_state.state = {k: 0.0 for k in st.session_state.state}
        st.rerun()
    st.caption("Use “Advance 5 s” to move the simulation clock.")

metrics = current_metrics(st.session_state.state, st.session_state.noise)
cols = st.columns(5)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, value)

left, right = st.columns([1.5, 1])
with left:
    st.subheader("Density Spectral Array (DSA)")
    f, tt, z = synthetic_dsa(st.session_state.state, st.session_state.t)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    mesh = ax.pcolormesh(tt, f, 10*np.log10(z + 1e-4), shading="auto", cmap="magma")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, 30)
    ax.set_title("Synthetic EEG power — illustrative")
    fig.colorbar(mesh, ax=ax, label="Relative power (dB, arbitrary units)")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with right:
    st.subheader("Drug state")
    active = [(name, round(st.session_state.state[key], 1)) for name, (key, _) in DRUGS.items()
              if st.session_state.state[key] > 0]
    if active:
        for name, level in active:
            st.write(f"**{name}:** illustrative level {level}/10")
    else:
        st.info("No simulated anaesthetic drugs selected.")
    st.warning("The index marked BIS* is a heuristic teaching value. It is not the proprietary BIS algorithm and must not be used to guide patient care.")

st.subheader("Trend")
if st.session_state.history:
    hist = pd.DataFrame(st.session_state.history).set_index("Time (s)")
    st.line_chart(hist[["BIS*", "SEF95 (Hz)", "MF (Hz)"]])
    st.dataframe(hist, use_container_width=True)
else:
    st.info("Press “Advance 5 s” to record trend points. The display updates when you change drug controls.")

st.markdown("---")
st.markdown("**Suggested next development steps:** add bolus/infusion timing and effect-site delay; create drug-specific scenario presets; improve DSA physiology using reference EEG; add raw EEG and suppression ratio; validate educational behaviours against published data or curated recordings.")

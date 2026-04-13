import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymcdm.methods import TOPSIS, MABAC, ARAS, WSM
from pymcdm import visuals

# Alias
SAW = WSM

st.set_page_config(page_title="MCDM Dashboard", layout="wide")
st.title("Multi-Criteria Decision Making (MCDM) Dashboard")

# --- 1. DATA INPUT ---
st.sidebar.header("1. Upload or Edit Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    # Auto-detect separator (handles ; or ,)
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
else:
    data = {
        'alternative': ['A1', 'A2', 'A3'],
        'discharge': [2.5, 3.0, 4.0],
        'cost': [50, 60, 80],
        'wetlands': [0.9, 0.6, 0.1],
        'forest': [0.1, 0.6, 0.3],
        'social acceptance': [0.17, 0.83, 0.50]
    }
    df = pd.DataFrame(data)

st.subheader("Decision Matrix")
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# Extract data
alts_names = edited_df.iloc[:, 0].astype(str).tolist()
criteria_names = edited_df.columns[1:]

# --- CLEAN DATA ---
alts_data = edited_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

if alts_data.isnull().values.any():
    st.error("❌ Data contains missing or non-numeric values. Please fix them.")
    st.stop()

# Replace zeros (important for WSM)
alts_data = alts_data.replace(0, 1e-6)

alts_data = alts_data.to_numpy(dtype=float)

# --- 2. WEIGHTS & TYPES ---
st.sidebar.header("2. Criteria Configuration")

weights_list = []
types_list = []

for col in criteria_names:
    st.sidebar.markdown(f"**{col}**")
    c1, c2 = st.sidebar.columns(2)

    with c1:
        weight = st.slider(
            f"Weight ({col})",
            min_value=0.0,
            max_value=1.0,
            value=1.0 / len(criteria_names),
            key=f"w_{col}"
        )
        weights_list.append(weight)

    with c2:
        ctype = st.radio(
            f"Type ({col})",
            options=["Benefit", "Cost"],
            key=f"t_{col}"
        )
        types_list.append(1 if ctype == "Benefit" else -1)

weights = np.array(weights_list)
if np.sum(weights) > 0:
    weights = weights / np.sum(weights)

types = np.array(types_list)

# --- 3. METHODS ---
st.sidebar.header("3. Select MCDM Methods")

available_methods = {
    'TOPSIS': TOPSIS(),
    'SAW': SAW(),
    'MABAC': MABAC(),
    'ARAS': ARAS(),
    'WSM': WSM()
}

selected_method_names = st.sidebar.multiselect(
    "Choose evaluation methods:",
    list(available_methods.keys()),
    default=['TOPSIS', 'SAW']
)

# --- 4. RUN ANALYSIS ---
if st.button("Run MCDM Analysis"):

    if not selected_method_names:
        st.warning("Please select at least one method.")
        st.stop()

    methods = [available_methods[name] for name in selected_method_names]

    prefs = []
    ranks = []

    for method in methods:
        try:
            pref = np.array(method(alts_data, weights, types), dtype=float).flatten()

            if np.isnan(pref).any() or np.isinf(pref).any():
                st.error("❌ Invalid values (NaN/Inf) detected in results.")
                st.stop()

            # Safe ranking (no rrankdata)
            rank = np.argsort(-pref) + 1

            prefs.append(pref)
            ranks.append(rank)

        except Exception as e:
            st.error(f"❌ Error in method {method.__class__.__name__}: {e}")
            st.stop()

    # --- RESULTS DISPLAY ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Preference Table")
        pref_df = pd.DataFrame(
            zip(*prefs),
            columns=selected_method_names,
            index=alts_names
        ).round(4)

        st.dataframe(pref_df, use_container_width=True)

    with col2:
        st.subheader("Ranking Table")
        rank_df = pd.DataFrame(
            zip(*ranks),
            columns=selected_method_names,
            index=alts_names
        ).astype(int)

        st.dataframe(rank_df, use_container_width=True)

    # --- POLAR PLOT ---
    st.subheader("Polar Ranking Plot")

    fig, ax = plt.subplots(
        figsize=(7, 7),
        dpi=150,
        subplot_kw=dict(projection='polar')
    )

    visuals.polar_plot(ranks, labels=selected_method_names, legend_ncol=2, ax=ax)

    st.pyplot(fig)
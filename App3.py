# ===============================
# MCDM Streamlit Dashboard (Clean Version)
# ===============================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from pymcdm.methods import TOPSIS, MABAC, ARAS, WSM
from pymcdm import visuals

SAW = WSM  # Alias

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="MCDM Dashboard", layout="wide")
st.title("Multi-Criteria Decision Making (MCDM) Dashboard")

# ===============================
# 1. DATA INPUT
# ===============================
st.sidebar.header("1. Upload or Edit Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

# --- Load Data ---
if uploaded_file is not None:
    try:
        # Auto-detect delimiter
        uploaded_file.seek(0)
        dialect = csv.Sniffer().sniff(uploaded_file.read(1024).decode())
        uploaded_file.seek(0)

        df = pd.read_csv(uploaded_file, sep=';')

    except Exception:
        st.error("Could not read file. Ensure it's a valid CSV.")
        st.stop()
else:
    # Default fallback dataset
    data = {
        'alternative': ['A1', 'A2', 'A3'],
        'discharge': [2.5, 3.0, 4.0],
        'cost': [50, 60, 80],
        'wetlands': [0.9, 0.6, 0.1],
        'forest': [0.1, 0.6, 0.3],
        'social acceptance': [0.17, 0.83, 0.50]
    }
    df = pd.DataFrame(data)

# Safety check
if df.shape[1] == 1:
    st.error("CSV not parsed correctly. Try using ';' or ',' as delimiter.")
    st.stop()

# Editable table
st.subheader("Decision Matrix")
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# Extract data
alts_names = edited_df.iloc[:, 0].tolist()
criteria_names = edited_df.columns[1:]

# ===============================
# DATA CLEANING (CRITICAL)
# ===============================
alts_df = edited_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

if alts_df.isnull().values.any():
    st.error("Dataset contains missing or invalid (non-numeric) values.")
    st.stop()

alts_data = alts_df.to_numpy()
min_val = np.min(alts_data)

if min_val <= 0:
    st.warning("Data contains zero/negative values. Adjusting for WSM compatibility.")
    alts_data = alts_data - min_val + 1e-6
# ===============================
# 2. WEIGHTS & TYPES
# ===============================
st.sidebar.header("2. Criteria Configuration")

weights_list = []
types_list = []

for col in criteria_names:
    st.sidebar.markdown(f"**{col}**")
    c1, c2 = st.sidebar.columns(2)

    with c1:
        weight = st.slider(
            f"Weight {col}",
            0.0, 1.0,
            1.0 / len(criteria_names),
            key=f"w_{col}"
        )
        weights_list.append(weight)

    with c2:
        ctype = st.radio(
            f"Type {col}",
            ["Benefit", "Cost"],
            key=f"t_{col}"
        )
        types_list.append(1 if ctype == "Benefit" else -1)

weights = np.array(weights_list)

# Normalize weights
if np.sum(weights) > 0:
    weights = weights / np.sum(weights)

types = np.array(types_list)

# Validation
if len(weights) != alts_data.shape[1]:
    st.error("Weights must match number of criteria.")
    st.stop()

if len(types) != alts_data.shape[1]:
    st.error("Types must match number of criteria.")
    st.stop()

# ===============================
# 3. METHOD SELECTION
# ===============================
st.sidebar.header("3. Select MCDM Methods")

available_methods = {
    'TOPSIS': TOPSIS(),
    'SAW': SAW(),
    'MABAC': MABAC(),
    'ARAS': ARAS()
}

selected_method_names = st.sidebar.multiselect(
    "Choose methods:",
    list(available_methods.keys()),
    default=['TOPSIS', 'SAW']
)

# ===============================
# 4. RUN ANALYSIS
# ===============================
if st.button("Run MCDM Analysis"):

    if not selected_method_names:
        st.warning("Please select at least one method.")
        st.stop()

    methods = [available_methods[name] for name in selected_method_names]

    prefs = []
    ranks = []

    # --- Compute ---
    for method in methods:
        try:
            pref = method(alts_data, weights, types)

            prefs.append(pref)

            # Robust ranking (stable)
            rank = np.argsort(np.argsort(-pref)) + 1
            ranks.append(rank)

        except Exception as e:
            st.error(f"Error in method {method}: {e}")
            st.stop()

    # ===============================
    # RESULTS DISPLAY
    # ===============================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Preference Table")
        pref_df = pd.DataFrame(
            zip(*prefs),
            columns=selected_method_names,
            index=alts_names
        ).round(3)

        st.dataframe(pref_df, use_container_width=True)

    with col2:
        st.subheader("Ranking Table")
        rank_df = pd.DataFrame(
            zip(*ranks),
            columns=selected_method_names,
            index=alts_names
        ).astype(int)

        st.dataframe(rank_df, use_container_width=True)

    # ===============================
    # POLAR PLOT
    # ===============================
    st.subheader("Polar Ranking Plot")

fig, ax = plt.subplots(
    figsize=(5, 5),
    dpi=150,
    subplot_kw=dict(projection='polar')
)

visuals.polar_plot(
    ranks,
    labels=selected_method_names,
    legend_ncol=2,
    ax=ax
)

st.pyplot(fig, use_container_width=False)
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.pyplot(fig, use_container_width=False)   
   

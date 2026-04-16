# -*- coding: utf-8 -*-
"""
Decision Support System – MCDM Analysis
Dashboard-Compatible Version (TOPSIS & SAW)
"""

# ===============================
# Import Libraries
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymcdm.methods import TOPSIS, WSM
SAW = WSM  # Match dashboard alias
from pymcdm import visuals


# ===============================
# Load Data
# ===============================
df = pd.read_csv('Assign2.csv', sep=';')

# Extract alternatives and criteria
alts_names = df.iloc[:, 0].tolist()
criteria_names = df.columns[1:]
alts_data = df.iloc[:, 1:].to_numpy()


# ===============================
# Define Weights and Criteria Types
# ===============================
weights = np.array([0.29, 0.24, 0.19, 0.14, 0.10, 0.05])

# Normalize weights (same as dashboard)
if np.sum(weights) > 0:
    weights = weights / np.sum(weights)

# 1 = benefit, -1 = cost
types = np.array([1, 1, -1, 1, 1, 1])


# ===============================
# Define Methods (like dashboard)
# ===============================
available_methods = {
    'TOPSIS': TOPSIS(),
    'SAW': SAW()
}

selected_method_names = ['TOPSIS', 'SAW']
methods = [available_methods[name] for name in selected_method_names]


# ===============================
# Compute Preferences and Rankings
# ===============================
prefs = []
ranks = []

for method in methods:
    pref = method(alts_data, weights, types)

    prefs.append(pref)

    # Use SAME ranking logic as dashboard
    rank = np.argsort(-pref) + 1
    ranks.append(rank)


# ===============================
# Display Results (same structure)
# ===============================
print("\nPreference Table\n")
pref_df = pd.DataFrame(
    zip(*prefs),
    columns=selected_method_names,
    index=alts_names
).round(3)

print(pref_df)


print("\nRanking Table\n")
rank_df = pd.DataFrame(
    zip(*ranks),
    columns=selected_method_names,
    index=alts_names
).astype(int)

print(rank_df)


# ===============================
# Polar Plot (same as dashboard)
# ===============================
fig, ax = plt.subplots(
    figsize=(7, 7),
    dpi=150,
    tight_layout=True,
    subplot_kw=dict(projection='polar')
)

visuals.polar_plot(
    ranks,
    labels=selected_method_names,
    legend_ncol=2,
    ax=ax
)

plt.show()
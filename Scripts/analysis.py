import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import t
import glob
import os
import re

import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

# ------------------------
# LOAD DATA
# ------------------------
df = pd.read_csv("Dataset.csv")

# Ensure numeric variables
df['ndq_total'] = pd.to_numeric(df['ndq_total'], errors='coerce')
df['haeufigkeit'] = pd.to_numeric(df['haeufigkeit'], errors='coerce')

df['group'] = df['group'].str.strip().astype(str)
df['VPCode_norm'] = df['VPCode_norm'].astype(str).str.strip()
df['timepoint'] = df['timepoint'].astype(str).str.strip()


def compare_ndq_scores(df):
    """
    Compares NDQ scores between intervention and control groups at each timepoint.

    Args:
        df (pd.DataFrame): Full DataFrame containing NDQ scores, with 'group', 'timepoint', and 'ndq_total'.
    """
    timepoints = sorted(df['timepoint'].unique())

    for tp in timepoints:
        tp_data = df[df['timepoint'] == tp]
        ndq_I = tp_data[tp_data['group'] == 'Intervention']['ndq_total'].dropna()
        ndq_C = tp_data[tp_data['group'] == 'Control']['ndq_total'].dropna()

        print(f"\n=== NDQ Total Score Summary — Timepoint: {tp} ===")
        print("Intervention Group:")
        print(f"  Mean: {ndq_I.mean():.2f}")
        print(f"  Std: {ndq_I.std():.2f}")
        print(f"  Count: {ndq_I.count()}")

        print("Control Group:")
        print(f"  Mean: {ndq_C.mean():.2f}")
        print(f"  Std: {ndq_C.std():.2f}")
        print(f"  Count: {ndq_C.count()}")

        # Optional t-test
        if len(ndq_I) > 1 and len(ndq_C) > 1:
            t_stat, p_val = stats.ttest_ind(ndq_I, ndq_C, equal_var=False)
            print(f"  t-test: t = {t_stat:.3f}, p = {p_val:.4f}")


# ------------------------
# 1. NF Score Comparison
# ------------------------

def compare_nf_scores(df, output_csv_path="nf_summary.csv"):
    """
    Compares Nightmare Frequency scores between intervention and control groups at each timepoint.
    Saves results to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing 'haeufigkeit', 'group', and 'timepoint'.
        output_csv_path (str): Path to save the CSV summary table.
    """
    timepoints = sorted(df['timepoint'].dropna().unique())
    results = []

    for tp in timepoints:
        tp_data = df[df['timepoint'] == tp]
        nf_I = tp_data[tp_data['group'] == 'Intervention']['haeufigkeit'].dropna()
        nf_C = tp_data[tp_data['group'] == 'Control']['haeufigkeit'].dropna()

        # Compute stats
        row = {
            'Timepoint': tp,
            'Intervention_Mean': nf_I.mean(),
            'Intervention_Std': nf_I.std(),
            'Intervention_N': nf_I.count(),
            'Control_Mean': nf_C.mean(),
            'Control_Std': nf_C.std(),
            'Control_N': nf_C.count(),
            't_stat': None,
            'p_val': None
        }

        if len(nf_I) > 1 and len(nf_C) > 1:
            t_stat, p_val = stats.ttest_ind(nf_I, nf_C, equal_var=False)
            row['t_stat'] = t_stat
            row['p_val'] = p_val

        results.append(row)

    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    # Save to CSV
    result_df.to_csv(output_csv_path, index=False)
    print(f"\n Nightmare Frequency summary saved to: {output_csv_path}")

    return result_df

#------------------------
# 2. Simple Comparisons + Data Overview Table
# ----------------

# --- Run Comparisons ---
compare_ndq_scores(df)

compare_nf_scores(df)
# --- Timepoint Completion Table ---

#table showing which participants completed which timepoints
# Create binary presence indicator
df['has_data'] = 1
# Pivot: VPCode_norm as rows, timepoints as columns
completion_table = df.pivot_table(
    index='VPCode_norm',
    columns='timepoint', 
    values='has_data',
    aggfunc='first',
    fill_value=0
).sort_index(axis=1)
# Extract unique group info per VPCode
group_info = df[['VPCode_norm', 'group']].drop_duplicates()
group_info['VPCode_norm'] = group_info['VPCode_norm'].astype(str).str.strip()
completion_table.index = completion_table.index.astype(str).str.strip()

# Remove total row if already added (to avoid merge issues)
if 'TotalParticipants' in completion_table.index:
    total_row = completion_table.loc['TotalParticipants']
    completion_table = completion_table.drop(index='TotalParticipants')
else:
    total_row = None

# Merge group info
completion_table = completion_table.reset_index().merge(group_info, on='VPCode_norm', how='left').set_index('VPCode_norm')

# Re-add total row if it was removed earlier
if total_row is not None:
    completion_table.loc['TotalParticipants'] = total_row

# --- Add Summary Columns and Rows ---

# Add column: number of timepoints completed per participant
completion_table['TotalCompleted'] = completion_table.drop(columns='group', errors='ignore').sum(axis=1)

# Add row: total number of participants per timepoint
participant_sums = completion_table.drop(columns=['group', 'TotalCompleted'], errors='ignore').sum(numeric_only=True)
completion_table.loc['TotalParticipants'] = participant_sums

# Create a separate pivot for nightmare frequency
haufigkeit_table = df.pivot_table(
    index='VPCode_norm',
    columns='timepoint',
    values='haeufigkeit',
    aggfunc='mean'  # or 'first', 'max', etc., depending on your data
)

# Merge the haufigkeit data into the completion table
for col in haufigkeit_table.columns:
    completion_table[f"haeufigkeit_{col}"] = haufigkeit_table[col]

df_ndq = df[['VPCode_norm', 'group', 'timepoint', 'ndq_total']].dropna()
df =    df[['VPCode_norm', 'group', 'timepoint', 'haeufigkeit', 'ndq_total']].dropna()
# Pivot NDQ scores to wide format
ndq_table = df_ndq.pivot_table(
    index='VPCode_norm',       # participant as rows
    columns='timepoint',       # each timepoint becomes a column
    values='ndq_total',        # values to fill
    aggfunc='mean'             # in case there are duplicates per timepoint
)

# Optionally, rename columns to be clearer
ndq_table = ndq_table.add_prefix('ndq_')

# Merge into your completion table
completion_table = completion_table.merge(
    ndq_table,
    left_index=True,
    right_index=True,
    how='left'
)

# Show first few rows

completion_table.to_csv('timepoint_completion_with_scores.csv')


# Filter out rows where 'haeufigkeit' is missing
df_nf = df[['VPCode_norm', 'group', 'timepoint', 'haeufigkeit']].dropna()
# Strip extra quotes here BEFORE fitting the model or plotting
df_nf['group'] = df_nf['group'].str.strip("'\"")


# Get a list of unique participant IDs
participant_list = df['VPCode_norm'].unique()

# Convert to a Python list (optional, mostly for readability)
participant_list = participant_list.tolist()

# Save the list to a text file, one ID per line
with open("participant_ids.txt", "w") as f:
    for pid in participant_list:
        f.write(f"{pid}\n")

# Or save as a CSV

pd.DataFrame(participant_list, columns=["VPCode_norm"]).to_csv("participant_ids.csv", index=False)


# ------------------------
# 3. Linear Mixed Models
# ----------------

# Fit linear mixed model
#alternatively: add re_formula="~timepoint" to include random slopes
model = smf.mixedlm("ndq_total ~ group * timepoint", df_ndq, groups=df_ndq["VPCode_norm"])
result = model.fit(reml=False)

# Print model summary
print(result.summary())


# Fit a linear mixed-effects model  Nightmare Frequency
#alternatively: add re_formula="~timepoint" to include random slopes
model_nf = smf.mixedlm(
    formula='haeufigkeit ~ group * timepoint',
    data=df_nf,
    groups=df_nf['VPCode_norm']
).fit(reml =False)

# Show summary
print(model_nf.summary())


"""Nightmare burden Index
To create a composite Nightmare Burden Index, we will standardize both the NDQ total score and the nightmare frequency (haeufigkeit) across all participants and timepoints. 
Then, we will average these two standardized scores to get a single index that reflects overall nightmare burden.
"""

# Standardize NDQ and NF and compute Nightmare Burden Index
df['ndq_z2'] = (df['ndq_total'] - df['ndq_total'].mean()) / df['ndq_total'].std()
df['nf_z2'] = (df['haeufigkeit'] - df['haeufigkeit'].mean()) / df['haeufigkeit'].std()
df['nightmare_burden2'] = (df['ndq_z2'] + df['nf_z2']) / 2


baseline = df[df['timepoint'] == 't0']
ndq_mean = baseline['ndq_total'].mean()
ndq_sd = baseline['ndq_total'].std()
nf_mean = baseline['haeufigkeit'].mean()
nf_sd = baseline['haeufigkeit'].std()
df['ndq_z'] = (df['ndq_total'] - ndq_mean) / ndq_sd
df['nf_z'] = (df['haeufigkeit'] - nf_mean) / nf_sd
df['nightmare_burden'] = (df['ndq_z'] + df['nf_z']) / 2


mean_burden_group = df.groupby(['group', 'timepoint'])['nightmare_burden'].mean()
print(mean_burden_group)

mean_std_burden2 = df.groupby(['group','timepoint'])['nightmare_burden2'].mean()
print(mean_std_burden2)

model_nbi_timepoint = smf.mixedlm(
    "nightmare_burden ~ group * timepoint",
    df,
    groups=df["VPCode_norm"]
).fit(reml=False)

print("\n=== LMM: Nightmare Burden Index ~ Group * Timepoint ===")
print(model_nbi_timepoint.summary())

#descriptive statistics function mean and SD
def descriptives_by_timepoint(df, outcome):
    return (
        df
        .groupby(['group', 'timepoint'])[outcome]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
desc_ndq = descriptives_by_timepoint(df, 'ndq_total')

# Format Mean (SD)
desc_ndq['Mean (SD)'] = desc_ndq.apply(
    lambda r: f"{r['mean']:.2f} ({r['std']:.2f})", axis=1
)

# Pivot to table format
table_ndq = desc_ndq.pivot(
    index='timepoint',
    columns='group',
    values='Mean (SD)'
)

print(table_ndq)
desc_nf = descriptives_by_timepoint(df, 'haeufigkeit')

desc_nf['Mean (SD)'] = desc_nf.apply(
    lambda r: f"{r['mean']:.2f} ({r['std']:.2f})", axis=1
)

table_nf = desc_nf.pivot(
    index='timepoint',
    columns='group',
    values='Mean (SD)'
)

print(table_nf)
desc_burden = descriptives_by_timepoint(df, 'nightmare_burden')

desc_burden['Mean (SD)'] = desc_burden.apply(
    lambda r: f"{r['mean']:.2f} ({r['std']:.2f})", axis=1
)

table_burden = desc_burden.pivot(
    index='timepoint',
    columns='group',
    values='Mean (SD)'
)

print(table_burden)


# Compute mean + 68% CI per group & timepoint (baseline-standardized)
df_plot = df.groupby(['group','timepoint'])['nightmare_burden'].agg(['mean','std','count']).reset_index()
df_plot['ci68'] = 1 * df_plot['std'] / df_plot['count']**0.5  # ~68% CI

# Line plot
plt.figure(figsize=(10,6))
sns.lineplot(data=df_plot, x='timepoint', y='mean', hue='group', marker='o')

# Add CI as shaded area
for group in df_plot['group'].unique():
    grp = df_plot[df_plot['group'] == group]
    plt.fill_between(grp['timepoint'], grp['mean']-grp['ci68'], grp['mean']+grp['ci68'], alpha=0.2)

plt.title("Nightmare Burden Over Time by Group (Baseline-standardized)")
plt.ylabel("Nightmare Burden (z-score vs Baseline)")
plt.xlabel("Timepoint")
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.legend(title='Group')
plt.savefig("nightmare_burden_plot.png", dpi=300, bbox_inches='tight')



#----------------------------------------------
# 4. Jackknife Estimation for NDQ Total Score
#----------------------------------------------

#df_ndq with 'VPCode_norm', 'group', 'timepoint', 'ndq_total'
participants = df_ndq['VPCode_norm'].unique()
n = len(participants)

jackknife_estimates = []

for i, vp in enumerate(participants):
    # Leave out one participant
    df_jack = df_ndq[df_ndq['VPCode_norm'] != vp]
    
    # Fit LMM on jackknife sample
    model = smf.mixedlm("ndq_total ~ group * timepoint", df_jack, groups=df_jack["VPCode_norm"])
    result = model.fit(reml=False)  # Use reml=False for better comparability
    
    # Store fixed effect estimates as a dictionary
    jackknife_estimates.append(result.params)

    if (i+1) % 10 == 0 or i == n-1:
        print(f"Jackknife iteration {i+1}/{n} complete")

# Convert to DataFrame
jackknife_df = pd.DataFrame(jackknife_estimates)

# Jackknife mean of estimates
jackknife_mean = jackknife_df.mean()

# Jackknife standard error (formula for leave-one-out jackknife std error)
jackknife_se = np.sqrt((n - 1) / n * ((jackknife_df - jackknife_mean) ** 2).sum())


# Confidence Intervals
ci_lower = jackknife_mean - 1.96 * jackknife_se
ci_upper = jackknife_mean + 1.96 * jackknife_se

# Degrees of freedom: n - 1
df_jack = n - 1  # n = number of participants

t_values = jackknife_mean / jackknife_se
p_values = 2 * (1 - t.cdf(abs(t_values), df=df_jack))
summary_ndq = pd.DataFrame({
    "Estimate": jackknife_mean,
    "SE": jackknife_se,
    "CI Lower": ci_lower,
    "CI Upper": ci_upper,
    "t": t_values,
    "p-value": p_values
})

# Optionally round for cleaner output
summary_ndq = summary_ndq.round(4)
print("\nJackknife Summary for NDQ Total Score:")
print(summary_ndq)
# --- Same for Nightmare Frequency ---

participants = df_nf['VPCode_norm'].unique()
n = len(participants)

jackknife_estimates_nf = []

# --- Jackknife Loop ---
for i, vp in enumerate(participants):
    # Leave-one-out dataset
    df_jack = df_nf[df_nf['VPCode_norm'] != vp]

    # Fit linear mixed model
    model = smf.mixedlm("haeufigkeit ~ group * timepoint", df_jack, groups=df_jack["VPCode_norm"])
    try:
        result = model.fit(reml=False)
        jackknife_estimates_nf.append(result.params)
    except Exception as e:
        print(f"Warning: Model failed to fit at iteration {i+1} (VP: {vp}) — {e}")
        continue

    if (i + 1) % 10 == 0 or i == n - 1:
        print(f"Jackknife iteration {i+1}/{n} complete")

# --- Convert to DataFrame ---
jackknife_df_nf = pd.DataFrame(jackknife_estimates_nf)

# --- Calculate Jackknife Statistics ---
jackknife_mean_nf = jackknife_df_nf.mean()
jackknife_se_nf = np.sqrt((n - 1) / n * ((jackknife_df_nf - jackknife_mean_nf) ** 2).sum())

# Confidence Intervals
ci_lower_nf = jackknife_mean_nf - 1.96 * jackknife_se_nf
ci_upper_nf = jackknife_mean_nf + 1.96 * jackknife_se_nf

# Degrees of freedom
df_deg = n - 1

# t-values and p-values
t_values_nf = jackknife_mean_nf / jackknife_se_nf
p_values_nf = 2 * (1 - t.cdf(abs(t_values_nf), df=df_deg))

# --- Summary Table ---
summary_nf = pd.DataFrame({
    "Estimate": jackknife_mean_nf,
    "SE": jackknife_se_nf,
    "CI Lower": ci_lower_nf,
    "CI Upper": ci_upper_nf,
    "t": t_values_nf,
    "p-value": p_values_nf
}).round(4)

print("\nJackknife Summary for Nightmare Frequency (haeufigkeit):")
print(summary_nf)

# Compute influence = deviation from jackknife mean
#influence = jackknife_df - jackknife_mean
#influence.index = participants  # label rows by participant ID

# Frobenius norm (overall influence across all coefficients)
#influence["total_influence"] = np.sqrt((influence**2).sum(axis=1))

#print(influence["total_influence"].sort_values(ascending=False))
# Compute influence = deviation from jackknife mean
#influence = jackknife_df_nf - jackknife_mean_nf
#influence.index = participants  # label rows by participant ID

# Frobenius norm (overall influence across all coefficients)
#influence["total_influence"] = np.sqrt((influence**2).sum(axis=1))

#print(influence["total_influence"].sort_values(ascending=False))


#----------------------------------------------
# 4. Permutation Tests
#----------------------------------------------


#: Permutation Tests for NDQ Total and Nightmare Frequency fitting Linear Mixed Models

def _permute_labels_by_cluster(df, group_col, cluster_col, categories=None):
    """
    Permute group labels across clusters (participants) and map back to rows.
    Preserves cluster structure and avoids mixing labels within a participant.
    """
    unique_clusters = (
        df[[cluster_col, group_col]]
        .drop_duplicates(subset=cluster_col)
        .reset_index(drop=True)
    )
    permuted = unique_clusters[group_col].sample(frac=1.0, replace=False, random_state=None).values
    mapping = dict(zip(unique_clusters[cluster_col].values, permuted))
    new_labels = df[cluster_col].map(mapping)
    if categories is not None:
        new_labels = pd.Categorical(new_labels, categories=categories)
    return new_labels


def permutation_test_lmm_interaction(data, formula, group_col, group_var, n_permutations=5000, combine_terms="l2"):
    """
    Permutation test for the group × timepoint interaction in an LMM.
    - Shuffles group labels across participants (clusters) to build a null distribution.
    - combine_terms: 'l2' (default) or 'maxabs' to combine multiple contrasts.
    Returns (observed_stat, p_value, matching_terms).
    """
    #fit the overserved model 
    model = smf.mixedlm(formula, data=data, groups=data[group_var])
    fit = model.fit(reml=False)
    print(fit.summary()) 

    term_mask = ["group[T.Intervention]:timepoint" in str(term) for term in fit.params.index]
    matching_terms = [term for term, keep in zip(fit.params.index, term_mask) if keep]
    if len(matching_terms) == 0:
        raise ValueError("Could not find group×timepoint interaction terms. Check formula or coding.")

    observed_betas = np.array([fit.params.get(term, np.nan) for term in matching_terms], dtype=float)
    if combine_terms == "l2":
        observed_stat = float(np.linalg.norm(observed_betas))
    elif combine_terms == "maxabs":
        observed_stat = float(np.max(np.abs(observed_betas)))
    else:
        raise ValueError("combine_terms must be 'l2' or 'maxabs'")

    categories = None
    if pd.api.types.is_categorical_dtype(data[group_col]):
        categories = data[group_col].cat.categories

    perm_stats = []
    for _ in tqdm(range(n_permutations), desc="Permuting group labels"):
        df_perm = data.copy()
        df_perm[group_col] = _permute_labels_by_cluster(df_perm, group_col, group_var, categories)
        perm_model = smf.mixedlm(formula, data=df_perm, groups=df_perm[group_var])
        perm_fit = perm_model.fit(reml=False)
        perm_betas = np.array([perm_fit.params.get(term, np.nan) for term in matching_terms], dtype=float)
        if combine_terms == "l2":
            perm_stats.append(float(np.linalg.norm(perm_betas)))
        else:
            perm_stats.append(float(np.max(np.abs(perm_betas))))

    perm_stats = np.array(perm_stats)
    p_value = float(np.mean(perm_stats >= observed_stat))

    return observed_stat, p_value, matching_terms


def permutation_test_lmm_all_interactions(data, formula, group_col, group_var, n_permutations=5000):
    """
    Compute permutation p-values for each individual group×timepoint interaction contrast.
    Returns a DataFrame with columns: term, observed_beta, p_value.
    """
    model = smf.mixedlm(formula, data=data, groups=data[group_var])
    fit = model.fit(reml=False)
    terms = [t for t in fit.params.index if 'group[T.Intervention]:timepoint' in str(t)]
    results = []

    categories = None
    if pd.api.types.is_categorical_dtype(data[group_col]):
        categories = data[group_col].cat.categories

    for term in terms:
        observed_beta = float(fit.params.get(term, np.nan))
        perm_betas = []
        for _ in range(n_permutations):
            df_perm = data.copy()
            df_perm[group_col] = _permute_labels_by_cluster(df_perm, group_col, group_var, categories)
            perm_model = smf.mixedlm(formula, data=df_perm, groups=df_perm[group_var])
            perm_fit = perm_model.fit(reml=False)
            perm_betas.append(float(perm_fit.params.get(term, np.nan)))
        perm_betas = np.array(perm_betas)
        p_value = float(np.mean(np.abs(perm_betas) >= np.abs(observed_beta)))
        results.append({"term": term, "observed_beta": observed_beta, "p_value": p_value})

    return pd.DataFrame(results)
perm_int_results_nf = permutation_test_lmm_all_interactions(
    df_nf, "haeufigkeit ~ group * timepoint", "group", "VPCode_norm"
)
print("\nPer-term permutation results (NF)1:")
print(perm_int_results_nf)
perm_int_results_ndq = permutation_test_lmm_all_interactions(
    df_ndq, "ndq_total ~ group * timepoint", "group", "VPCode_norm"
    )
print("\nPer-term permutation results (NDQ)1:")
print(perm_int_results_ndq)
# Optionally save:
# perm_results_ndq.to_csv("perm_test_ndq.csv", index=False)
# perm_results_nf.to_csv("perm_test_nf.csv", index=False)

# --- Run and print permutation tests ---
try:
    obs_ndq_stat, obs_ndq_p, ndq_terms = permutation_test_lmm_interaction(
        df_ndq,
        "ndq_total ~ group * timepoint",
        group_col="group",
        group_var="VPCode_norm",
        n_permutations=1000,
    )
    print("\nPermutation test (NDQ total ~ group * timepoint):")
    print("terms:", ndq_terms)
    print(f"observed_stat: {obs_ndq_stat:.6f}")
    print(f"p_value: {obs_ndq_p:.6f}")

except Exception as e:
    print(f"NDQ permutation test failed: {e}")

try:
    obs_nf_stat, obs_nf_p, nf_terms = permutation_test_lmm_interaction(
        df_nf,
        "haeufigkeit ~ group * timepoint",
        group_col="group",
        group_var="VPCode_norm",
        n_permutations=1000,
    )
    print("\nPermutation test (NF haeufigkeit ~ group * timepoint):")
    print("terms:", nf_terms)
    print(f"observed_stat: {obs_nf_stat:.6f}")
    print(f"p_value: {obs_nf_p:.6f}")

    perm_int_results_nf = permutation_test_lmm_all_interactions(
        df_nf,
        "haeufigkeit ~ group * timepoint",
        group_col="group",
        group_var="VPCode_norm",
        n_permutations=1000,
    )
    print("\nPer-term permutation results (NF):")
    print(perm_int_results_nf)
except Exception as e:
    print(f"NF permutation test failed: {e}")

# ----------------------------------------------    
# Bootstrapp LMMs 
# ----------------------------------------------
def intervention_vs_control_at_t4(fit):
    p = fit.params

    # main group effect
    effect = p.get("group[T.Intervention]", 0)

    # interaction at t4
    effect += p.get("group[T.Intervention]:timepoint[T.t4]", 0)

    return effect
def intervention_vs_control_at_t3(fit):
    p = fit.params

    # main group effect
    effect = p.get("group[T.Intervention]", 0)

    # interaction at t4
    effect += p.get("group[T.Intervention]:timepoint[T.t3]", 0)

    return effect
#cluster bootstrap function
def cluster_bootstrap_lmm(df, formula, cluster_col, contrast_func, n_boot=1000, random_state=None):
    """
    Cluster bootstrap for linear mixed models.
    
    Parameters:
        df : pd.DataFrame
            Data containing dependent variable, predictors, and cluster column.
        formula : str
            Patsy formula for the mixed model.
        cluster_col : str
            Column for random effect grouping (e.g., participant ID).
        contrast_func : function
            Function that takes a fitted model and returns the contrast of interest (scalar or array).
        n_boot : int
            Number of bootstrap iterations.
        random_state : int or None
            Random seed.
    
    Returns:
        contrast_mean : float or np.array
            Mean contrast across bootstrap samples.
        ci_lower : float or np.array
            Lower bound of 95% CI.
        ci_upper : float or np.array
            Upper bound of 95% CI.
        bootstrap_values : np.array
            All bootstrap contrast values.
    """
    if random_state is not None:
        np.random.seed(random_state)

    unique_clusters = df[cluster_col].unique()
    n_clusters = len(unique_clusters)
    bootstrap_values = []

    for _ in tqdm(range(n_boot), desc="Cluster bootstrap"):
        # Resample clusters with replacement
        sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)
        df_boot = pd.concat([df[df[cluster_col] == c] for c in sampled_clusters], ignore_index=True)
        
        # Fit the model
        try:
            model_boot = smf.mixedlm(formula, data=df_boot, groups=df_boot[cluster_col])
            fit_boot = model_boot.fit(reml=False)
            # Compute contrast
            contrast_val = contrast_func(fit_boot)
            bootstrap_values.append(contrast_val)
        except Exception as e:
            print(f"Bootstrap iteration failed: {e}")
            continue

    bootstrap_values = np.array(bootstrap_values)
    contrast_mean = np.mean(bootstrap_values, axis=0)
    ci_lower = np.percentile(bootstrap_values, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_values, 97.5, axis=0)

    return contrast_mean, ci_lower, ci_upper, bootstrap_values

mean_diff, ci_low, ci_high, boot_vals = cluster_bootstrap_lmm(
    df_ndq,
    formula="ndq_total ~ group * timepoint",
    cluster_col="VPCode_norm",
    contrast_func=intervention_vs_control_at_t4,
    n_boot=1000,
    random_state=42
)

print("Intervention vs Control at t4 (NDQ):")
print(f"Mean difference: {mean_diff:.3f}")
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
mean_diff, ci_low, ci_high, boot_vals = cluster_bootstrap_lmm(
    df_nf,
    formula="haeufigkeit ~ group * timepoint",
    cluster_col="VPCode_norm",
    contrast_func=intervention_vs_control_at_t4,
    n_boot=1000,
    random_state=42
)

print("Intervention vs Control at t4 (NF):")
print(f"Mean difference: {mean_diff:.3f}")
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
print(result.params.index.tolist())


mean_diff, ci_low, ci_high, boot_vals = cluster_bootstrap_lmm(
    df_ndq,
    formula="ndq_total ~ group * timepoint",
    cluster_col="VPCode_norm",
    contrast_func=intervention_vs_control_at_t3,
    n_boot=1000,
    random_state=42
)

print("Intervention vs Control at t3 (NDQ):")
print(f"Mean difference: {mean_diff:.3f}")
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
mean_diff, ci_low, ci_high, boot_vals = cluster_bootstrap_lmm(
    df_nf,
    formula="haeufigkeit ~ group * timepoint",
    cluster_col="VPCode_norm",
    contrast_func=intervention_vs_control_at_t3,
    n_boot=1000,
    random_state=42
)

print("Intervention vs Control at t3 (NF):")
print(f"Mean difference: {mean_diff:.3f}")
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
print(result.params.index.tolist())


# ------------------------
# PRE POST ANALYSIS (LMM with condition variable)
# 
# ------------------------


df_simple3 = df.copy() 
df_simple3['phase'] = df_simple3['timepoint'].apply(
    lambda x: 'Baseline' if x == 't0' else 'Post'
)
df_simple3['phase'] = pd.Categorical(
    df_simple3['phase'],
    categories=['Baseline','Post']
)

df_simple3['group'] = pd.Categorical(
    df_simple3['group'],
    categories=['Control','Intervention']
)

# Ensure NDQ and NF are numeric
df_simple3['ndq_total'] = pd.to_numeric(df_simple3['ndq_total'], errors='coerce')
df_simple3['haeufigkeit'] = pd.to_numeric(df_simple3['haeufigkeit'], errors='coerce')


baseline = df_simple3[df_simple3['timepoint'] == 't0']
ndq_mean = baseline['ndq_total'].mean()
ndq_sd = baseline['ndq_total'].std()
nf_mean = baseline['haeufigkeit'].mean()
nf_sd = baseline['haeufigkeit'].std()
df_simple3['ndq_z'] = (df_simple3['ndq_total'] - ndq_mean) / ndq_sd
df_simple3['nf_z'] = (df_simple3['haeufigkeit'] - nf_mean) / nf_sd
df_simple3['nightmare_burden'] = (df_simple3['ndq_z'] + df_simple3['nf_z']) / 2


# 3. Fit mixed model with Baseline as reference

model_ndq = smf.mixedlm(
    "ndq_total ~ group * phase",
    df_simple3,
    groups=df_simple3["VPCode_norm"]
).fit(reml=False)
print("\n=== Condition Model (Baseline vs Post)Distress ===")
print(model_ndq.summary())

# --- Same for Nightmare Frequency ---

# 3. Fit mixed model with Baseline as reference Nightmare Frequency

model_nf = smf.mixedlm(
    "haeufigkeit ~ group * phase",
    df_simple3,
    groups=df_simple3["VPCode_norm"]
).fit(reml=False)
print("\n=== Condition Model (Baseline vs Post) Frequency ===")
print(model_nf.summary())

model_ndq2 = smf.mixedlm(
    "ndq_total ~ group * phase + haeufigkeit",
    df_simple3,
    groups=df_simple3["VPCode_norm"]
).fit(reml=False)
print("\n=== Condition Model (Baseline vs Post)Distress + Frequency ===")
print(model_ndq2.summary())

# --- Same for Nightmare Frequency ---

# 3. Fit mixed model with Baseline as reference Nightmare Frequency

model_nf2 = smf.mixedlm(
    "haeufigkeit ~ group * phase + ndq_total",
    df_simple3,
    groups=df_simple3["VPCode_norm"]
).fit(reml=False)
print("\n=== Condition Model (Baseline vs Post) Frequency + Distress ===")
print(model_nf2.summary())


# Fit mixed model for Nightmare Burden
model_condition_nbi = smf.mixedlm(
    "nightmare_burden ~ group * phase",
    df_simple3,
    groups=df_simple3["VPCode_norm"]
).fit(reml=False)

print("\n=== Condition Model: Nightmare Burden Index ===")
print(model_condition_nbi.summary())

# Select outcomes of interest
outcomes = ['haeufigkeit', 'ndq_total', 'nightmare_burden']
desc_stats = (
    df_simple3
    .groupby(['group','phase'], observed=False)[outcomes]
    .agg(['mean','std'])
    .round(2)
)

desc_stats.columns = ['_'.join(col) for col in desc_stats.columns]
desc_stats = desc_stats.reset_index()

print(desc_stats)
# save desc stats
desc_stats.to_csv("descriptive_stats_by_condition.csv", index=False)
# -------------------------------
# Function for jackknife
# -------------------------------
def jackknife_lmm(df, formula, group_col):
    """
    Performs leave-one-out jackknife estimation for a mixed-effects model.
    
    Parameters:
        df : pd.DataFrame
            Data containing dependent variable, predictors, and group column.
        formula : str
            Patsy formula for mixed model, e.g. "ndq_total ~ C(condition)"
        group_col : str
            Column name for random effect grouping (e.g., "VPCode_norm")
    
    Returns:
        summary_df : pd.DataFrame
            DataFrame containing jackknife estimates, SE, CI, t, p-values.
        influence_df : pd.DataFrame
            Frobenius norm of influence per participant.
    """
    participants = df[group_col].unique()
    n = len(participants)
    jack_estimates = []

    # Jackknife loop
    for i, vp in enumerate(participants):
        df_jack = df[df[group_col] != vp]
        try:
            model = smf.mixedlm(formula, df_jack, groups=df_jack[group_col])
            result = model.fit(reml=False)
            jack_estimates.append(result.params)
        except Exception as e:
            print(f"Warning: Model failed for participant {vp}: {e}")
            continue
        
        if (i+1) % 10 == 0 or i == n-1:
            print(f"Jackknife iteration {i+1}/{n} complete")
    
    # Convert to DataFrame
    jack_df = pd.DataFrame(jack_estimates)
    
    # Jackknife mean and SE
    jack_mean = jack_df.mean()
    jack_se = np.sqrt((n-1)/n * ((jack_df - jack_mean)**2).sum())
    
    # Confidence intervals
    ci_lower = jack_mean - 1.96 * jack_se
    ci_upper = jack_mean + 1.96 * jack_se
    
    # t-values and p-values (df = n-1)
    df_jack = n - 1
    t_values = jack_mean / jack_se
    p_values = 2 * (1 - t.cdf(abs(t_values), df=df_jack))
    
    summary_df = pd.DataFrame({
        "Estimate": jack_mean,
        "SE": jack_se,
        "CI Lower": ci_lower,
        "CI Upper": ci_upper,
        "t": t_values,
        "p-value": p_values
    }).round(4)
    
    # Compute influence
    influence = jack_df - jack_mean
    influence["total_influence"] = np.sqrt((influence**2).sum(axis=1))
    influence.index = participants
    
    return summary_df, influence
test = df_simple3.groupby('timepoint')[['ndq_total','haeufigkeit']].corr()
print(test)

# -------------------------------
# Run jackknife for NDQ pre-post
# -------------------------------
summary_ndq2, influence_ndq = jackknife_lmm(df_simple3, "ndq_total ~ group * phase", "VPCode_norm")
print("\nJackknife Summary NDQ (Baseline vs PostIntervention vs PostControl):")
print(summary_ndq2)
#print("\nParticipant Influence NDQ:")
#print(influence_ndq["total_influence"].sort_values(ascending=False))

# -------------------------------
# Run jackknife for Nightmare Frequency pre-post
# -------------------------------
summary_nf2, influence_nf = jackknife_lmm(df_simple3, "haeufigkeit ~ group * phase", "VPCode_norm")
print("\nJackknife Summary Nightmare Frequency (Baseline vs PostIntervention vs PostControl):")
print(summary_nf2)
#print("\nParticipant Influence Nightmare Frequency:")
#print(influence_nf["total_influence"].sort_values(ascending=False))

# ------------------------
# 10. save everything to a pdf
# ------------------------
with PdfPages('analysis_results.pdf') as pdf:
   
    # === 1. NDQ LMM Summary ===
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    ax.text(0, 1, str(result.summary()), fontsize=8, fontfamily='monospace', verticalalignment='top')
    pdf.savefig(fig)
    plt.close(fig)

    # === 2. Nightmare Frequency LMM Summary ===
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    ax.text(0, 1, str(model_nf.summary()), fontsize=8, fontfamily='monospace', verticalalignment='top')
    pdf.savefig(fig)
    plt.close(fig)

    # === 3. NDQ Jackknife Table ===
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    table = ax.table(cellText=summary_ndq.round(3).values,
                     colLabels=summary_ndq.columns,
                     rowLabels=summary_ndq.index,
                     loc='center')
    pdf.savefig(fig)
    plt.close(fig)

    # === 4. NF Jackknife Table ===
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    table = ax.table(cellText=summary_nf.round(3).values,
                     colLabels=summary_nf.columns,
                     rowLabels=summary_nf.index,
                     loc='center')
    pdf.savefig(fig)
    plt.close(fig)

    # === 5. NDQ Plot Over Time ===
    fig, ax = plt.subplots(figsize=(10,6))
    sns.lineplot(data=df_ndq, x="timepoint", y="ndq_total", hue="group",
                 estimator="mean", errorbar=('ci', 68), marker="o", ax=ax)
    ax.set_title("NDQ Total Score Over Time by Group")
    pdf.savefig(fig)
    plt.close(fig)

    # === 6. NF Plot Over Time ===
    fig, ax = plt.subplots(figsize=(10,6))
    sns.lineplot(data=df_nf, x="timepoint", y="haeufigkeit", hue="group",
                 estimator="mean", errorbar=('ci', 68), marker="o", ax=ax)
    ax.set_title("Nightmare Frequency Over Time by Group")
    pdf.savefig(fig)
    plt.close(fig)

        # === 7. Nightmare Burden Index Plot Over Time ===
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="timepoint",
        y="nightmare_burden",
        hue="group",
        estimator="mean",
        errorbar=("ci", 68),
        marker="o",
        ax=ax
    )
    ax.set_title("Nightmare Burden Index Over Time by Group")
    ax.set_ylabel("Nightmare Burden (z-score units)")
    ax.set_xlabel("Timepoint")

    pdf.savefig(fig)
    plt.close(fig)
        
        
    
    # === 7. NDQ LMM Summary ===
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    ax.text(0, 1, str(summary_ndq2.summary()), fontsize=8, fontfamily='monospace', verticalalignment='top')
    pdf.savefig(fig)
    plt.close(fig)

    # === 8. Nightmare Frequency LMM Summary ===
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    ax.text(0, 1, str(summary_nf2.summary()), fontsize=8, fontfamily='monospace', verticalalignment='top')
    pdf.savefig(fig)
    plt.close(fig)

    
    # === 7. NDQ Pre-Post Jackknife Summary ===
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    table = ax.table(cellText=summary_ndq2.round(3).values,
                     colLabels=summary_ndq2.columns,
                     rowLabels=summary_ndq2.index,
                     loc='center')
    pdf.savefig(fig)
    plt.close(fig)     
    # === 8. NF Pre-Post Jackknife Summary ===
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    table = ax.table(cellText=summary_nf2.round(3).values,
                     colLabels=summary_nf2.columns,
                     rowLabels=summary_nf2.index,
                     loc='center')
    pdf.savefig(fig)
    plt.close(fig)  






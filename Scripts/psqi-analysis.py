import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Load data
# =========================
df = pd.read_csv("Data/Dataset_complete_PSQI.csv")

# =========================
# Helper functions
# =========================

def parse_time(t):
    """Parse HH:MM strings safely, return NaN if invalid"""
    if pd.isna(t):
        return np.nan
    t = str(t).strip()
    try:
        return datetime.strptime(t, "%H:%M")
    except ValueError:
        return np.nan

def compute_time_in_bed(bedtime, waketime):
    """Return hours in bed, handling midnight crossover"""
    bed = parse_time(bedtime)
    wake = parse_time(waketime)
    if pd.isna(bed) or pd.isna(wake):
        return np.nan
    if wake <= bed:
        wake += timedelta(days=1)
    return (wake - bed).seconds / 3600

def recode_latency(minutes):
    if pd.isna(minutes):
        return np.nan
    elif minutes <= 15:
        return 0
    elif minutes <= 30:
        return 1
    elif minutes <= 60:
        return 2
    else:
        return 3

def recode_sleep_duration(hours):
    if pd.isna(hours):
        return np.nan
    elif hours > 7:
        return 0
    elif hours >= 6:
        return 1
    elif hours >= 5:
        return 2
    else:
        return 3

def recode_efficiency(eff):
    if pd.isna(eff):
        return np.nan
    elif eff >= 85:
        return 0
    elif eff >= 75:
        return 1
    elif eff >= 65:
        return 2
    else:
        return 3

# =========================
# Mapping German responses to numeric
# =========================

freq_map = {
    "Während der letzten vier Wochen gar nicht": 0,
    "Weniger als einmal pro Woche": 1,
    "Einmal oder zweimal pro Woche": 2,
    "Dreimal oder häufiger pro Woche": 3
}

subjective_quality_map = {
    "Sehr gut": 0,
    "Ziemlich gut": 1,
    "Ziemlich schlecht": 2,
    "Sehr schlecht": 3
}

daytime_enthusiasm_map = {
    "Kein Problem": 0,
    "Nur ein geringes Problem": 1,
    "Etwas Probleme": 2,
    "Große Probleme": 3
}

# =========================
# Recode categorical columns
# =========================

# Sleep disturbances (Q5b–Q5j + Q5)
psqi5_cols = ["psqi5","psqi5b","psqi5c","psqi5d","psqi5e",
              "psqi5f","psqi5g","psqi5h","psqi5i","psqi5j"]
for col in psqi5_cols:
    if col in df.columns:
        df[col] = df[col].map(freq_map)

# C1 – Subjective sleep quality (Q6)
if "psqi6" in df.columns:
    df["psqi6"] = df["psqi6"].map(subjective_quality_map)

# C6 – Sleep medication (Q7)
if "psqi7" in df.columns:
    df["psqi7"] = df["psqi7"].map(freq_map)

# C7 – Daytime dysfunction (Q8 + Q9)
if "psqi8" in df.columns:
    df["psqi8"] = df["psqi8"].map(freq_map)
if "psqi9" in df.columns:
    df["psqi9"] = df["psqi9"].map(daytime_enthusiasm_map)

# Ensure numeric for latency and duration
df["psqi2"] = pd.to_numeric(df["psqi2"], errors="coerce")  # sleep latency

# Fix decimal commas in psqi4
df["psqi4"] = df["psqi4"].astype(str).str.replace(",", ".", regex=False)
df["psqi4"] = pd.to_numeric(df["psqi4"], errors="coerce")

# =========================
# Compute time in bed
# =========================
df["time_in_bed"] = df.apply(lambda r: compute_time_in_bed(r.get("psqi1"), r.get("psqi3")), axis=1)
df["sleep_efficiency"] = (df["psqi4"] / df["time_in_bed"]) * 100

# =========================
# Compute PSQI components
# =========================

# C1
df["C1"] = df["psqi6"]

# C2 – Sleep latency
df["latency_score"] = df["psqi2"].apply(recode_latency)
if "psqi5" in df.columns:
    df["latency_sum"] = df["latency_score"] + df["psqi5"]
else:
    df["latency_sum"] = df["latency_score"]
df["C2"] = df["latency_sum"].apply(
    lambda x: np.nan if pd.isna(x) else 0 if x == 0 else 1 if x <= 2 else 2 if x <= 4 else 3
)

# C3 – Sleep duration
df["C3"] = df["psqi4"].apply(recode_sleep_duration)

# C4 – Sleep efficiency
df["C4"] = df["sleep_efficiency"].apply(recode_efficiency)

# C5 – Sleep disturbances
df["disturbance_sum"] = df[psqi5_cols].sum(axis=1)
df["C5"] = df["disturbance_sum"].apply(
    lambda x: np.nan if pd.isna(x) else 0 if x == 0 else 1 if x <= 9 else 2 if x <= 18 else 3
)

# C6 – Sleep medication
df["C6"] = df["psqi7"]

# C7 – Daytime dysfunction
df["daytime_sum"] = df[["psqi8","psqi9"]].sum(axis=1)
df["C7"] = df["daytime_sum"].apply(
    lambda x: np.nan if pd.isna(x) else 0 if x == 0 else 1 if x <= 2 else 2 if x <= 4 else 3
)

# =========================
# Global PSQI and poor sleeper
# =========================
df["PSQI_global"] = df[["C1","C2","C3","C4","C5","C6","C7"]].sum(axis=1)
df["poor_sleeper"] = (df["PSQI_global"] > 5).astype(int)

# =========================
# Warn about missing components
# =========================
for c in ["C1","C2","C3","C4","C5","C6","C7"]:
    missing = df[c].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} missing values in {c}")

# =========================
# Save results
# =========================
df.to_csv("PSQI_scored_results.csv", index=False)
print("PSQI scoring complete. Results saved to PSQI_scored_results.csv")

# Optional: quick summary
print(df[["C1","C2","C3","C4","C5","C6","C7","PSQI_global","poor_sleeper"]].describe())


# =========================
# Mixed Linear Model Analysis
# =========================
df['timepoint'] = pd.Categorical(
    df['timepoint'],
    categories=['t0','t1','t2','t3','t4'],
    ordered=True
)

df['group'] = pd.Categorical(
    df['group'],
    categories=['Control','Intervention']
)


# Random intercept für Teilnehmer
model_psqi = smf.mixedlm(
    "PSQI_global ~ group * timepoint",   # fixed effects
    df,                                  # DataFrame
    groups=df["VPCode_norm"]             # random intercepts pro Teilnehmer
)
result_psqi = model_psqi.fit(reml=False)

# Ergebnis anzeigen
print(result_psqi.summary())


# Setze Seaborn Style
sns.set(style="whitegrid")
# Descriptive statistics for PSQI
psqi_table = (
    df.groupby(['group', 'timepoint'])['PSQI_global']
      .agg(
          Mean='mean',
          SD='std',
          N='count'
      )
      .reset_index()
)

# Optional: round for publication
psqi_table[['Mean', 'SD']] = psqi_table[['Mean', 'SD']].round(2)

print(psqi_table)
# =========================
# 1️⃣ Trajektorien PSQI_global
# =========================

# Mittelwert + SD pro Gruppe und Zeitpunkt
summary = df.groupby(['group', 'timepoint'])['PSQI_global'].agg(['mean','std','count']).reset_index()
summary['sem'] = summary['std'] / np.sqrt(summary['count'])  # Standard Error

# Plot
plt.figure(figsize=(10,6))
for g in summary['group'].unique():
    data = summary[summary['group']==g]
    plt.plot(data['timepoint'], data['mean'], marker='o', label=g)
    plt.fill_between(data['timepoint'],
                     data['mean'] - data['sem'],  # 95% CI approx: mean ± 1.96*SEM
                     data['mean'] + data['sem'],
                     alpha=0.2)

plt.xlabel("Timepoint")
plt.ylabel("PSQI Global")
plt.title("PSQI Global Trajectories per Group")
plt.legend(title="Group")
# Speichern statt show
plt.savefig("PSQI_Global_Trajectories.png", dpi=300)
plt.close()
# =========================
# 2️⃣ Trajektorien für Subskalen C1–C7
# =========================

components = ['C1','C2','C3','C4','C5','C6','C7']

for comp in components:
    summary_c = df.groupby(['group','timepoint'])[comp].agg(['mean','std','count']).reset_index()
    summary_c['sem'] = summary_c['std'] / np.sqrt(summary_c['count'])
    
    plt.figure(figsize=(10,6))
    for g in summary_c['group'].unique():
        data = summary_c[summary_c['group']==g]
        plt.plot(data['timepoint'], data['mean'], marker='o', label=g)
        plt.fill_between(data['timepoint'],
                         data['mean'] - data['sem'],
                         data['mean'] + data['sem'],
                         alpha=0.2)
    plt.xlabel("Timepoint")
    plt.ylabel(f"{comp} Score")
    plt.title(f"Trajectories of PSQI Component {comp} per Group")
    plt.legend(title="Group")
    # Speichern statt show
    #plt.savefig(f"PSQI_Component_{comp}_Trajectories.png", dpi=300)
    plt.close()


components = ['C1','C2','C3','C4','C5','C6','C7']
results = {}

components = ['C1','C2','C3','C4','C5','C6','C7']

for comp in components:
    df_comp = df.dropna(subset=[comp, "group", "timepoint", "VPCode_norm"])

    # Check that both factors have ≥2 levels
    if df_comp['timepoint'].nunique() < 2 or df_comp['group'].nunique() < 2:
        print(f"Skipping {comp}: insufficient factor levels")
        continue

    model = smf.mixedlm(
        f"{comp} ~ group * timepoint",
        df_comp,
        groups=df_comp["VPCode_norm"]
    )

    res = model.fit(reml=False)

    print(f"\n=== Ergebnisse für {comp} ===")
    print(res.summary())


df_simple3 = df.copy() 
# 1. Create a new condition variable
def condition_label(row):
    if row['timepoint'] == 't0':
        return 'Baseline'
    elif row['group'] == 'Intervention' and row['timepoint'] in ['t1', 't2', 't3', 't4']:
        return 'PostIntervention'
    elif row['group'] == 'Control' and row['timepoint'] in ['t1', 't2', 't3', 't4']:
        return 'PostControl'
    else:
        return np.nan
    
df_simple3['condition'] = df_simple3.apply(condition_label, axis=1)

model_condition_ndq = smf.mixedlm(
    "PSQI_global ~ C(condition)",   # categorical fixed effect
    df_simple3,
    groups=df_simple3["VPCode_norm"]
).fit(reml=False)
print("\n=== Condition Model (Baseline vs PostIntervention vs PostControl)Distress ===")
print(model_condition_ndq.summary())

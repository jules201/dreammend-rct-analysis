import pandas as pd
import re

from scipy.stats import chi2_contingency

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import chi2_contingency

# CSV-Datei einlesen
demographics_filtered = pd.read_csv("Data/Demographics_filtered.csv", encoding='utf-8')
#TODO check if we use the same data as for the LMM statistics i.e. used in descriptive_st.py


print(f"✔️ Using {len(demographics_filtered)} participants for demographics tables.")


# ------------------------------------------
# 📊 Häufigkeitsauswertung (auf demographics_df!)
# ------------------------------------------
demographic_columns = [col for col in demographics_filtered.columns if col != "VPCode"]

for col in demographic_columns:
    print(f"\n📊 Häufigkeiten für: {col}")
    print(demographics_filtered[col].value_counts(dropna=False))


# List of columns to summarize (exclude VPCode_norm and Gruppe)
demographic_columns = [col for col in demographics_filtered.columns if col not in ["VPCode", "VPCode_norm", "Gruppe", 'Alter']]
# Map age ranges to numeric midpoints
age_map = {
    "18 - 24": 21,
    "25 - 34": 29.5,
    "35 - 44": 39.5,
    "45 - 54": 49.5,
    "55 und älter": 60
}

demographics_filtered = demographics_filtered.copy()
demographics_filtered['Alter_num'] = demographics_filtered['Alter'].map(age_map)
# Ensure demographics_filtered is a proper copy to avoid warnings
demographics_filtered = demographics_filtered.copy()

# Numeric summary for Age
age_summary = demographics_filtered.groupby("Gruppe")["Alter_num"].agg(['mean', 'std'])
age_summary_formatted = age_summary.apply(lambda x: f"{x['mean']:.1f} ± {x['std']:.1f}", axis=1)

# Categorical summary for Gender
gender_summary = pd.crosstab(demographics_filtered['Geschlecht'], demographics_filtered['Gruppe'])

# Display
print("📊 Age (mean ± SD) per group:")
print(age_summary_formatted)

print("\n📊 Gender counts per group:")
print(gender_summary)

vergleichsvariablen = [
    "Alter", "Geschlecht", "Abschluss", "Beschaeftigung",
    "Alkoholkonsum", "Cannabiskonsum", "Alptraumhaeufigkeit", "Belastung"
]

# PDF-Datei zum Speichern öffnen
with PdfPages("kreuztabellen_chi2_ergebnisse.pdf") as pdf:
    for var in vergleichsvariablen:
        # Kreuztabelle berechnen
        table = pd.crosstab(demographics_df[var], demographics_df["Gruppe"], margins=True)

        # Chi²-Test (nur echte Werte, ohne Summenzeile/-spalte)
        chi2, p, dof, expected = chi2_contingency(table.iloc[:-1, :-1])

        # Diagramm vorbereiten
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')

        # Tabelle als Liste aufbereiten
        table_data = [[str(cell) for cell in row] for row in table.reset_index().values]
        columns = [table.index.name if table.index.name else var] + list(table.columns)

        # Tabelle in Plot einfügen
        table_display = ax.table(
            cellText=table_data,
            colLabels=columns,
            loc='center',
            cellLoc='center'
        )

        # Titel setzen
        ax.set_title(f"{var} nach Gruppe", fontsize=14, pad=20)

        # Chi²-Testergebnisse unter der Tabelle anzeigen
        plt.figtext(0.1, 0.05, f"Chi² = {chi2:.2f}, p-Wert = {p:.4f}, Freiheitsgrade = {dof}", fontsize=10)

        # Seite zum PDF hinzufügen
        pdf.savefig(fig)
        plt.close()

print("✅ PDF 'kreuztabellen_chi2_ergebnisse.pdf' erfolgreich erstellt.")

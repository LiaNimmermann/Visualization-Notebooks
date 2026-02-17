import pandas as pd
import os

# === Einstellungen ===
data_path = '/home/lstracke/Visualization-Notebooks/Data-Visualization/zero_shot/val_comparison_preprocessing_trained_zero_shot.csv'
zero_shot_bool = True
architecture = 'upernet'
dataset = 'acdc_night'
preprocessing = ['grayscale', 'color-opponency', 'single-color']

# Optional: Zero-shot Präprocessing anpassen
if zero_shot_bool:
    preprocessing = [p + '_zero_shot' for p in preprocessing]

# === CSV laden ===
colnames = ['architecture','dataset','preprocessing','sparsity',
            'mIoU_mean','mIoU_std','mAcc_mean','mAcc_std','aAcc_mean','aAcc_std']
if os.path.exists(data_path):
    df = pd.read_csv(data_path, header=None, names=colnames)
else:
    raise FileNotFoundError(f"CSV-Datei nicht gefunden: {data_path}")

# Filter für gewählte Architektur, Dataset und Preprocessing
df = df[(df['architecture']==architecture) & 
        (df['dataset']==dataset) &
        (df['preprocessing'].isin(preprocessing))].copy()

# Sparsity als int
df['sparsity'] = df['sparsity'].astype(int)
# Lesbarer Präprocessing-Name
df['preprocessing_display'] = df['preprocessing'].str.replace('_zero_shot','', regex=False)

# Sparsity-Reihenfolge: 0 -> descending
present = sorted(df['sparsity'].unique()) #, reverse=True)
order = [0] + sorted([s for s in present if s != 0]) #, reverse=True)

# Funktion für mean ± std
num_cols = ['mIoU_mean','mIoU_std','mAcc_mean','mAcc_std','aAcc_mean','aAcc_std']
for col in num_cols:
    df[col] = df[col].astype(float)

# --- Formatierungsfunktion ---
def fmt(mean, std):
    return f"{float(mean):.2f} ~$\\pm$~ {float(std):.2f}"

# === LaTeX Header ===
header = rf"""\begin{{table}}[h]
  \centering
  \caption{{\textbf{{Zero-shot}} evaluation of the {architecture} architecture trained on Cityscapes without sparsity and \textbf{{validated on {dataset}}} at different sparsities. Metrics reported are mean Intersection over Union (mIoU), mean Accuracy (mAcc), and average Accuracy (aAcc).}}
  \begin{{tabular}}{{@{{}}lccc ccc@{{}}}}
    \toprule
    Architecture & Preprocessing & Sparsity & mIoU & mAcc & aAcc \\
    \midrule"""

# === LaTeX Zeilen ===
rows = []
for s in order:
    sparsity_display = '-' if s == 0 else f"{s}\\%"
    for i, p in enumerate(preprocessing):
        sel = df[(df['preprocessing']==p) & (df['sparsity']==s)]
        if sel.empty:
            mIoU = mIoU_std = mAcc = mAcc_std = aAcc = aAcc_std = float('nan')
        else:
            row = sel.iloc[0]
            mIoU, mIoU_std = row['mIoU_mean'], row['mIoU_std']
            mAcc, mAcc_std = row['mAcc_mean'], row['mAcc_std']
            aAcc, aAcc_std = row['aAcc_mean'], row['aAcc_std']

        line = (f"    \\multirow{{{len(preprocessing)}}}{{*}}{{{architecture}}} & {p.replace('_zero_shot','')} & "
                f"\\multirow{{{len(preprocessing)}}}{{*}}{{{sparsity_display}}} & "
                f"{fmt(mIoU,mIoU_std)} & {fmt(mAcc,mAcc_std)} & {fmt(aAcc,aAcc_std)} \\\\" if i==0 else
                f"    & {p.replace('_zero_shot','')} &  & {fmt(mIoU,mIoU_std)} & {fmt(mAcc,mAcc_std)} & {fmt(aAcc,aAcc_std)} \\\\")
        rows.append(line)
    rows.append("    \\midrule")

# === Footer ===
footer = rf"""    \bottomrule
  \end{{tabular}}
  \label{{tab:sparse_results_{architecture}_{dataset}}}
\end{{table}}"""

# === Ganze Tabelle zusammenfügen ===
latex_table = "\n".join([header] + rows + [footer])

# === In Datei speichern ===
out_path = f"generated_table_for_{architecture}_{dataset}.tex"
if zero_shot_bool:
    out_path = f"generated_table_for_{architecture}_{dataset}_zero_shot.tex"
with open(out_path, "w") as f:
    f.write(latex_table)

print(f"LaTeX-Tabelle erfolgreich generiert: {out_path}")

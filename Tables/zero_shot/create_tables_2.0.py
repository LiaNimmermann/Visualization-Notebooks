import pandas as pd

# === Einstellungen ===
data_path = "/home/lstracke/blur_preprocessing/blur_preprocessing_git/work_dir_zero_shot/metrics_averaged_over_five_seed_dataset.csv"

# Reihenfolge der Preprocessings + Anzeige Namen
preprocessing_map = {
    "bl": "-",
    "bw": "Luminance",
    "co": "Color-opponency",
    "sc": "Single-color"
}

for architecture in ["segformer", "upernet", "deeplabv3plus", "mask2former"]:
    for dataset in ["cityscapes", "dark_zurich", "acdc_night", "acdc_fog", "acdc_rain", "acdc_snow", "acdc_full"]:

        preprocessing = list(preprocessing_map.keys())

        # === CSV laden ===
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()

        # Architektur-Name für LaTeX
        arch_name_map = {
            "segformer": "SegFormer",
            "deeplabv3plus": "DeepLabv3+",
            "upernet": "UPerNet",
            "mask2former": "Mask2Former"
        }
        architecture_name = arch_name_map[architecture]

        # Filter nach Architektur, Dataset und Preprocessing
        df = df[
            (df["Architecture"] == architecture) &
            (df["Dataset"] == dataset) &
            (df["Preprocessing"].isin(preprocessing))
        ].copy()

        # Datentypen konvertieren
        df["Sparsity"] = pd.to_numeric(df["Sparsity"], errors="coerce")
        for col in ["mIoU","mIoU_std","mAcc","mAcc_std","aAcc","aAcc_std"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sparsity Reihenfolge
        sparsity_order = sorted(df["Sparsity"].unique())

        # === Formatierungsfunktion ===
        def fmt(mean, std):
            return f"{mean:.2f}~$\\pm$~{std:.2f}"

        # === LaTeX-Header ===
        header = rf"""\begin{{table}}[h]
\centering
\caption{{Results for the {architecture} architecture validated on {dataset} at different sparsities.}}
\begin{{tabular}}{{@{{}}lccc ccc@{{}}}}
\toprule
Architecture & Preprocessing & Sparsity & mIoU & mAcc & aAcc \\
\midrule
"""

        rows = []

        for s in sparsity_order:
            sparsity_display = "-" if s == 0 else f"{int(s*100)}\\%"

            for i, p in enumerate(preprocessing):

                if p == "bw":
                    sel = df[
                        (df["Preprocessing"] == p) &
                        (df["Sparsity"] == s) &
                        (df["Validated"] == "bw")  
                    ]
                elif p == "co":
                    sel = df[
                        (df["Preprocessing"] == p) &
                        (df["Sparsity"] == s) &
                        (df["Validated"] == "co")   
                    ]
                elif p == "sc":
                    sel = df[
                        (df["Preprocessing"] == p) &
                        (df["Sparsity"] == s) &
                        (df["Validated"] == "sc")  
                    ]
                else:
                    sel = df[
                        (df["Preprocessing"] == p) &
                        (df["Sparsity"] == s)
                    ]

                if sel.empty:
                    continue

                row = sel.iloc[0]

                mIoU = fmt(row.mIoU,row.mIoU_std)
                mAcc = fmt(row.mAcc,row.mAcc_std)
                aAcc = fmt(row.aAcc,row.aAcc_std)

                prep_name = preprocessing_map[p]

                if i == 0:
                    line = (
                        f"\\multirow{{{len(preprocessing)}}}{{*}}{{{architecture_name}}} & {prep_name} & "
                        f"\\multirow{{{len(preprocessing)}}}{{*}}{{{sparsity_display}}} & "
                        f"{mIoU} & {mAcc} & {aAcc} \\\\"
                    )
                else:
                    line = f"& {prep_name} & & {mIoU} & {mAcc} & {aAcc} \\\\"

                rows.append(line)

            rows.append("\\midrule")

        # === LaTeX-Footer ===
        footer = rf"""\bottomrule
\end{{tabular}}
\label{{tab:sparse_results_{architecture}_{dataset}}}
\end{{table}}"""

        # === LaTeX-Tabelle speichern ===
        latex_table = "\n".join([header] + rows + [footer])
        out_path = f"generated_table_for_{architecture}_{dataset}_zero_shot_five_seeds.tex"

        with open(out_path, "w") as f:
            f.write(latex_table)

        print("LaTeX Tabelle generiert:", out_path)

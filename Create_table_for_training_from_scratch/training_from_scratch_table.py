import pandas as pd

# ==============================
# CSV laden
# ==============================
csv_path = "/home/lstracke/Visualization-Notebooks/Create_table_for_training_from_scratch/aggregated_results_trained_from_scratch_clean.csv"
df = pd.read_csv(csv_path)

# ==============================
# Gewünschte Dataset-Reihenfolge
# ==============================
dataset_order = [
    "cityscapes",
    "dark_zurich",
    "acdc_night",
    "acdc_fog",
    "acdc_rain",
    "acdc_snow"
]

# ==============================
# Gewünschte Preprocessing-Reihenfolge
# baseline zuerst
# ==============================
preprocessing_order = ["baseline", "greyscale", "color_opponency", "single_color"]

rows = []

for dataset in dataset_order:

    metrics = [
        f"{dataset}_mIoU",
        f"{dataset}_mAcc",
        f"{dataset}_aAcc"
    ]

    if not all(m in df.columns for m in metrics):
        continue

    grouped = (
        df.groupby(["model_name", "preprocessing_method"])[metrics]
        .agg(["mean", lambda x: x.std(ddof=1)])
    )

    grouped.columns = [
        "_".join(col).replace("<lambda_0>", "std")
        for col in grouped.columns
    ]
    grouped = grouped.reset_index()

    # Sortiere nach gewünschter Preprocessing-Reihenfolge
    grouped["preprocessing_order"] = grouped["preprocessing_method"].apply(
        lambda x: preprocessing_order.index(x) if x in preprocessing_order else 99
    )
    grouped = grouped.sort_values(["model_name", "preprocessing_order"])

    for _, row in grouped.iterrows():
        preprocessing = row["preprocessing_method"]
        depth = 0 if preprocessing == "baseline" else 5

        rows.append({
            "Architecture": row["model_name"],
            "Dataset": dataset,
            "Preprocessing": preprocessing,
            "Depth": depth,
            "Channel": 3,
            "mIoU (\\%)": f"{row[f'{dataset}_mIoU_mean']:.2f} $\\pm$ {row[f'{dataset}_mIoU_std']:.2f}",
            "mAcc (\\%)": f"{row[f'{dataset}_mAcc_mean']:.2f} $\\pm$ {row[f'{dataset}_mAcc_std']:.2f}",
            "aAcc (\\%)": f"{row[f'{dataset}_aAcc_mean']:.2f} $\\pm$ {row[f'{dataset}_aAcc_std']:.2f}",
        })

latex_df = pd.DataFrame(rows)

# ==============================
# LaTeX Tabelle erzeugen
# ==============================
latex_table = latex_df.to_latex(
    index=False,
    escape=False,
    column_format="lllccccc"
)

print(latex_table)
output_file = "results_table_shashank.tex"

with open(output_file, "w") as f:
    f.write(latex_table)

print(f"LaTeX table saved as '{output_file}' in current directory.")
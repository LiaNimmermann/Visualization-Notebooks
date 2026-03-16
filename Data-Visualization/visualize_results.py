import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from io import StringIO
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ===================
# LaTeX-Tabellen als Dict
# ===================
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

# Pfad zu Times New Roman (auf deinem System anpassen!)
font_path = "/home/lstracke/Data/times.ttf"  # Beispiel Linux-Pfad
times_new_roman = FontProperties(fname=font_path, size=16)



group_width = 0.8
sns.set(style="whitegrid")
#TODO:black white depth search

# Reihenfolge der Datasets
datasets = [
    "cityscapes",
    "dark_zurich",
    "acdc_night",
    "acdc_snow",
    "acdc_fog",
    "acdc_rain",
]

# Linien (Methoden)
lines = [
    ("baseline", "Baseline"),
    ("bw_average_3channel", "Black white average (3 channel)"),
    ("bw_average_1channel", "Black white average (1 channel)"),
    ("bw_green_bias_3channel", "Black white green bias (3 channel)"),
    ("bw_green_bias_1channel", "Black white green bias (1 channel)"),
]

# Farben (gut unterscheidbar)
"""
color_map = {
    "baseline": "#505A5B",               # Schwarz
    "bw_average_3channel": "#7775A9",    # Blau
    "bw_average_1channel": "#A8A7C8",    # Grün
    "bw_green_bias_3channel": "#62A87C", # Rot
    "bw_green_bias_1channel": "#70C47F", # Lila
}"""
palette = sns.color_palette("colorblind", n_colors=len(lines))
color_map = {line[0]: palette[i] for i, line in enumerate(lines)}

# Mittelwerte und Standardabweichungen der mIoU
data = {
    "cityscapes": {
        "baseline": (59.09, 0.30),
        "bw_average_3channel": (59.08, 1.82),
        "bw_average_1channel": (57.61, 0.57),
        "bw_green_bias_3channel": (57.51, 0.40),
        "bw_green_bias_1channel": (56.16, 1.60),
    },
    "dark_zurich": {
        "baseline": (9.56, 1.13),
        "bw_average_3channel": (17.10, 1.87),
        "bw_average_1channel": (17.64, 0.63),
        "bw_green_bias_3channel": (16.82, 0.63),
        "bw_green_bias_1channel": (19.14, 2.64),
    },
    "acdc_night": {
        "baseline": (10.77, 1.72),
        "bw_average_3channel": (18.90, 2.11),
        "bw_average_1channel": (19.20, 0.76),
        "bw_green_bias_3channel": (19.06, 0.68),
        "bw_green_bias_1channel": (20.06, 1.21),
    },
    "acdc_snow": {
        "baseline": (25.31, 0.53),
        "bw_average_3channel": (34.69, 0.60),
        "bw_average_1channel": (31.56, 0.60),
        "bw_green_bias_3channel": (32.55, 0.11),
        "bw_green_bias_1channel": (31.54, 0.36),
    },
    "acdc_rain": {
        "baseline": (32.22, 0.29),
        "bw_average_3channel": (37.13, 1.23),
        "bw_average_1channel": (35.06, 1.17),
        "bw_green_bias_3channel": (34.55, 2.19),
        "bw_green_bias_1channel": (33.52, 1.81),
    },
    "acdc_fog": {
        "baseline": (36.09, 1.90),
        "bw_average_3channel": (47.16, 1.49),
        "bw_average_1channel": (43.54, 1.06),
        "bw_green_bias_3channel": (44.04, 0.96),
        "bw_green_bias_1channel": (41.96, 0.41),
    },
}


x = np.arange(len(datasets))
bar_width = group_width / len(lines)

fig, ax = plt.subplots(figsize=(14, 6))

for i, (line_key, line_label) in enumerate(lines):
    means, stds = zip(*[data[ds][line_key] for ds in datasets])
    offset = (i - len(lines)/2) * bar_width + bar_width / 2
    positions = x + offset
    ax.bar(positions, means, yerr=stds, width=bar_width,
           label=line_label, color=color_map[line_key],
           capsize=4, linewidth=0)

# Achsen beschriften und gestalten
ax.set_xticks(x)
ax.set_xticklabels(["Cityscapes", "Dark Zurich", "ACDC night", "ACDC snow", "ACDC fog", "ACDC rain",], rotation=0, fontproperties=times_new_roman)
ax.set_xlabel("Dataset", fontproperties=times_new_roman)
ax.set_ylabel("mIoU (%)", fontproperties=times_new_roman)
ax.set_title("Deeplabv3+ (ResNet18) - Black/White validation",fontproperties=times_new_roman)
legend = ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

print(ax.get_xticklabels()[0].get_fontproperties().get_size())

for text in legend.get_texts():
    text.set_fontproperties(times_new_roman)

legend.get_title().set_fontproperties(times_new_roman)
ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
fig.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("visualization/depth_search_black_white.pdf", dpi=800)
print("Font used for xlabel:", ax.xaxis.get_label().get_fontname())
plt.show()


#TODO: architectures black-white

# Dataset-Namen in der Reihenfolge der Validierungen
datasets = ["cityscapes", "dark_zurich", "acdc_night", "acdc_snow", "acdc_fog", "acdc_rain"]
x = np.arange(len(datasets))
# Beispielhafte Daten extrahiert aus den Tabellen
# Format: data[dataset][network][preprocessing_channel] = (mean, std)
data = {
    "acdc_fog": {
        "deeplabv3plus": {
            "bw_average_3channel": (49.84, 2.31),
            "bw_average_1channel": (52.60, 1.94),
            "bw_green_bias_3channel": (47.49, 3.24),
            "bw_green_bias_1channel": (51.88, 1.48),
            "baseline": (39.20, 4.35 ),
        },
        "segformer": {
            "bw_average_3channel": (57.64, 0.08),
            "bw_average_1channel": (56.75, 1.48),
            "bw_green_bias_3channel": (58.77, 2.45),
            "bw_green_bias_1channel": (55.76, 0.43),
            "baseline": (61.71, 0.79),
        },
        "internImage": {
            "bw_average_3channel": (47.19, 2.61),
            "bw_average_1channel": (49.41, 2.44),
            "bw_green_bias_3channel": (47.93, 2.64),
            "bw_green_bias_1channel": (48.98, 1.06),
            "baseline": (40.96, 1.73),
        },
    },
    "acdc_night": {
        "deeplabv3plus": {
            "bw_average_3channel": (21.87, 0.71),
            "bw_average_1channel": (21.62, 0.58),
            "bw_green_bias_3channel": (20.74, 1.91),
            "bw_green_bias_1channel": (22.48, 0.71),
            "baseline": (8.01, 0.45),
        },
        "segformer": {
            "bw_average_3channel": (23.78, 0.10),
            "bw_average_1channel": (22.34, 0.32),
            "bw_green_bias_3channel": (23.19, 1.88),
            "bw_green_bias_1channel": (23.72, 1.55),
            "baseline": (15.37, 1.58 ),
        },
        "internImage": {
            "bw_average_3channel": (18.70, 0.75),
            "bw_average_1channel": (20.31, 0.87),
            "bw_green_bias_3channel": (20.81, 1.67),
            "bw_green_bias_1channel": (20.55, 0.94),
            "baseline": (9.64, 1.84),
        },
    },
    "acdc_rain": {
        "deeplabv3plus": {
            "bw_average_3channel": (37.73, 1.37),
            "bw_average_1channel": (39.12, 1.13),
            "bw_green_bias_3channel": (34.45, 4.04),
            "bw_green_bias_1channel": (38.60, 0.79),
            "baseline": (33.31, 2.15),
        },
        "segformer": {
            "bw_average_3channel": (37.43, 0.35),
            "bw_average_1channel": (38.05, 1.67),
            "bw_green_bias_3channel": (39.69, 0.45),
            "bw_green_bias_1channel": (38.93, 0.16),
            "baseline": (47.67, 2.45 ),
        },
        "internImage": {
            "bw_average_3channel": (32.54, 0.68),
            "bw_average_1channel": (32.60, 1.39),
            "bw_green_bias_3channel": (30.20, 2.71),
            "bw_green_bias_1channel": (33.20, 1.41),
            "baseline": (31.62, 2.19),
        },
    },
    "acdc_snow": {
        "deeplabv3plus": {
            "bw_average_3channel": (36.34, 1.03),
            "bw_average_1channel": (37.75, 1.66),
            "bw_green_bias_3channel": (33.79, 2.96),
            "bw_green_bias_1channel": (36.48, 0.17),
            "baseline": (24.18, 3.60),
        },
        "segformer": {
            "bw_average_3channel": (42.87, 0.94),
            "bw_average_1channel": (41.19, 0.69),
            "bw_green_bias_3channel": (43.17, 0.53),
            "bw_green_bias_1channel": (42.55, 0.78),
            "baseline": (45.35, 0.38 ),
        },
        "internImage": {
            "bw_average_3channel": (30.19, 1.33),
            "bw_average_1channel": (32.40, 3.65),
            "bw_green_bias_3channel": (29.07, 0.99),
            "bw_green_bias_1channel": (33.47, 0.81),
            "baseline": (26.90, 1.92),
        },
    },
    "cityscapes": {
        "deeplabv3plus": {
            "bw_average_3channel": (63.25, 2.45),
            "bw_average_1channel": (66.02, 2.18),
            "bw_green_bias_3channel": (64.97, 1.19),
            "bw_green_bias_1channel": (65.14, 2.13),
            "baseline": (66.68, 0.44),
        },
        "segformer": {
            "bw_average_3channel": (69.67, 0.23),
            "bw_average_1channel": (68.67, 0.79),
            "bw_green_bias_3channel": (69.40, 0.22),
            "bw_green_bias_1channel": (68.94, 0.82),
            "baseline": (72.65, 0.08),
        },
        "internImage": {
            "bw_average_3channel": (62.84, 0.41),
            "bw_average_1channel": (63.87, 4.03),
            "bw_green_bias_3channel": (62.51, 2.38),
            "bw_green_bias_1channel": (60.90, 0.30),
            "baseline": (61.87, 0.21),
            
        },
    },
    "dark_zurich": {
        "deeplabv3plus": {
            "bw_average_3channel": (20.79, 0.61),
            "bw_average_1channel": (19.92, 0.55),
            "bw_green_bias_3channel": (19.04, 2.31),
            "bw_green_bias_1channel": (20.89, 0.64),
            "baseline": (7.64, 1.00),
        },
        "segformer": {
            "bw_average_3channel": (22.60, 0.41),
            "bw_average_1channel": (20.53, 0.22),
            "bw_green_bias_3channel": (21.41, 1.09),
            "bw_green_bias_1channel": (20.84, 0.91),
            "baseline": (14.18, 0.84),
        },
        "internImage": {
            "bw_average_3channel": (18.61, 1.09),
            "bw_average_1channel": (19.63, 0.62),
            "bw_green_bias_3channel": (20.34, 1.87),
            "bw_green_bias_1channel": (19.66, 0.51),
            "baseline": (10.07, 1.49),
        },
    },
}

networks = ["deeplabv3plus", "segformer", "internImage"]
network_list = ["Deeplabv3+ (ResNet50)", "SegFormer", "InternImage"]
"""
lines = [
    ("baseline", "Baseline 3ch"),
    ("bw_average_3channel", "bw_average 3ch"),
    ("bw_average_1channel", "bw_average 1ch"),
    ("bw_green_bias_3channel", "bw_green_bias 3ch"),
    ("bw_green_bias_1channel", "bw_green_bias 1ch"),

]

color_map = {
    "baseline": "#505A5B",
    "bw_average_3channel": "#7775A9",     # Blau
    "bw_average_1channel": "#A8A7C8",     # Grün
    "bw_green_bias_3channel": "#62A87C",  # Rot
    "bw_green_bias_1channel": "#70C47F",  # Lila (statt Orange)

}"""
bar_width = group_width / 5
x = np.arange(len(datasets))

for k,network in enumerate(networks):
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (line_key, line_label) in enumerate(lines):
        means = []
        stds = []
        for ds in datasets:
            mean, std = data[ds][network][line_key]
            means.append(mean)
            stds.append(std)

        offset = (i - len(lines) / 2) * bar_width + bar_width / 2
        positions = x + offset

        ax.bar(positions, means,
               yerr=stds,
               capsize=4,
               width=bar_width,
               color=color_map[line_key],
               label=line_label,
               edgecolor=color_map[line_key],
               linewidth=0)

    # Achsenbeschriftung und Gestaltung analog
    ax.set_xticks(x)
    ax.set_xticklabels(["Cityscapes", "Dark Zurich", "ACDC night", "ACDC snow", "ACDC fog", "ACDC rain"],
                       rotation=0, fontproperties=times_new_roman)
    ax.set_xlabel("Dataset", fontproperties=times_new_roman)
    ax.set_ylabel("mIoU (%)", fontproperties=times_new_roman)
    ax.set_title(f"{network_list[k]} - Black/White validation", fontproperties=times_new_roman)
    ax.set_ylim(0, 80)
    print(ax.get_xticklabels()[0].get_fontproperties().get_size())

    # Legende wie gehabt
    legend = ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    for text in legend.get_texts():
        text.set_fontproperties(times_new_roman)
    legend.get_title().set_fontproperties(times_new_roman)

    # Gitter und Layout
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
    fig.tight_layout(rect=[0, 0, 1, 1])

    # Speichern und anzeigen
    plt.savefig(f"visualization/architecture_{network}_d5_black_white_bar.pdf", dpi=800)
    plt.show()


#TODO: here architectures d5 visualization

latex_tables = {
    'cityscapes': r"""
    \begin{table*}[h]
    \begin{tabular}{@{}lcccccc@{}}
    \toprule
    Validation on cityscapes & preprocessing &depth & channel  &  aAcc_mean & mIoU_mean & mAcc_mean \\
    \midrule
    deeplabv3plus_baseline &- & 5& 3 & 94.68 ~$\pm$~ 0.05 & 66.68 ~$\pm$~ 0.44 & 76.58 ~$\pm$~ 0.60 \\
    deeplabv3plus_bw &bw_average & 5& 3 & 94.35 ~$\pm$~ 0.35 & 66.53 ~$\pm$~ 1.11 & 77.86 ~$\pm$~ 0.63 \\
    deeplabv3plus_co &co & 5& 3 & 94.88 ~$\pm$~ 0.15 & 66.64 ~$\pm$~ 1.39 & 75.80 ~$\pm$~ 1.46 \\
    deeplabv3plus_sc &sc & 5& 3 & 94.90 ~$\pm$~ 0.16 & 66.08 ~$\pm$~ 3.05 & 76.33 ~$\pm$~ 3.03 \\
    segformer_baseline &- & 5& 3 & 95.11 ~$\pm$~ 0.02 & 72.65 ~$\pm$~ 0.08 & 81.37 ~$\pm$~ 0.39 \\
    segformer_bw &bw_average & 5& 3 & 93.99 ~$\pm$~ 0.02 & 67.15 ~$\pm$~ 0.29 & 76.94 ~$\pm$~ 0.24 \\
    segformer_co &co & 5& 3 & 94.48 ~$\pm$~ 0.05 & 69.17 ~$\pm$~ 0.58 & 78.60 ~$\pm$~ 0.21 \\
    segformer_sc &sc & 5& 3 & 94.57 ~$\pm$~ 0.09 & 69.71 ~$\pm$~ 0.15 & 79.23 ~$\pm$~ 0.15 \\
    internImage_baseline &- & 5& 3 & 94.01 ~$\pm$~ 0.34 & 61.87 ~$\pm$~ 0.21 & 69.87 ~$\pm$~ 1.49 \\
    internImage_bw &bw_average & 5& 3 & 93.70 ~$\pm$~ 0.044 & 59.50 ~$\pm$~ 0.94 & 67.65 ~$\pm$~ 1.91 \\
    internImage_co &co & 5& 3 & 94.35 ~$\pm$~ 0.12 & 61.93 ~$\pm$~ 0.88 & 70.28 ~$\pm$~ 0.72 \\
    internImage_sc &sc & 5& 3 & 94.45 ~$\pm$~ 0.07 & 63.17 ~$\pm$~ 1.77 & 71.45 ~$\pm$~ 2.05 \\
    \end{tabular}
    """,

    'dark_zurich': r"""
    \begin{table*}[h]
    \begin{tabular}{@{}lcccccc@{}}
    \toprule
    Validation on dark_zurich & preprocessing &depth & channel  &  aAcc_mean & mIoU_mean & mAcc_mean \\
    \midrule
    deeplabv3plus_baseline &- & 5& 3 & 30.13 ~$\pm$~ 1.41 & 7.64 ~$\pm$~ 1.00 & 16.40 ~$\pm$~ 2.74 \\
    deeplabv3plus_bw &bw_average & 5& 3 & 55.35 ~$\pm$~ 1.81 & 18.39 ~$\pm$~ 0.54 & 34.73 ~$\pm$~ 1.68 \\
    deeplabv3plus_co &co & 5& 3 & 50.68 ~$\pm$~ 4.61 & 17.46 ~$\pm$~ 1.70 & 32.32 ~$\pm$~ 2.85 \\
    deeplabv3plus_sc &sc & 5& 3 & 51.35 ~$\pm$~ 4.26 & 18.54 ~$\pm$~ 2.18 & 32.79 ~$\pm$~ 1.49 \\
    segformer_baseline &- & 5& 3 & 47.50 ~$\pm$~ 2.42 & 14.18 ~$\pm$~ 0.84 & 28.51 ~$\pm$~ 0.92 \\
    segformer_bw &bw_average & 5& 3 & 54.26 ~$\pm$~ 1.84 & 21.02 ~$\pm$~ 0.39 & 34.35 ~$\pm$~ 0.43 \\
    segformer_co &co & 5& 3 & 53.56 ~$\pm$~ 0.39 & 18.72 ~$\pm$~ 0.59 & 32.42 ~$\pm$~ 1.29 \\
    segformer_sc &sc & 5& 3 & 53.07 ~$\pm$~ 0.89 & 18.13 ~$\pm$~ 0.69 & 32.34 ~$\pm$~ 0.57 \\
    internImage_baseline &- & 5& 3 & 32.28 ~$\pm$~ 5.55 & 10.07 ~$\pm$~ 1.49 & 17.46 ~$\pm$~ 2.39 \\
    internImage_bw &bw_average & 5& 3 & 53.40 ~$\pm$~ 2.25 & 17.66 ~$\pm$~ 0.47 & 29.58 ~$\pm$~ 0.75 \\
    internImage_co &co & 5& 3 & 40.99 ~$\pm$~ 9.39 & 14.31 ~$\pm$~ 1.88 & 26.85 ~$\pm$~ 1.97 \\
    internImage_sc &sc & 5& 3 & 41.43 ~$\pm$~ 6.72 & 13.47 ~$\pm$~ 2.45 & 25.99 ~$\pm$~ 2.23 \\
    \end{tabular}
    """,

    'acdc_night': r"""
    \begin{table*}[h]
    \centering
    \begin{tabular}{@{}lcccccc@{}}
    \toprule
    Validation on acdc\_night & preprocessing &depth & channel  &  aAcc\_mean & mIoU\_mean & mAcc\_mean \\
    \midrule
    deeplabv3plus\_baseline &- & 5& 3 & 31.94 ~$\pm$~ 2.20 & 8.01  ~$\pm$~ 0.45 & 18.14 ~$\pm$~ 2.24 \\
    deeplabv3plus\_bw &bw\_average & 5& 3 & 58.90 ~$\pm$~ 1.33 & \textbf{19.76} ~$\pm$~ 0.53 & 34.13 ~$\pm$~ 1.05 \\
    deeplabv3plus\_co &co & 5& 3 & 53.51 ~$\pm$~ 3.20 & 18.05 ~$\pm$~ 1.55 & 31.73 ~$\pm$~ 2.54 \\
    deeplabv3plus\_sc &sc & 5& 3 & 54.79 ~$\pm$~ 2.95 & 19.75 ~$\pm$~ 1.99 & 32.24 ~$\pm$~ 1.17 \\
    segformer\_baseline &- & 5& 3 & 50.08 ~$\pm$~ 4.39 & 15.37 ~$\pm$~ 1.58 & 28.99 ~$\pm$~ 1.55 \\
    segformer\_bw &bw\_average & 5& 3 & 57.68 ~$\pm$~ 1.03 & \textbf{22.45} ~$\pm$~ 0.37 & 34.58 ~$\pm$~ 0.90 \\
    segformer\_co &co & 5& 3 & 56.14 ~$\pm$~ 0.34 & 19.51 ~$\pm$~ 0.36 & 31.95 ~$\pm$~ 0.31 \\
    segformer\_sc &sc & 5& 3 & 55.95 ~$\pm$~ 1.01 & 18.56 ~$\pm$~ 0.62 & 31.71 ~$\pm$~ 1.65 \\
    internImage\_baseline &- & 5& 3 & 33.43 ~$\pm$~ 5.45 & 9.64 ~$\pm$~ 1.84 & 19.00 ~$\pm$~ 0.63 \\
    internImage\_bw &bw\_average & 5& 3 & 56.82 ~$\pm$~ 1.61 & \textbf{17.89} ~$\pm$~ 0.43 & 28.10 ~$\pm$~ 0.75 \\
    internImage\_co &co & 5& 3 & 44.51 ~$\pm$~ 9.51 & 15.09 ~$\pm$~ 1.49 & 27.03 ~$\pm$~ 0.89 \\
    internImage\_sc &sc & 5& 3 & 46.29 ~$\pm$~ 6.03 & 14.30 ~$\pm$~ 1.86 & 26.64 ~$\pm$~ 1.33 \\
    \end{tabular}
    """,

    'acdc_snow': r"""
    \begin{table*}[h]
  \centering
  \begin{tabular}{@{}lcccccc@{}}
    \toprule
    Validation on acdc\_snow & preprocessing &depth & channel  &  aAcc\_mean & mIoU\_mean & mAcc\_mean \\
    \midrule
    deeplabv3plus\_baseline &- & 5& 3 & 64.03 ~$\pm$~ 10.17 & 24.18 ~$\pm$~ 3.60 & 36.05 ~$\pm$~ 4.01 \\
    deeplabv3plus\_bw &bw\_average & 5& 3 & 70.05 ~$\pm$~ 3.19 & 31.58 ~$\pm$~ 1.88 & 45.09 ~$\pm$~ 3.17 \\
    deeplabv3plus\_co &co & 5& 3 & 76.43 ~$\pm$~ 5.42 & \textbf{32.97} ~$\pm$~ 2.97 & 44.40 ~$\pm$~ 3.83 \\
    deeplabv3plus\_sc &sc & 5& 3 & 69.89 ~$\pm$~ 1.47 & 32.60  ~$\pm$~ 3.50 & 44.66 ~$\pm$~ 2.88 \\
    segformer\_baseline &- & 5& 3 & 86.10 ~$\pm$~ 0.36 & \textbf{45.35} ~$\pm$~ 0.38 & 55.85 ~$\pm$~ 1.06 \\
    segformer\_bw &bw\_average & 5& 3 & 75.98 ~$\pm$~ 3.70 & 37.23 ~$\pm$~ 1.26 & 46.03 ~$\pm$~ 1.13 \\
    segformer\_co &co & 5& 3 & 83.08 ~$\pm$~ 1.24 & 41.35 ~$\pm$~ 1.18 & 49.96 ~$\pm$~ 1.47 \\
    segformer\_sc &sc & 5& 3 & 84.30  ~$\pm$~ 0.83 & 42.10 ~$\pm$~ 1.53 & 51.19 ~$\pm$~ 1.68 \\
    internImage\_baseline &- & 5& 3 & 68.91 ~$\pm$~ 3.87 & \textbf{26.90} ~$\pm$~ 1.92 & 40.17 ~$\pm$~ 2.42 \\
    internImage\_bw &bw\_average & 5& 3 & 65.72 ~$\pm$~ 7.38 & 25.84 ~$\pm$~ 2.24 & 36.61 ~$\pm$~ 4.14 \\
    internImage\_co &co & 5& 3 & 69.17 ~$\pm$~ 1.68 & 26.74 ~$\pm$~ 1.63 & 37.03 ~$\pm$~ 1.48 \\
    internImage\_sc &sc & 5& 3 & 66.81 ~$\pm$~ 7.99 & 25.30 ~$\pm$~ 2.26 & 35.74 ~$\pm$~ 2.98 \\
    \end{tabular}
    """,

    'acdc_fog': r"""
    \begin{table*}[h]
  \centering
  \begin{tabular}{@{}lcccccc@{}}
    \toprule
    Validation on acdc\_fog & preprocessing &depth & channel  &  aAcc\_mean & mIoU\_mean & mAcc\_mean \\
    \midrule
    deeplabv3plus\_baseline &- & 5& 3 & 76.52 ~$\pm$~ 6.46 & 39.20 ~$\pm$~ 4.35 & 54.93 ~$\pm$~ 6.11 \\
    deeplabv3plus\_bw &bw\_average & 5& 3 & 84.55 ~$\pm$~ 3.74 & 48.90  ~$\pm$~ 3.67 & 62.73 ~$\pm$~ 3.23 \\
    deeplabv3plus\_co &co & 5& 3 & 87.64 ~$\pm$~ 2.93 & 49.46 ~$\pm$~ 3.07 & 60.40  ~$\pm$~ 4.01 \\
    deeplabv3plus\_sc &sc & 5& 3 & 83.45 ~$\pm$~ 1.46 & \textbf{50.66} ~$\pm$~ 3.38 & 62.22 ~$\pm$~ 2.75 \\
    segformer\_baseline &- & 5& 3 & 92.47 ~$\pm$~ 0.19 & \textbf{61.71} ~$\pm$~ 0.79 & 72.98 ~$\pm$~ 0.98 \\
    segformer\_bw &bw\_average & 5& 3 & 87.59 ~$\pm$~ 1.92 & 52.07 ~$\pm$~ 0.47 & 61.88 ~$\pm$~ 0.86 \\
    segformer\_co &co & 5& 3 & 89.75 ~$\pm$~ 0.32 & 54.13 ~$\pm$~ 2.75 & 64.84 ~$\pm$~ 2.18 \\
    segformer\_sc &sc & 5& 3 & 91.03 ~$\pm$~ 0.33 & 58.12 ~$\pm$~ 2.73 & 69.61 ~$\pm$~ 2.69 \\
    internImage\_baseline &- & 5& 3 & 82.94 ~$\pm$~ 2.47 & 40.96 ~$\pm$~ 1.73 & 51.63 ~$\pm$~ 2.55 \\
    internImage\_bw &bw\_average & 5& 3 & 81.43 ~$\pm$~ 5.19 & 42.62 ~$\pm$~ 1.16 & 52.51 ~$\pm$~ 4.33 \\
    internImage\_co &co & 5& 3 & 84.51 ~$\pm$~ 1.64 & 44.93 ~$\pm$~ 0.41 & 54.20 ~$\pm$~ 1.18 \\
    internImage\_sc &sc & 5& 3 & 82.61 ~$\pm$~ 6.45 & \textbf{45.26} ~$\pm$~ 2.88 & 55.95 ~$\pm$~ 4.02 \\
    \end{tabular}
    """,

    'acdc_rain': r"""
    \begin{table*}[h]
  \centering
  \begin{tabular}{@{}lcccccc@{}}
    \toprule
    Validation on acdc\_rain & preprocessing &depth & channel  &  aAcc\_mean & mIoU\_mean & mAcc\_mean\\
    deeplabv3plus\_baseline &- & 5& 3 & 77.43 ~$\pm$~ 3.46 & 33.31 ~$\pm$~ 2.15 & 48.13 ~$\pm$~ 4.57 \\
    deeplabv3plus\_bw &bw\_average & 5& 3 & 76.72 ~$\pm$~ 3.07 & 34.20  ~$\pm$~ 2.50 & 47.68 ~$\pm$~ 4.67 \\
    deeplabv3plus\_co &co & 5& 3 & 83.44 ~$\pm$~ 3.64 & \textbf{38.87} ~$\pm$~ 2.24 & 51.26 ~$\pm$~ 3.57 \\
    deeplabv3plus\_sc &sc & 5& 3 & 77.82 ~$\pm$~ 1.63 & 38.60 ~$\pm$~ 3.04 & 51.81 ~$\pm$~ 2.65 \\
    segformer\_baseline &- & 5& 3 & 87.84 ~$\pm$~ 0.56 & \textbf{47.67} ~$\pm$~ 2.45 & 66.38 ~$\pm$~ 2.95 \\
    segformer\_bw &bw\_average & 5& 3 & 80.99 ~$\pm$~ 1.77 & 35.91 ~$\pm$~ 0.63 & 47.94 ~$\pm$~ 0.84 \\
    segformer\_co &co & 5& 3 & 84.87 ~$\pm$~ 1.52 & 41.21 ~$\pm$~ 1.46 & 56.26 ~$\pm$~ 2.79 \\
    segformer\_sc &sc & 5& 3 & 86.47 ~$\pm$~ 0.19 & 41.91 ~$\pm$~ 0.43 & 56.88 ~$\pm$~ 1.95 \\
    internImage\_baseline &- & 5& 3 & 80.34 ~$\pm$~ 2.63 & 31.62 ~$\pm$~ 2.19 & 45.78 ~$\pm$~ 3.08 \\
    internImage\_bw &bw\_average & 5& 3 & 67.32 ~$\pm$~ 5.93 & 28.14 ~$\pm$~ 0.50 & 38.23 ~$\pm$~ 2.57 \\
    internImage\_co &co & 5& 3 & 75.13 ~$\pm$~ 5.91 & \textbf{33.96} ~$\pm$~ 2.67 & 44.00    ~$\pm$~ 2.42 \\
    internImage\_sc &sc & 5& 3 & 73.61 ~$\pm$~ 6.08 & 33.35 ~$\pm$~ 1.69 & 43.90  ~$\pm$~ 2.94 \\
    \end{tabular}
    """

}

# ===================
# Hilfsfunktion
# ===================

def extract_val_std(s):
    s = s.replace('~', '').replace('$', '').replace('\\pm', '').strip()
    parts = re.findall(r'[\d.]+', s)
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    elif len(parts) == 1:
        return float(parts[0]), 0.0
    else:
        raise ValueError(f"Unrecognized format: {s}")


# --- LaTeX-Zeilen extrahieren ---
all_rows = []
for dataset, latex in latex_tables.items():
    for line in latex.splitlines():
        if '&' in line and '~' in line:
            line_clean = re.sub(r'\\[a-zA-Z]+\*?', '', line)
            line_clean = re.sub(r'[^a-zA-Z0-9.\-_&\s]', '', line_clean)
            parts = [p.strip() for p in line_clean.split('&')]
            if len(parts) == 7:
                run, prep, depth, channel, aAcc_str, mIoU_str, mAcc_str = parts
                aAcc_mean, aAcc_std = extract_val_std(aAcc_str)
                mIoU_mean, mIoU_std = extract_val_std(mIoU_str)
                mAcc_mean, mAcc_std = extract_val_std(mAcc_str)
                model, method = run.split('_', 1)
                all_rows.append({
                    'dataset': dataset,
                    'run': run,
                    'model': model,
                    'method': method,
                    'preprocessing': prep,
                    'depth': int(depth),
                    'channel': int(channel),
                    'aAcc_mean': aAcc_mean,
                    'aAcc_std': aAcc_std,
                    'mIoU_mean': mIoU_mean,
                    'mIoU_std': mIoU_std,
                    'mAcc_mean': mAcc_mean,
                    'mAcc_std': mAcc_std,
                })

df = pd.DataFrame(all_rows)

# --- Farbschema und Legenden-Benennung ---
label_map = {
    'baseline': 'baseline',
    'bw': 'bw_average',
    'co': 'co',
    'sc': 'sc'
}
"""
palette = {
    'baseline': '#505A5B',
    'bw_average': '#7775A9',
    'co': '#33A5D7',
    'sc': '#FC5F67'
}"""
palette_two = sns.color_palette("colorblind", n_colors=len(label_map))
color_map_two = {line[0]: palette_two[i] for i, line in enumerate(label_map.items())}

method_order = ['baseline', 'bw', 'co', 'sc']

# ===================
# Balkendiagramm (bar plot) statt Linienplot
# ===================

sns.set(style="whitegrid")

datasets_order = ["cityscapes", "dark_zurich", "acdc_night", "acdc_snow", "acdc_fog", "acdc_rain"]
x = np.arange(len(datasets_order))
group_width = 0.8
bar_width = group_width / len(method_order)

for k, model in enumerate(df['model'].unique()):
    fig, ax = plt.subplots(figsize=(14, 6))
    subset = df[df['model'] == model]

    for i, method in enumerate(method_order):
        method_data = subset[subset['method'] == method]
        if not method_data.empty:
            means = []
            stds = []
            positions = x - group_width / 2 + i * bar_width + bar_width / 2

            for ds in datasets_order:
                row = method_data[method_data['dataset'] == ds]
                if not row.empty:
                    means.append(row['mIoU_mean'].values[0])
                    stds.append(row['mIoU_std'].values[0])
                else:
                    means.append(0.0)
                    stds.append(0.0)

            label = label_map[method]
            color = color_map_two[method]


            ax.bar(
                positions,
                means,
                yerr=stds,
                capsize=4,
                width=bar_width,
                color=color,
                edgecolor=color,
                linewidth=0,
                label=label
            )

    # Achsenbeschriftung
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Cityscapes", "Dark Zurich", "ACDC night", "ACDC snow", "ACDC fog", "ACDC rain"],
        fontproperties=times_new_roman
    )
    ax.set_xlabel("Dataset", fontproperties=times_new_roman)
    ax.set_ylabel("mIoU [%]", fontproperties=times_new_roman)
    ax.set_title(f"{network_list[k]} - Validation", fontproperties=times_new_roman)
    ax.set_ylim(0, 80)
    print(ax.get_xticklabels()[0].get_fontproperties().get_size())

    # Legende außen
    legend_names = ['Baseline', 'Black white average (3 channel)', 'Color opponency', 'Single color']
    legend = ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    for i, text in enumerate(legend.get_texts()):
        text.set_text(legend_names[i])
        text.set_fontproperties(times_new_roman)
    legend.get_title().set_fontproperties(times_new_roman)

    # Gitter & Layout
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
    fig.tight_layout(rect=[0, 0, 1, 1])

    # Speichern
    plt.savefig(f"visualization/architecture_{model}_d5_bar.pdf", dpi=800)
    plt.show()


#TODO: here depth-search visualization

save_name = "acdc_rain"
plt_name = "ACDC rain"
# === 1. LaTeX-Tabelle als String einfügen ===
latex_table = r"""
    \begin{table*}[H]
  \centering
  \begin{tabular}{@{}lcccccc@{}}
    \toprule
    Validation on acdc\_rain & preprocessing & dept   &  aAcc\_mean  & mIoU\_mean  & mAcc\_mean  \\
    \midrule
    baseline/sc\_d0 & -/sc &0 & 82.86 ~$\pm$~ 0.98 & 32.22 ~$\pm$~ 0.29 & 46.63 ~$\pm$~ 0.91 \\
    \midrule
    bw\_d0 & bw\_average&0 & 85.47 ~$\pm$~ 1.32 & \textbf{37.13} ~$\pm$~ 1.23 & 50.35 ~$\pm$~ 0.68 \\
    bw\_d1 & bw\_average&1 & 79.31 ~$\pm$~ 2.43 & 33.68 ~$\pm$~ 0.71 & 47.46 ~$\pm$~ 2.47 \\
    bw\_d2 & bw\_average&2 & 82.98 ~$\pm$~ 2.50 & 32.74 ~$\pm$~ 1.77 & 43.15 ~$\pm$~ 2.03 \\
    bw\_d3 & bw\_average&3 & 78.91 ~$\pm$~ 4.43 & 32.43 ~$\pm$~ 2.13 & 43.65 ~$\pm$~ 2.58 \\
    bw\_d4 & bw\_average&4 & 80.59 ~$\pm$~ 2.58 & 32.88 ~$\pm$~ 0.80 & 45.12 ~$\pm$~ 1.97 \\
    bw\_d5 & bw\_average&5 & 81.10  ~$\pm$~ 4.44 & 33.89 ~$\pm$~ 1.08 & 45.96 ~$\pm$~ 1.55 \\
    bw\_d6 & bw\_average&6 & 79.44 ~$\pm$~ 3.39 & 33.14 ~$\pm$~ 1.67 & 45.54 ~$\pm$~ 1.42 \\
    bw\_d7 & bw\_average&7 & 82.05 ~$\pm$~ 3.17 & 32.10 ~$\pm$~ 2.10 & 43.55 ~$\pm$~ 1.80 \\
    bw\_d8 & bw\_average&8 & 81.14 ~$\pm$~ 3.68 & 32.70  ~$\pm$~ 1.02 & 45.16 ~$\pm$~ 1.84 \\
    bw\_d9 & bw\_average&9 & 82.90  ~$\pm$~ 1.48 & 33.39 ~$\pm$~ 0.72 & 46.41 ~$\pm$~ 2.41 \\
    bw\_d10 & bw\_average&10 & 79.71 ~$\pm$~ 4.93 & 33.68 ~$\pm$~ 1.81 & 46.06 ~$\pm$~ 2.69 \\
    \midrule
    co\_d0 & co &0 & 76.14 ~$\pm$~ 2.89 & 28.18 ~$\pm$~ 2.47 & 42.07 ~$\pm$~ 7.19 \\
    co\_d1 & co &1 & 79.26 ~$\pm$~ 1.16 & 32.90 ~$\pm$~ 1.54 & 46.46 ~$\pm$~ 0.51 \\
    co\_d2 & co &2 & 81.64 ~$\pm$~ 1.62 & 34.62 ~$\pm$~ 1.09  & 48.45 ~$\pm$~ 0.93 \\
    co\_d3 & co &3 & 80.14 ~$\pm$~ 4.86 & 31.64 ~$\pm$~ 2.92 & 45.95 ~$\pm$~ 1.61 \\
    co\_d4 & co &4 & 84.31 ~$\pm$~ 1.26 & 33.94 ~$\pm$~ 1.03 & 48.64 ~$\pm$~ 2.10 \\
    co\_d5 & co &5 & 79.89 ~$\pm$~ 4.85 & 34.44 ~$\pm$~ 1.27 & 48.87 ~$\pm$~ 2.13 \\
    co\_d6 & co &6 & 83.91 ~$\pm$~ 0.99 & \textbf{36.32} ~$\pm$~ 0.83 & 49.84 ~$\pm$~ 2.24 \\
    co\_d7 & co &7 & 81.36 ~$\pm$~ 4.55 & 34.01 ~$\pm$~ 0.21 & 45.53 ~$\pm$~ 1.99 \\
    co\_d8 & co &8 & 84.77 ~$\pm$~ 1.33 & 36.09 ~$\pm$~ 0.55 & 48.47 ~$\pm$~ 1.99 \\
    co\_d9 & co &9 & 84.21 ~$\pm$~ 0.81 & 35.55 ~$\pm$~ 0.65 & 47.42 ~$\pm$~ 1.62 \\
    co\_d10 & co &10 & 78.06 ~$\pm$~ 4.17 & 33.28 ~$\pm$~ 0.53 & 47.01 ~$\pm$~ 0.92 \\
    \midrule
    sc\_d1 & sc &1 & 79.68 ~$\pm$~ 3.54 & 32.60  ~$\pm$~ 1.80 & 46.00 ~$\pm$~ 2.26 \\
    sc\_d2 & sc &2 & 82.00 ~$\pm$~ 1.62 & 33.80  ~$\pm$~ 0.58 & 46.25 ~$\pm$~ 1.02 \\
    sc\_d3 & sc &3 & 80.43 ~$\pm$~ 4.44 & \textbf{34.50} ~$\pm$~ 1.30 & 48.92 ~$\pm$~ 0.63 \\
    sc\_d4 & sc &4 & 76.73 ~$\pm$~ 9.03 & 32.07 ~$\pm$~ 3.11 & 46.41 ~$\pm$~ 1.16 \\
    sc\_d5 & sc &5 & 83.23 ~$\pm$~ 1.10 & 33.73 ~$\pm$~ 1.07 & 46.91 ~$\pm$~ 1.56 \\
    sc\_d6 & sc &6 & 80.71 ~$\pm$~ 2.92 & 33.08 ~$\pm$~ 0.61 & 46.39 ~$\pm$~ 1.33 \\
    sc\_d7 & sc &7 & 80.02 ~$\pm$~ 5.51 & 32.80 ~$\pm$~ 1.76 & 46.38 ~$\pm$~ 2.19 \\
    sc\_d8 & sc &8 & 82.40 ~$\pm$~ 4.75 & 33.95 ~$\pm$~ 2.12 & 47.19 ~$\pm$~ 1.76 \\
    sc\_d9 & sc &9 & 78.91 ~$\pm$~ 6.71 & 34.22 ~$\pm$~ 1.54 & 46.64 ~$\pm$~ 0.36 \\
    sc\_d10 & sc &10 & 81.52 ~$\pm$~ 1.96 & 32.85 ~$\pm$~ 1.05 & 45.74 ~$\pm$~ 1.17 \\
    \bottomrule
    \end{tabular}
    \caption{Validation on acdc\_rain}
  \label{tab:addlabel}%
\end{table*}%

"""  # Gekürzt für Demo – du kannst den gesamten Tabelleninhalt hier einfügen


def extract_val_std(s):
    s = s.replace('~', '').replace('$', '').replace('\\pm', '').strip()
    parts = re.findall(r'[\d.]+', s)
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    elif len(parts) == 1:
        return float(parts[0]), 0.0
    else:
        raise ValueError(f"Unrecognized format: {s}")

# === 3. Tabelle parsen ===
rows = []
for line in latex_table.splitlines():
    if '&' in line and '~' in line:
        line_clean = re.sub(r'\\[a-zA-Z]+', '', line)  # LaTeX-Befehle raus
        line_clean = re.sub(r'[^a-zA-Z0-9\.\-\_&\s]', '', line_clean)  # Sonderzeichen entfernen
        parts = [p.strip() for p in line_clean.split('&')]
        if len(parts) == 6:
            name, prep, depth, aAcc_str, mIoU_str, mAcc_str = parts
            aAcc_mean, aAcc_std = extract_val_std(aAcc_str)
            mIoU_mean, mIoU_std = extract_val_std(mIoU_str)
            mAcc_mean, mAcc_std = extract_val_std(mAcc_str)
            rows.append({
                'run': name,
                'preprocessing': prep,
                'depth': int(depth),
                'aAcc_mean': aAcc_mean,
                'aAcc_std': aAcc_std,
                'mIoU_mean': mIoU_mean,
                'mIoU_std': mIoU_std,
                'mAcc_mean': mAcc_mean,
                'mAcc_std': mAcc_std,
            })

df = pd.DataFrame(rows)

# === 4. Methode extrahieren ===
def extract_method(run):
    if 'baseline' in run:
        return 'baseline'
    elif 'bw' in run:
        return 'bw_average'  # Umbenennung hier schon
    elif 'co' in run:
        return 'co'
    elif 'sc' in run:
        return 'sc'
    return 'other'

df['method'] = df['run'].apply(extract_method)

# === 5. Baseline separat speichern (depth=0) ===
baseline_df = df[df['method'] == 'baseline']
if baseline_df.empty:
    raise ValueError("Keine Baseline-Daten gefunden!")

# === 6. sc depth=0 hinzufügen, wenn nicht vorhanden ===
baseline_d0 = baseline_df[baseline_df['depth'] == 0].iloc[0]

if not ((df['method'] == 'sc') & (df['depth'] == 0)).any():
    sc_d0_row = {
        'run': 'sc_d0',
        'preprocessing': 'sc',
        'depth': 0,
        'aAcc_mean': baseline_d0['aAcc_mean'],
        'aAcc_std': baseline_d0['aAcc_std'],
        'mIoU_mean': baseline_d0['mIoU_mean'],
        'mIoU_std': baseline_d0['mIoU_std'],
        'mAcc_mean': baseline_d0['mAcc_mean'],
        'mAcc_std': baseline_d0['mAcc_std'],
        'method': 'sc'
    }
    df = pd.concat([df, pd.DataFrame([sc_d0_row])], ignore_index=True)

df = df.sort_values(['method', 'depth']).reset_index(drop=True)

# === 7. Plot ===
sns.set(style="whitegrid")

method_order = ['baseline', 'bw_average', 'co', 'sc']
display_names = {
    'baseline': 'Baseline',
    'bw_average': 'Black white average (3 channel)',
    'co': 'Color opponency',
    'sc': 'Single color'
}

palette_list = sns.color_palette("colorblind", n_colors=len(method_order))
color_map = {method: palette_list[i] for i, method in enumerate(method_order)}

fig, ax = plt.subplots(figsize=(14, 6))

# === 4. Plotten ===
depths_all = np.arange(0, 11)
baseline_all = pd.DataFrame({
    'depth': depths_all,
    'mIoU_mean': [baseline_d0['mIoU_mean']] * len(depths_all),
    'mIoU_std': [baseline_d0['mIoU_std']] * len(depths_all)
})

for method in method_order:
    if method == 'baseline':
        ax.plot(baseline_all['depth'], baseline_all['mIoU_mean'],
                marker='o', label=display_names[method], color=color_map[method], linewidth=2)
        ax.errorbar(baseline_all['depth'], baseline_all['mIoU_mean'],
                    yerr=baseline_all['mIoU_std'], fmt='none', capsize=4, color=color_map[method])
    else:
        subset = df[df['method'] == method].sort_values('depth')
        if not subset.empty:
            ax.plot(subset['depth'], subset['mIoU_mean'],
                    marker='o', label=display_names[method], color=color_map[method], linewidth=2)
            ax.errorbar(subset['depth'], subset['mIoU_mean'],
                        yerr=subset['mIoU_std'], fmt='none', capsize=4, color=color_map[method])

# === 5. Achsen & Legende ===
ax.set_xticks(np.arange(0, 11))
for label in ax.get_xticklabels():
    label.set_fontproperties(times_new_roman)
ax.set_xlabel("Depth", fontproperties=times_new_roman)
ax.set_ylabel("mIoU [%]", fontproperties=times_new_roman)
ax.set_title(f"{plt_name} - Depth-search validation", fontproperties=times_new_roman)

print("Font size in FontProperties:", times_new_roman.get_size())
print(ax.get_xticklabels()[0].get_fontproperties().get_size())

legend = ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
for text in legend.get_texts():
    text.set_fontproperties(times_new_roman)
legend.get_title().set_fontproperties(times_new_roman)

# === 6. Styling & Export ===
ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
ax.set_ylim(min(df['mIoU_mean']) - 1, max(df['mIoU_mean']) + 1)

fig.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(f"visualization/{save_name}_depth_search.pdf", dpi=800)
plt.show()
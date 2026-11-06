"""Due istogrammi comparabili per la relazione di stage.
Dipendenze: numpy, pandas, matplotlib. I CSV originali non vengono modificati.
"""
from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def load_data(path, keep_duplicates=False):
    df = pd.read_csv(path)
    if "exact_match" not in df and "Exact_Match_Score" in df:
        df = df.rename(columns={"Exact_Match_Score": "exact_match"})
    for col in ("Confidence", "exact_match"):
        if col not in df:
            raise ValueError(f"{path}: colonna mancante: {col}")
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not np.isfinite(df[["Confidence", "exact_match"]].to_numpy()).all():
        raise ValueError(f"{path}: valori mancanti o non finiti")
    if not df.Confidence.between(0, 1).all() or not df.exact_match.isin([0, 1]).all():
        raise ValueError(f"{path}: confidenza fuori da [0,1] o EM non binario")
    if df.empty:
        raise ValueError(f"{path}: nessuna osservazione")
    raw = len(df)
    # Solo ID non nulli consentono di riconoscere con sicurezza una ripetizione.
    if "ID" in df:
        duplicate_ids = df.ID.notna() & df.ID.duplicated(keep=False)
        if duplicate_ids.any():
            warnings.warn(f"{Path(path).name}: ID ripetuti rilevati", stacklevel=2)
            if not keep_duplicates:
                mask = df.ID.notna() & df.duplicated(keep="first")
                df = df.loc[~mask].copy()
                if (df.ID.notna() & df.ID.duplicated()).any():
                    raise ValueError(f"{path}: ID ripetuti con dati discordanti; verificarli prima dell'analisi")
    removed = raw - len(df)
    if removed:
        warnings.warn(f"{Path(path).name}: rimosse {removed} righe duplicate identiche", stacklevel=2)
    return df, raw, removed


def plot_distribuzione_confidenza(file_modello1, file_modello2, nome_modello1,
                                  nome_modello2, titolo_contesto, file_output):
    dati = [load_data(p)[0] for p in (file_modello1, file_modello2)]
    intervalli = np.linspace(0, 1, 21)
    colori = ["#D55E00", "#0072B2"]
    etichette = ["Non esatta (EM=0)", "Esatta (EM=1)"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    fig.suptitle(titolo_contesto, fontsize=16)
    for ax, df, nome in zip(axes, dati, (nome_modello1, nome_modello2)):
        valori = [df.loc[df.exact_match == c, "Confidence"].to_numpy() for c in (0, 1)]
        # Percentuali sul totale del modello: le barre impilate sommano al 100%.
        ax.hist(valori, bins=intervalli,
                weights=[np.full(len(v), 100/len(df)) for v in valori],
                stacked=True, color=colori, edgecolor="white", linewidth=.5,
                label=[f"{e}: {len(v)}" for e, v in zip(etichette, valori)])
        em = f"{df.exact_match.mean():.2%}".replace(".", ",")
        ax.set_title(f"{nome}\n{len(df)} domande · Exact Match: {em}", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.set_xlabel("Confidenza media dei token")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, fontsize=10)
        print(f"{titolo_contesto} — {nome}: N={len(df)}, EM={em}")
    axes[0].set_ylabel("Percentuale delle risposte")
    fig.tight_layout(rect=(0, 0, 1, .94))
    output = Path(file_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent/"output_squad")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.input_dir/"grafici_relazione"
    for condizione in ("possibili", "impossibili"):
        plot_distribuzione_confidenza(
            args.input_dir/f"squad_predictions_7B_{condizione}.csv",
            args.input_dir/f"squad_predictions_1_5B_{condizione}.csv",
            "Qwen 2.5 7B Instruct", "Qwen 2.5 1.5B Instruct",
            f"Domande {condizione}", output/f"istogramma_confidenza_{condizione}.png")
    print(f"Istogrammi PNG e SVG salvati in: {output.resolve()}")


if __name__ == "__main__":
    main()

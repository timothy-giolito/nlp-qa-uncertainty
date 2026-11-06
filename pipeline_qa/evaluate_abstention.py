"""Analisi esplorativa dell'astensione sui CSV già valutati.

Non genera nuove risposte e non sceglie automaticamente una soglia ottimale.
Esempio: python evaluate_abstention.py --thresholds 0.80 0.85 0.90 0.95 0.99
"""
from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd


def load_predictions(path, answerable):
    # Una risposta letterale "NA" non deve essere trasformata in astensione.
    df = pd.read_csv(path, keep_default_na=False)
    if "exact_match" not in df and "Exact_Match_Score" in df:
        df = df.rename(columns={"Exact_Match_Score": "exact_match"})
    required = {"ID", "Predicted_Answer", "Confidence", "exact_match"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path}: colonne mancanti {sorted(required-set(df.columns))}")
    if df.empty or df.ID.astype(str).str.strip().eq("").any():
        raise ValueError(f"{path}: file vuoto o ID mancanti")
    for col in ("Confidence", "exact_match"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not np.isfinite(df[["Confidence", "exact_match"]].to_numpy()).all():
        raise ValueError(f"{path}: valori non finiti")
    if not df.Confidence.between(0, 1).all() or not df.exact_match.isin([0, 1]).all():
        raise ValueError(f"{path}: confidenza fuori intervallo o EM non binario")
    original_count = len(df)
    df = df.drop_duplicates().copy()
    if len(df) != original_count:
        warnings.warn(f"{Path(path).name}: escluse {original_count-len(df)} righe identiche", stacklevel=2)
    if df.ID.duplicated().any():
        raise ValueError(f"{path}: ID ripetuti con risultati discordanti")
    df["answerable"] = bool(answerable)
    answer = df.Predicted_Answer.astype(str).str.strip()
    # Nel generatore della repository il marcatore viene già convertito in stringa vuota.
    # Si riconosce qui anche il marcatore esatto, per CSV esportati prima della conversione.
    df["model_abstained"] = answer.eq("") | answer.str.upper().eq("NESSUNA RISPOSTA")
    return df


def apply_policy(df, threshold=None):
    """None = comportamento originale. Le astensioni native restano astensioni."""
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("La soglia deve appartenere a [0,1]")
    result = df.copy()
    result["filter_abstained"] = False
    if threshold is not None:
        result["filter_abstained"] = (~result.model_abstained) & (result.Confidence < threshold)
    result["answered"] = ~(result.model_abstained | result.filter_abstained)
    result["decision"] = np.select(
        [result.model_abstained, result.filter_abstained],
        ["astensione_modello", "astensione_filtro"], default="risposta")
    result["Final_Answer"] = result.Predicted_Answer.where(result.answered, "NESSUNA RISPOSTA")
    return result


def ratio(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def summarize(df, threshold, scope):
    n = len(df)
    answered = df.answered
    # Per domande impossibili qualsiasi risposta fornita è un errore, anche se
    # la precedente normalizzazione EM avesse cancellato tutte le sue parole.
    errors = answered & ((df.exact_match == 0) | (~df.answerable))
    possible = df.answerable
    impossible = ~possible
    correct_decision = (possible & answered & (df.exact_match == 1)) | (impossible & ~answered)
    return {
        "policy": "baseline" if threshold is None else "threshold",
        "threshold": threshold,
        "scope": scope,
        "n": n,
        "n_answered": int(answered.sum()),
        "coverage": ratio(int(answered.sum()), n),
        "n_errors_answered": int(errors.sum()),
        "risk": ratio(int(errors.sum()), int(answered.sum())),
        "n_model_abstained": int(df.model_abstained.sum()),
        "n_filter_abstained": int(df.filter_abstained.sum()),
        "abstention_rate_answerable": ratio(int((possible & ~answered).sum()), int(possible.sum())),
        "abstention_rate_unanswerable": ratio(int((impossible & ~answered).sum()), int(impossible.sum())),
        "task_success_rate": ratio(int(correct_decision.sum()), n),
    }


def evaluate(df, thresholds):
    records = []
    for threshold in [None, *thresholds]:
        evaluated = apply_policy(df, threshold)
        for scope, subset in [("mixed", evaluated),
                              ("answerable", evaluated.loc[evaluated.answerable]),
                              ("unanswerable", evaluated.loc[~evaluated.answerable])]:
            records.append(summarize(subset, threshold, scope))
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent/"output_squad")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[.80, .85, .90, .95, .99, 1.0])
    parser.add_argument("--export-threshold", type=float, help="Esporta anche le decisioni per questa soglia, senza dichiararla ottimale")
    args = parser.parse_args()
    thresholds = sorted(set(args.thresholds))
    for value in thresholds + ([] if args.export_threshold is None else [args.export_threshold]):
        if not np.isfinite(value) or not 0 <= value <= 1:
            parser.error("Le soglie devono appartenere a [0,1]")
    datasets = {}
    for model in ("7B", "1_5B"):
        frames = [load_predictions(args.input_dir/f"squad_predictions_{model}_{condition}.csv", answerable)
                  for condition, answerable in [("possibili", True), ("impossibili", False)]]
        combined = pd.concat(frames, ignore_index=True)
        if combined.ID.duplicated().any():
            raise ValueError(f"{model}: ID presenti sia tra le domande possibili sia tra le impossibili")
        datasets[model] = combined
    alignment = []
    for answerable, condition in [(True, "possibili"), (False, "impossibili")]:
        ids = [set(df.loc[df.answerable == answerable, "ID"]) for df in datasets.values()]
        alignment.append(dict(condition=condition, n_7B=len(ids[0]), n_1_5B=len(ids[1]),
                              common_ids=len(ids[0] & ids[1]), same_ids=ids[0] == ids[1]))
        if ids[0] != ids[1]:
            warnings.warn(f"{condition}: campioni diversi tra modelli ({len(ids[0]&ids[1])} ID comuni). "
                          "Il confronto tra modelli non è controllato sullo stesso campione.", stacklevel=2)
    output = args.output_dir or args.input_dir/"analisi_astensione"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(alignment).to_csv(output/"controllo_campioni.csv", index=False)
    for model, df in datasets.items():
        report = evaluate(df, thresholds)
        report.insert(0, "model", model)
        report.to_csv(output/f"astensione_{model}.csv", index=False)
        print(f"\nModello {model} — analisi esplorativa, non test indipendente")
        print(report.loc[report.scope == "mixed", ["policy", "threshold", "n_answered", "coverage", "risk"]].to_string(index=False))
        if args.export_threshold is not None:
            apply_policy(df, args.export_threshold).to_csv(output/f"decisioni_{model}.csv", index=False)
    print(f"\nRisultati: {output.resolve()}\nProporzioni tra 0 e 1; valori non definiti lasciati vuoti nei CSV.")


if __name__ == "__main__":
    main()

from pathlib import Path
from shutil import copy2
import pandas as pd
import string
import re
from collections import Counter

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = ''.join(ch for ch in text if ch not in set(string.punctuation))
    stop_words = r'\b(a|an|the|in|on|at|of|and|or|but|to)\b'
    text = re.sub(stop_words, ' ', text)
    return ' '.join(text.split())

def f1_score(prediction, reference):
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 1.0 if pred_tokens == ref_tokens else 0.0
    
    common_tokens = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common_tokens.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)

def evaluate_metrics(df, col_generated="Predicted_Answer", col_reference="True_Answer"):
    exact_matches = 0
    f1_total = 0.0
    total_questions = len(df)
    
    em_scores = []
    f1_scores = []

    for index, row in df.iterrows():
        gen_text = row[col_generated]
        ref_text = row[col_reference]
        
        # Exact Match
        if normalize_text(gen_text) == normalize_text(ref_text):
            em_scores.append(1)
            exact_matches += 1
        else:
            em_scores.append(0)
            
        # F1 Score
        f1 = f1_score(gen_text, ref_text)
        f1_scores.append(f1)
        f1_total += f1
        
    em_percentage = (exact_matches / total_questions) * 100 if total_questions > 0 else 0
    f1_average = (f1_total / total_questions) * 100 if total_questions > 0 else 0
    
    return em_percentage, f1_average, em_scores, f1_scores

def main():
    cartella_output = Path(__file__).resolve().parent / "output_squad"
    nomi_file = [
       "squad_predictions_7B_possibili.csv",
       "squad_predictions_7B_impossibili.csv",
       "squad_predictions_1_5B_possibili.csv",
       "squad_predictions_1_5B_impossibili.csv",
       "squad_predictions_llama_3_2_1B_possibili.csv",
       "squad_predictions_llama_3_2_1B_impossibili.csv",
       "squad_predictions_llama_3_1_8B_possibili.csv",
       "squad_predictions_llama_3_1_8B_impossibili.csv"
    ]
    soglia_confidenza = 0.85
    print(f"Avvio valutazione di {len(nomi_file)} file...", flush=True)

    # Controlla TUTTI i file prima di modificare qualsiasi CSV.
    dati_validati = []
    for nome_file in nomi_file:
        percorso_csv = cartella_output / nome_file
        if not percorso_csv.is_file():
            raise FileNotFoundError(f"File richiesto non trovato: {percorso_csv}")
        df_risposte = pd.read_csv(percorso_csv)
        richieste = {"ID", "Predicted_Answer", "True_Answer"}
        if not richieste.issubset(df_risposte.columns):
            raise ValueError(f"{nome_file}: colonne mancanti {richieste - set(df_risposte.columns)}")
        if df_risposte.ID.isna().any() or df_risposte.ID.astype(str).str.strip().eq("").any():
            raise ValueError(f"{nome_file}: sono presenti ID mancanti")
        numero_originale = len(df_risposte)
        df_risposte = df_risposte.drop_duplicates().copy()
        if df_risposte.ID.duplicated().any():
            raise ValueError(f"{nome_file}: stesso ID con dati differenti; controllare manualmente")
        if len(df_risposte) != 500:
            raise ValueError(f"{nome_file}: trovate {len(df_risposte)} domande uniche, attese 500. Nessun file modificato.")
        if "Confidence" in df_risposte:
            df_risposte["Confidence"] = pd.to_numeric(df_risposte["Confidence"], errors="raise")
            if not df_risposte.Confidence.between(0, 1).all():
                raise ValueError(f"{nome_file}: valori di confidenza mancanti o non validi")
        dati_validati.append((percorso_csv, df_risposte, numero_originale - len(df_risposte)))

    for percorso_csv, df_risposte, duplicati in dati_validati:
        backup = percorso_csv.with_suffix(".csv.bak")
        if not backup.exists():
            copy2(percorso_csv, backup)

        em, f1, em_list, f1_list = evaluate_metrics(df_risposte)
        df_risposte["exact_match"] = em_list
        df_risposte["F1_Score"] = f1_list
        df_risposte.to_csv(percorso_csv, index=False)

        print(f"\nFile: {percorso_csv.name}")
        print(f"Domande valutate: {len(df_risposte)}")
        print(f"Duplicati rimossi: {duplicati}")
        print(f"Exact Match: {em:.2f}%")
        print(f"F1-Score: {f1:.2f}%")

        if "Confidence" not in df_risposte.columns:
            print("Confidence assente: estrazione dei casi critici saltata.")
            continue

        suffisso = percorso_csv.stem.removeprefix("squad_predictions_")
        alta_confidenza = df_risposte["Confidence"] >= soglia_confidenza
        for metrica, colonna in [("em", "exact_match"), ("f1", "F1_Score")]:
            casi_critici = df_risposte[alta_confidenza & (df_risposte[colonna] == 0)]
            destinazione = cartella_output / f"casi_critici_{suffisso}_{metrica}_alta_confidenza.csv"
            casi_critici.to_csv(destinazione, index=False)
            print(f"Casi critici {metrica.upper()}: {len(casi_critici)} -> {destinazione.name}")

    print("\nValutazione completata.")


if __name__ == "__main__":
    main()

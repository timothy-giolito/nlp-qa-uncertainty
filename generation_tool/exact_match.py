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

if __name__ == "__main__":
    percorso_csv = "output_squad/squad_predictions.csv"
    
    try:
        df_risposte = pd.read_csv(percorso_csv)
        
        em_perc, f1_avg, em_list, f1_list = evaluate_metrics(
            df_risposte, 
            col_generated="Predicted_Answer", 
            col_reference="True_Answer"
        )
        
        df_risposte["Exact_Match_Score"] = em_list
        df_risposte["F1_Score"] = f1_list
        
        print(f"Totale domande analizzate: {len(df_risposte)}")
        print(f"Score Exact Match: {em_perc:.2f}%")
        print(f"Score F1 Medio: {f1_avg:.2f}%")
        
        # --- ANALISI INCERTEZZA ---
        soglia_confidenza = 0.85
        
        if "Confidence" in df_risposte.columns:
            # Estrazione casi critici basata su F1 Score
            casi_critici_f1 = df_risposte[
                (df_risposte["Confidence"] >= soglia_confidenza) & 
                (df_risposte["F1_Score"] == 0.0) 
            ]
            print(f"\nTrovate {len(casi_critici_f1)} risposte errate con confidenza >= {soglia_confidenza} e F1 Score pari a 0")
            
            percorso_critici_f1 = "output_squad/casi_critici_f1_alta_confidenza.csv"
            casi_critici_f1.to_csv(percorso_critici_f1, index=False)
            print(f"Casi critici (F1=0) salvati con successo in: '{percorso_critici_f1}'")
            
            # Manteniamo anche l'estrazione basata su Exact Match per confronto
            casi_critici_em = df_risposte[
                (df_risposte["Confidence"] >= soglia_confidenza) & 
                (df_risposte["Exact_Match_Score"] == 0)
            ]
            print(f"\nTrovate {len(casi_critici_em)} risposte errate con confidenza >= {soglia_confidenza} e Exact Match pari a 0")
            
            percorso_critici_em = "output_squad/casi_critici_em_alta_confidenza.csv"
            casi_critici_em.to_csv(percorso_critici_em, index=False)
            print(f"Casi critici (EM=0) salvati con successo in: '{percorso_critici_em}'")
            
        else:
            print("\nAttenzione: La colonna 'Confidence' non è presente nel file CSV. Assicurati di aver eseguito il nuovo script di generazione.")
            
    except FileNotFoundError:
        print(f"Errore: Il file {percorso_csv} non è stato trovato.")
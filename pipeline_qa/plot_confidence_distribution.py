import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribuzione_confidenza(file_modello1, file_modello2, nome_modello1, nome_modello2, titolo_contesto, file_output):
    # Caricamento dei dati
    df1 = pd.read_csv(file_modello1)
    df2 = pd.read_csv(file_modello2)

    # --- FIX: Uniformiamo il nome della colonna per la compatibilità con i file vecchi ---
    if 'Exact_Match_Score' in df1.columns:
        df1.rename(columns={'Exact_Match_Score': 'exact_match'}, inplace=True)
    if 'Exact_Match_Score' in df2.columns:
        df2.rename(columns={'Exact_Match_Score': 'exact_match'}, inplace=True)
    # -----------------------------------------------------------------------------------

    # Impostazione della figura
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # Palette per distinguere Exact Match
    palette = {True: '#2ecc71', False: '#e74c3c', 1: '#2ecc71', 0: '#e74c3c', 1.0: '#2ecc71', 0.0: '#e74c3c'}
    
    # Plot Modello 1 (7B)
    sns.histplot(data=df1, x='Confidence', hue='exact_match', palette=palette, bins=25, kde=True, ax=axes[0], stat="density", common_norm=False, alpha=0.6)
    axes[0].set_title(f'{nome_modello1}\n({titolo_contesto})')
    axes[0].set_xlabel('Livello di Confidenza')
    axes[0].set_ylabel('Densità')
    
    # Plot Modello 2 (1.5B)
    sns.histplot(data=df2, x='Confidence', hue='exact_match', palette=palette, bins=25, kde=True, ax=axes[1], stat="density", common_norm=False, alpha=0.6)
    axes[1].set_title(f'{nome_modello2}\n({titolo_contesto})')
    axes[1].set_xlabel('Livello di Confidenza')
    
    # Personalizzazione legenda
    labels = ['Sbagliata (EM=0)', 'Corretta (EM=1)']
    for ax in axes:
        legend = ax.get_legend()
        if legend:
            for t, l in zip(legend.texts, labels):
                t.set_text(l)
                
    plt.tight_layout()
    plt.savefig(file_output, dpi=300, bbox_inches='tight')
    plt.close() 
    print(f"Grafico salvato con successo: {file_output}")

if __name__ == "__main__":
    # 1. Confronto sulle Domande POSSIBILI
    plot_distribuzione_confidenza(
        file_modello1="output_squad/squad_predictions_7B_possibili.csv",
        file_modello2="output_squad/squad_predictions_1_5B_possibili.csv",
        nome_modello1="Qwen 2.5 7B Instruct",
        nome_modello2="Qwen2.5-1.5B-Instruct",
        titolo_contesto="Domande Possibili",
        file_output="output_squad/istogramma_confidenza_possibili.png"
    )

    # 2. Confronto sulle Domande IMPOSSIBILI
    plot_distribuzione_confidenza(
        file_modello1="output_squad/squad_predictions_7B_impossibili.csv",
        file_modello2="output_squad/squad_predictions_1_5B_impossibili.csv",
        nome_modello1="Qwen 2.5 7B Instruct",
        nome_modello2="Qwen2.5-1.5B-Instruct",
        titolo_contesto="Domande Impossibili",
        file_output="output_squad/istogramma_confidenza_impossibili.png"
    )
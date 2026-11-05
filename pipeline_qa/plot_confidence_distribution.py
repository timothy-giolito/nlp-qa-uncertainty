import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caricamento dei dati dai file generati nel progetto
# Assicurati che i nomi dei file CSV corrispondano a quelli reali nel tuo progetto
percorso_modello_1 = "pipeline_qa/output_squad/squad_predictions.csv"
percorso_modello_2 = "pipeline_qa/output_squad/squad_predictions_1_5B.csv"

df_modello_1 = pd.read_csv(percorso_modello_1)
df_modello_2 = pd.read_csv(percorso_modello_2)

def plot_distribuzione_confidenza(df1, df2, nome_modello1="Modello 1", nome_modello2="Modello 2"):
    # Impostazione della figura con due subplot affiancati
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # Palette cromatica per distinguere a colpo d'occhio il match
    # Supporta valori booleani (True/False) e numerici (1/0 o 1.0/0.0)[cite: 2]
    palette = {True: '#2ecc71', False: '#e74c3c', 1: '#2ecc71', 0: '#e74c3c', 1.0: '#2ecc71', 0.0: '#e74c3c'}
    
    # Plot per il Modello 1
    sns.histplot(
        data=df1, 
        x='confidence', 
        hue='exact_match', 
        palette=palette, 
        bins=25, 
        kde=True,          # Aggiunge la curva di densità kernel[cite: 2]
        ax=axes[0], 
        stat="density",    # Normalizza l'istogramma (densità invece di conteggio)[cite: 2]
        common_norm=False, # Calcola la densità separatamente per EM=1 ed EM=0[cite: 2]
        alpha=0.6          # Trasparenza per visualizzare le sovrapposizioni[cite: 2]
    )
    axes[0].set_title(f'Distribuzione Confidenza\n{nome_modello1}')
    axes[0].set_xlabel('Livello di Confidenza')
    axes[0].set_ylabel('Densità')
    
    # Plot per il Modello 2
    sns.histplot(
        data=df2, 
        x='confidence', 
        hue='exact_match', 
        palette=palette, 
        bins=25, 
        kde=True, 
        ax=axes[1], 
        stat="density", 
        common_norm=False,
        alpha=0.6
    )
    axes[1].set_title(f'Distribuzione Confidenza\n{nome_modello2}')
    axes[1].set_xlabel('Livello di Confidenza')
    
    # Personalizzazione delle legende
    labels = ['Sbagliata (EM=0)', 'Corretta (EM=1)']
    for ax in axes:
        legend = ax.get_legend()
        if legend:
            for t, l in zip(legend.texts, labels):
                t.set_text(l)
                
    # Ottimizzazione degli spazi
    plt.tight_layout()
    
    # Salvataggio dell'immagine (sostituisce plt.show() per generare direttamente il file)[cite: 2]
    percorso_salvataggio = "pipeline_qa/output_squad/istogramma_confidenza_confronto.png"
    plt.savefig(percorso_salvataggio, dpi=300, bbox_inches='tight')
    print(f"Grafico generato e salvato con successo in: {percorso_salvataggio}")

# Esecuzione del plot 
plot_distribuzione_confidenza(
    df1=df_modello_1, 
    df2=df_modello_2, 
    nome_modello1="Qwen 2.5 7B Instruct", 
    nome_modello2="Qwen2.5-1.5B-Instruct"
)
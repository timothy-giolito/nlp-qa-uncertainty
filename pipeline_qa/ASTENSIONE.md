# Politica di astensione: primo esperimento

Aggiungi `evaluate_abstention.py` alla cartella `pipeline_qa`, accanto agli altri script.
Richiede pandas e numpy (numpy è già dipendenza di pandas).
Dalla cartella pipeline_qa esegui:

```bash
python evaluate_abstention.py
```

Legge i quattro CSV esistenti in output_squad; non usa il modello, non modifica i CSV e non richiede GPU. Salva i risultati in output_squad/analisi_astensione. Per scegliere soglie ed esportare le risposte filtrate:

```bash
python evaluate_abstention.py --thresholds 0.85 0.90 0.95 0.99 --export-threshold 0.95
```

Sono disponibili --input-dir e --output-dir per specificare percorsi alternativi.

## Regola

1. Risposta vuota o marcatore esatto NESSUNA RISPOSTA: mantiene l'astensione nativa.
2. Risposta non vuota con Confidence < soglia: astensione introdotta dal filtro.
3. Risposta non vuota con Confidence >= soglia: risposta mostrata.

apply_policy(df, threshold) restituisce una copia con model_abstained, filter_abstained, answered, decision e Final_Answer. Il CSV originale resta invariato. Il filtro interviene dopo la generazione: non risparmia il costo necessario a generare la risposta.

## Risultati

- astensione_7B.csv e astensione_1_5B.csv: comportamento originale (baseline) e soglie, per campione misto, domande possibili e impossibili.
- controllo_campioni.csv: numerosità e ID comuni dei due modelli.
- decisioni_*.csv: opzionali, con --export-threshold; contengono le risposte originali e quelle filtrate.

Tutte le proporzioni sono tra 0 e 1. Un campo vuoto significa non definito, non zero.

| Colonna | Definizione |
| --- | --- |
| coverage | Risposte mostrate / tutte le domande del gruppo |
| risk | Risposte mostrate non esatte / risposte mostrate |
| n_model_abstained | Astensioni già prodotte dal modello |
| n_filter_abstained | Astensioni aggiunte dal filtro |
| abstention_rate_answerable | Astensioni / domande possibili |
| abstention_rate_unanswerable | Astensioni / domande impossibili |
| task_success_rate | (Risposte esatte a domande possibili + astensioni a domande impossibili) / totale |

Per una domanda impossibile, qualsiasi risposta fornita conta come errore. Questo evita che la normalizzazione preesistente, cancellando parole come “the”, consideri esatta una risposta non vuota. Per le domande possibili si usa l'EM già presente: EM=0 non equivale necessariamente a errore semantico. Se non si forniscono risposte, risk non è definito.

## Risultati esplorativi inclusi

Ogni modello ha 500 domande possibili e 500 impossibili, dopo la rimozione del duplicato identico nel 7B. L'analisi usa una miscela artificiale 50/50: il rischio complessivo dipende da questa composizione.

Per il 7B, senza filtro: 631 risposte mostrate su 1000 (copertura 63,1%) e rischio 50,08%. Con soglia 0,95: 453 risposte (45,3%) e rischio 41,28%. Con soglia 0,99: 285 risposte (28,5%) e rischio 27,72%.

Il rischio qui è calcolato SOLO sulle risposte mostrate: non è il complemento dell'EM complessivo del precedente report, che includeva le astensioni corrette sulle domande impossibili.

## Limiti e prossimo passo

Questa è un'analisi esplorativa, non la selezione definitiva della soglia. Non usare gli stessi esempi sia per scegliere la soglia sia per dichiararne le prestazioni finali. La soglia va scelta su validazione e verificata su test separato, preferibilmente separando per articolo/contesto e con le stesse partizioni per entrambi i modelli.

Verifica importante: i CSV delle domande possibili hanno soltanto 4 ID comuni tra i due modelli. Le domande impossibili hanno 500 ID comuni. I confronti diretti tra modelli sui campioni possibili/misti non sono quindi controllati. Il confronto di ogni modello con il proprio comportamento originale rimane descrittivamente utile. Per un confronto controllato, raccogliere predizioni sullo stesso insieme di domande.

Le confidenze sono già arrotondate a quattro decimali. Un valore salvato come 1 non è una garanzia di correttezza o una probabilità esattamente unitaria prima dell'arrotondamento. La media delle probabilità dei token non va interpretata come probabilità di correttezza.

Fonte: CSV della repository al commit 6aa4e10f3c3a4946806be24245509a5637c4b614. I CSV originali non sono inclusi nel pacchetto.

## Verifiche

```bash
python -m unittest discover -s tests -v
```

Cinque test: soglia inclusiva e astensione nativa, denominatori del rischio, nessuna risposta, baseline, conservazione della risposta letterale NA e duplicati. Eseguita anche l'analisi sui quattro CSV reali.

# Casi critici ad alta confidenza

Fonte: otto CSV forniti da Timothy Giolito, generati da evaluate_metrics.py nel progetto nlp-qa-uncertainty. Le statistiche seguenti sono state calcolate direttamente sugli allegati; i collegamenti finali conservano l’accesso ai dati completi (ID, contesto, domanda, riferimento, predizione, confidenza e metriche).

Criterio comune: Confidence ≥ 0,85. Selezione EM: exact_match = 0. Selezione F1: F1_Score = 0. I valori di confidenza sono medie delle probabilità dei token, non probabilità calibrate di correttezza.

| Modello | Domande | Criterio | N casi | Confidenza media | Minimo | Massimo |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 7B | possibili | EM=0 | 176 | 0.9581 | 0.8508 | 1.0000 |
| 7B | possibili | F1=0 | 40 | 0.9409 | 0.8508 | 1.0000 |
| 7B | impossibili | EM=0 | 117 | 0.9601 | 0.8527 | 1.0000 |
| 7B | impossibili | F1=0 | 117 | 0.9601 | 0.8527 | 1.0000 |
| 1.5B | possibili | EM=0 | 224 | 0.9220 | 0.8501 | 0.9938 |
| 1.5B | possibili | F1=0 | 31 | 0.9120 | 0.8501 | 0.9920 |
| 1.5B | impossibili | EM=0 | 175 | 0.9180 | 0.8511 | 0.9915 |
| 1.5B | impossibili | F1=0 | 175 | 0.9180 | 0.8511 | 0.9915 |

## Interpretazione e controlli

- Tutti gli otto file rispettano i criteri dichiarati e non presentano ID duplicati al loro interno.
- Per le domande impossibili i file EM=0 e F1=0 coincidono integralmente, separatamente per ciascun modello: non sommare i loro conteggi.
- Per le domande possibili i casi F1=0 sono un sottoinsieme dei casi EM=0. Restano 136 casi del 7B e 193 del modello 1.5B con EM=0 e F1>0, cioè corrispondenza lessicale parziale secondo la normalizzazione adottata.
- EM=0 non implica necessariamente errore semantico; F1=0 indica assenza di token condivisi dopo normalizzazione e non costituisce da solo una valutazione semantica.
- Questi file contengono solo casi selezionati: non permettono da soli di ricostruire accuratezza complessiva, copertura o rischio tra tutte le risposte ad alta confidenza. Per tali denominatori usare i CSV completi.
- Le medie di confidenza in tabella si riferiscono esclusivamente ai rispettivi sottoinsiemi di casi critici.
- Il precedente controllo sui CSV della repository aveva trovato solo 4 ID comuni tra i campioni di domande possibili dei due modelli. Questi conteggi non vanno interpretati come confronto controllato sugli stessi esempi senza verificare i CSV completi aggiornati.

## File completi

- [casi_critici_1_5B_impossibili_em_alta_confidenza.csv](https://chatgpt.com/library?file_id=libfile_021c174ca3c881918dd5f0c8838bb2e1&account_id=personal)
- [casi_critici_1_5B_impossibili_f1_alta_confidenza.csv](https://chatgpt.com/library?file_id=libfile_ff83086f66a88191bc6f02f4f8ec27ea&account_id=personal)
- [casi_critici_1_5B_possibili_em_alta_confidenza.csv](https://chatgpt.com/library?file_id=libfile_564f4c19af348191aef40a72b248e386&account_id=personal)
- [casi_critici_1_5B_possibili_f1_alta_confidenza.csv](https://chatgpt.com/library?file_id=libfile_8f55d63587048191bca53f2cee676fbd&account_id=personal)
- [casi_critici_7B_impossibili_em_alta_confidenza.csv](https://chatgpt.com/library?file_id=libfile_2913b3f8e9648191870c96e90a6799f6&account_id=personal)
- [casi_critici_7B_impossibili_f1_alta_confidenza.csv](https://chatgpt.com/library?file_id=libfile_27993e00d7f081919ede034901ad8098&account_id=personal)
- [casi_critici_7B_possibili_em_alta_confidenza.csv](https://chatgpt.com/library?file_id=libfile_f1f1968be97c81919712bc96edd47a6b&account_id=personal)
- [casi_critici_7B_possibili_f1_alta_confidenza.csv](https://chatgpt.com/library?file_id=libfile_4a64c10833e48191a746e62cadb6f1d6&account_id=personal)

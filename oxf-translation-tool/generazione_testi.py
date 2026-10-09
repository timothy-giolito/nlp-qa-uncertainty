import csv
import io
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. Caricamento ambiente e percorsi
load_dotenv()
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

FILE_INPUT = Path("data/Oxford 5000.txt")
FOLDER_OUTPUT = Path("output_traduzioni")
FILE_OUTPUT = FOLDER_OUTPUT / "oxford5000_tradotto_transformers.csv"

# Modello consigliato per la traduzione (7B è velocissimo e preciso)
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
BATCH_SIZE = 15  # Numero di parole elaborate per ogni generazione

FOLDER_OUTPUT.mkdir(parents=True, exist_ok=True)

# 2. Rilevamento Hardware e Configurazione Quantizzazione
print("Rilevamento hardware in corso...")
quantization_config = None

if torch.cuda.is_available():
    device = "cuda"
    print("-> Utilizzo GPU NVIDIA (CUDA)")
    # Quantizzazione a 4-bit per ridurre il consumo di RAM/VRAM a circa 5-6 GB
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
elif torch.backends.mps.is_available():
    device = "mps"
    print("-> Utilizzo GPU Apple (MPS)")
else:
    device = "cpu"
    print("-> Utilizzo CPU")

# 3. Caricamento Tokenizer e Modello
print(f"Caricamento del modello '{MODEL_NAME}' in corso...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_kwargs = {
    "token": token,
    "device_map": "auto",
}

if quantization_config and device == "cuda":
    model_kwargs["quantization_config"] = quantization_config
else:
    # Per MPS o CPU usa bfloat16/float16 per bilanciare velocità e RAM
    model_kwargs["torch_dtype"] = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
model.eval()
torch.set_grad_enabled(False)


# 4. Funzione per verificare i progressi precedenti
def carica_parole_gia_fatte(filepath):
    if not os.path.exists(filepath):
        return set()
    try:
        df = pd.read_csv(filepath, on_bad_lines="skip")
        if "English" in df.columns:
            return set(df["English"].astype(str).str.lower().str.strip())
    except Exception:
        pass
    return set()


# 5. Costruzione del Prompt per il Modello Chat
def genera_prompt_batch(batch_words):
    system_prompt = (
        "Sei un docente di lingua inglese esperto in lessicografia.\n"
        "Traduci le parole richieste dall'inglese all'italiano.\n"
        "Per ciascuna parola fornisci:\n"
        "1. La parte del discorso (Sostantivo, Verbo, Aggettivo, Avverbio, ecc.)\n"
        "2. I significati principali in italiano. Se ci sono più significati, SEPARALI TASSATIVAMENTE con la barra '/' (NON usare mai virgole per separare i significati).\n\n"
        "Rispondi ESCLUSIVAMENTE con il testo in formato CSV senza commenti, introduzioni o formattazione markdown.\n"
        "Intestazione del formato:\n"
        "English,PoS,Italian_Meanings"
    )

    user_prompt = f"Parole da tradurre:\n{', '.join(batch_words)}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# 6. Ciclo principale di esecuzione
def main():
    if not os.path.exists(FILE_INPUT):
        print(f"Errore: Il file di input '{FILE_INPUT}' non è stato trovato.")
        return

    with open(FILE_INPUT, "r", encoding="utf-8") as f:
        tutte_le_parole = [
            linea.strip()
            for linea in f
            if linea.strip() and not linea.startswith("#")
        ]

    parole_completate = carica_parole_gia_fatte(FILE_OUTPUT)
    parole_da_fare = [
        p for p in tutte_le_parole if p.lower().strip() not in parole_completate
    ]

    print(
        f"Totale parole: {len(tutte_le_parole)} | Già tradotte: {len(parole_completate)} | Rimanenti: {len(parole_da_fare)}"
    )

    if not parole_da_fare:
        print("Tutte le parole sono già state tradotte!")
        return

    # Inizializza il file CSV di output se non esiste
    if not os.path.exists(FILE_OUTPUT):
        with open(FILE_OUTPUT, "w", encoding="utf-8-sig", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["English", "PoS", "Italian_Meanings"])

    # Processamento a batch
    for i in tqdm(
        range(0, len(parole_da_fare), BATCH_SIZE), desc="Traduzione in corso"
    ):
        batch = parole_da_fare[i : i + BATCH_SIZE]
        prompt_text = genera_prompt_batch(batch)

        try:
            inputs = tokenizer(prompt_text, return_tensors="pt").to(
                model.device
            )

            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,  # do_sample=False rende l'output deterministico e più preciso
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Estrazione del solo testo generato (escludendo il prompt)
            prompt_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][prompt_len:]
            response_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()

            # Rimozione di eventuali blocchi ```csv ... ```
            if "```" in response_text:
                blocchi = response_text.split("```")
                for blocco in blocchi:
                    if (
                        "English" in blocco
                        or "PoS" in blocco
                        or "," in blocco
                    ):
                        response_text = blocco.replace("csv", "").strip()
                        break

            # Parsing del CSV e validazione delle 3 colonne
            f_in = io.StringIO(response_text)
            reader = csv.reader(f_in)
            righe = list(reader)

            righe_valide = []
            for riga in righe:
                if len(riga) == 3 and riga[0].lower().strip() != "english":
                    righe_valide.append(riga)

            # Scrittura incrementale sul file
            if righe_valide:
                with open(
                    FILE_OUTPUT, "a", encoding="utf-8-sig", newline=""
                ) as f_out:
                    writer = csv.writer(f_out)
                    writer.writerows(righe_valide)

        except Exception as e:
            print(f"\n[Errore durante il batch {i}]: {e}")
            continue

    print(f"\nGenerazione completata! File salvato in '{FILE_OUTPUT}'.")


if __name__ == "__main__":
    main()
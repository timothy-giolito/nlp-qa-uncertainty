import os
import csv
from pathlib import Path
from dotenv import load_dotenv
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

# 1. Caricamento ambiente e percorsi
load_dotenv()
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

FOLDER_OUTPUT = Path("output_squad")
FILE_OUTPUT = FOLDER_OUTPUT / "squad_predictions.csv"
FOLDER_OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" 

# 2. Rilevamento Hardware e Configurazione Quantizzazione
print("Rilevamento hardware in corso...")
quantization_config = None

if torch.cuda.is_available():
    device = "cuda"
    print("-> Utilizzo GPU NVIDIA (CUDA)")
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
    # Gestione ottimizzata per Mac (MPS) o CPU
    model_kwargs["torch_dtype"] = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
model.eval()
torch.set_grad_enabled(False)


# 4. Costruzione del Prompt per QA
def genera_prompt_qa(contesto, domanda):
    system_prompt = (
        "Sei un assistente AI esperto in comprensione del testo. "
        "Rispondi alla domanda che ti viene posta basandoti ESCLUSIVAMENTE sul contesto fornito. "
        "Sii molto conciso e diretto, fornendo solo la risposta esatta senza introduzioni o spiegazioni aggiuntive."
    )

    user_prompt = f"Contesto:\n{contesto}\n\nDomanda:\n{domanda}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# 5. Ciclo principale di esecuzione
def main():
    print("Caricamento del dataset SQuAD...")
    # 100 esempi dal set di addestramento
    dataset = load_dataset("rajpurkar/squad", split="train").select(range(100))

    # Inizializza il file CSV di output se non esiste
    if not os.path.exists(FILE_OUTPUT):
        with open(FILE_OUTPUT, "w", encoding="utf-8-sig", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["ID", "Context", "Question", "True_Answer", "Predicted_Answer"])

    print(f"\nInizio interrogazione per {len(dataset)} domande...")

    # Processamento (senza batch per gestire dinamicamente contesti di lunghezze molto diverse)
    for item in tqdm(dataset, desc="Generazione risposte"):
        id_domanda = item['id']
        contesto = item['context']
        domanda = item['question']
        
        # SQuAD salva le risposte in una lista. Prendiamo la prima disponibile.
        risposta_reale = item['answers']['text'][0] if len(item['answers']['text']) > 0 else ""

        prompt_text = genera_prompt_qa(contesto, domanda)

        try:
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=50, # Limite basso, le risposte di SQuAD sono solitamente brevi
                do_sample=False,   # do_sample=False rende l'output deterministico 
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Estrazione del solo testo generato (escludendo il prompt)
            prompt_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][prompt_len:]
            predicted_answer = tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()

            # Scrittura incrementale sul file CSV
            with open(FILE_OUTPUT, "a", encoding="utf-8-sig", newline="") as f_out:
                writer = csv.writer(f_out)
                writer.writerow([id_domanda, contesto, domanda, risposta_reale, predicted_answer])

        except Exception as e:
            print(f"\n[Errore sulla domanda {id_domanda}]: {e}")
            continue

    print(f"\nGenerazione completata! File salvato in '{FILE_OUTPUT}'.")


if __name__ == "__main__":
    main()
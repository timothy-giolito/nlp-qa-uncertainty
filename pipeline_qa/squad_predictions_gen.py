import os
import csv
from pathlib import Path
from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

# 1. Caricamento ambiente e percorsi
load_dotenv()
token = os.environ.get("HF_TOKEN")

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
    dataset = load_dataset("rajpurkar/squad", split="train").select(range(100))

    # Inizializza il file CSV di output aggiungendo la colonna Confidence
    if not os.path.exists(FILE_OUTPUT):
        with open(FILE_OUTPUT, "w", encoding="utf-8-sig", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["ID", "Context", "Question", "True_Answer", "Predicted_Answer", "Confidence"])

    print(f"\nInizio interrogazione per {len(dataset)} domande...")

    for item in tqdm(dataset, desc="Generazione risposte"):
        id_domanda = item['id']
        contesto = item['context']
        domanda = item['question']
        
        risposta_reale = item['answers']['text'][0] if len(item['answers']['text']) > 0 else ""

        prompt_text = genera_prompt_qa(contesto, domanda)

        try:
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

            # Aggiunti output_scores e return_dict_in_generate per catturare le probabilità
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,   
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

            prompt_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs.sequences[0][prompt_len:]
            predicted_answer = tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()

            # Calcolo della confidenza
            token_probs = []
            for i, logits in enumerate(outputs.scores):
                # Trasforma i logits in probabilità
                probs = F.softmax(logits[0], dim=-1)
                # Recupera la probabilità del token effettivamente scelto dal modello
                token_id = generated_tokens[i]
                token_prob = probs[token_id].item()
                token_probs.append(token_prob)
            
            # Media delle probabilità dei singoli token
            confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0

            # Scrittura incrementale sul file CSV (incluso il valore di confidenza arrotondato a 4 decimali)
            with open(FILE_OUTPUT, "a", encoding="utf-8-sig", newline="") as f_out:
                writer = csv.writer(f_out)
                writer.writerow([id_domanda, contesto, domanda, risposta_reale, predicted_answer, round(confidence, 4)])

        except Exception as e:
            print(f"\n[Errore sulla domanda {id_domanda}]: {e}")
            continue

    print(f"\nGenerazione completata! File salvato in '{FILE_OUTPUT}'.")


if __name__ == "__main__":
    main()
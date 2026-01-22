import json
import os
from datasets import Dataset
from transformers import MT5ForConditionalGeneration, AutoTokenizer, Trainer, TrainingArguments

# Forçar uso da CPU se o driver de vídeo estiver instável (opcional, mas seguro)
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

# 1. Carregar dados de forma eficiente
with open('train_t5.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
dataset = dataset.train_test_split(test_size=0.1)

# 2. Configurar Modelo e Tokenizer (usando AutoTokenizer para evitar warnings)
model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

def preprocess_function(examples):
    # Reduzimos para 64 para economizar MUITA RAM e VRAM
    max_len = 64 
    model_inputs = tokenizer(examples["input_text"], max_length=max_len, truncation=True, padding="max_length")
    labels = tokenizer(examples["target_text"], max_length=max_len, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# O parâmetro load_from_cache_file=False ajuda se o disco estiver cheio
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

training_args = TrainingArguments(
    output_dir="./resultados_asqp",
    eval_strategy="no",
    learning_rate=3e-4,
    per_device_train_batch_size=2,    # Pode tentar subir para 2 na 1050 Ti
    gradient_accumulation_steps=8,    # Mantém o treino estável
    num_train_epochs=5,
    fp16=False,                       # A 1050 Ti (Pascal) não ganha muita performance com FP16, melhor manter False para evitar erros
    save_total_limit=1,
    logging_steps=10,
    # Adicione isso para otimizar o uso da memória de vídeo:
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

print("Iniciando o treinamento... Se for 'Killed' novamente, feche o navegador e o VS Code.")
trainer.train()

model.save_pretrained("./modelo_asqp_final")
tokenizer.save_pretrained("./modelo_asqp_final")
import json

def linearize_asqp(annotations):
    quads = []
    for ann in annotations:
        # Usamos str(valor) para garantir que mesmo None ou números virem string
        # O .get('campo') retorna None se a chave não existir
        aspect = str(ann.get('aspect') or 'none').strip().lower()
        category = str(ann.get('category') or 'none').strip().lower()
        sentiment = str(ann.get('sentiment') or 'none').strip().lower()
        polarity = str(ann.get('polarity') or 'none').strip().lower()
        
        quad = f"({aspect}, {category}, {polarity}, {sentiment})"
        quads.append(quad)
    
    return " [SEP] ".join(quads)

# --- Processamento do Arquivo ---

def gerar_dataset_linearizado(caminho_entrada, caminho_saida):
    with open(caminho_entrada, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    linearized_data = []
    
    for key, value in data.items():
        text = value['text']
        target = linearize_asqp(value['annotations'])
        
        linearized_data.append({
            "id": key,
            "input_text": text,
            "target_text": target
        })
    
    # Salva para uso no treinamento do Hugging Face
    with open(caminho_saida, 'w', encoding='utf-8') as f:
        json.dump(linearized_data, f, ensure_ascii=False, indent=4)
    
    print(f"Dataset linearizado com sucesso! Salvo em: {caminho_saida}")
    return linearized_data

# Execução
gerar_dataset_linearizado('train_augmented.json', 'train_t5.json')
import os
import json
import random
from collections import Counter, defaultdict

import matplotlib.pyplot as plt

# ------------------ CONFIG ------------------
RANDOM_SEED = 42
INPUT_PATH = "train.json"
OUT_DIR = "backtranslation"
OUTPUT_PATH = os.path.join(OUT_DIR, "backtranslation.json")

# Modelo NLLB-200 (600M) - Ótimo custo-benefício para 4GB VRAM
MODEL_NAME = "facebook/nllb-200-distilled-600M"

CAT_W = {
    "others": 1.8,
    "price": 1.6,
    "location": 1,
    "general": 1,
    "service": 0.6,
    "structure": 0.6,
}
POL_W = {
    "NEU": 1.6,
    "NEG": 1,
    "POS": 0.35,
}

MAX_MULT_OVER_BASE = 3.0
PIVOT_LANGS = ["eng_Latn", "spa_Latn"]
SRC_LANG = "por_Latn"
MAX_NEW_TOKENS = 256

# Aumentado para aproveitar o paralelismo da GTX 1050 Ti
# Se der erro de "Out of Memory", reduza para 4 ou 2.
BATCH_SIZE = 8 

# ------------------ UTILS ------------------
def percentile(values, p):
    if not values: return 0
    v = sorted(values)
    k = (len(v) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(v) - 1)
    if f == c: return v[f]
    return v[f] + (v[c] - v[f]) * (k - f)

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ------------------ NLLB TRANSLATION ------------------
class NLLBBackTranslator:
    def __init__(self, model_name=MODEL_NAME):
        print(f"🚀 Carregando modelo no WSL...")
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Uso de float16 para economizar VRAM e ganhar velocidade na GPU
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        self.model.eval()
        self.cache = {}
        print(f"✅ Modelo carregado em: {self.device.upper()}")

    @torch.inference_mode()
    def translate_batch(self, texts, src_lang, tgt_lang):
        out = [None] * len(texts)
        todo_idx, todo_texts = [], []
        
        for i, t in enumerate(texts):
            key = (t, src_lang, tgt_lang)
            if key in self.cache: out[i] = self.cache[key]
            else:
                todo_idx.append(i)
                todo_texts.append(t)

        if not todo_texts: return out

        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(
            todo_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        # Otimizado para Backtranslation: num_beams=1 + do_sample gera mais variedade
        generated = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1, 
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        for i, t_in, t_out in zip(todo_idx, todo_texts, decoded):
            self.cache[(t_in, src_lang, tgt_lang)] = t_out
            out[i] = t_out

        return out

    def backtranslate_batch(self, texts, pivot_lang):
        # Step 1: PT -> Pivot
        step1 = []
        for batch in chunked(texts, BATCH_SIZE):
            step1.extend(self.translate_batch(batch, SRC_LANG, pivot_lang))

        # Step 2: Pivot -> PT
        step2 = []
        for batch in chunked(step1, BATCH_SIZE):
            step2.extend(self.translate_batch(batch, pivot_lang, SRC_LANG))

        return step2

# ------------------ MAIN ------------------
def main():
    random.seed(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        print(f"❌ Erro: Arquivo {INPUT_PATH} não encontrado.")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for ex in data.values():
        text = ex.get("text", "")
        for ann in ex.get("annotations", []) or []:
            cat, pol = ann.get("category"), ann.get("polarity")
            if cat and pol:
                samples.append({"text": text, "annotation": ann, "category": cat, "polarity": pol})


    by_pair = defaultdict(list)
    for s in samples:
        by_pair[(s["category"], s["polarity"])].append(s)

    all_counts = [len(lst) for lst in by_pair.values()]
    BASE_TARGET = max(int(percentile(all_counts, 75)), 1)
    
    tasks = []
    for (cat, pol), lst in by_pair.items():
        w = CAT_W.get(cat, 1.0) * POL_W.get(pol, 1.0)
        target = min(int(BASE_TARGET * w), int(BASE_TARGET * MAX_MULT_OVER_BASE))
        
        if len(lst) < target:
            chosen = random.choices(lst, k=target - len(lst))
            for i, s in enumerate(chosen):
                pivot = PIVOT_LANGS[i % len(PIVOT_LANGS)]
                tasks.append(((cat, pol), s, pivot))

    print(f" Gerando {len(tasks)} novas amostras via GPU...")

    bt = NLLBBackTranslator()
    new_samples = list(samples)

    # Processamento otimizado por idioma pivô
    for pivot in PIVOT_LANGS:
        pivot_tasks = [t for t in tasks if t[2] == pivot]
        if not pivot_tasks: continue

        texts_in = [t[1]["text"] for t in pivot_tasks]
        texts_out = []
        
        # Progresso simples
        print(f"🌐 Traduzindo via {pivot}...")
        for batch in chunked(texts_in, BATCH_SIZE * 2): # Batch maior no loop externo
            texts_out.extend(bt.backtranslate_batch(batch, pivot))

        for (pair, s, _), new_text in zip(pivot_tasks, texts_out):
            final_text = new_text if (new_text and new_text.strip()) else s["text"]
            new_samples.append({
                "text": final_text,
                "annotation": s["annotation"],
                "category": s["category"],
                "polarity": s["polarity"],
            })

    random.shuffle(new_samples)
    
    # Salvar resultados
    out_dict = {f"os_{i:07d}": {"text": s["text"], "annotations": [s["annotation"]]} 
                for i, s in enumerate(new_samples, start=1)}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False)

    print(f"Processo concluído! ")
    print(f"Salvo em: {OUTPUT_PATH}")

    # --- Gráficos ---
    categorias = [s["category"] for s in new_samples]
    cont_cat = Counter(categorias)

    plt.figure(figsize=(10, 6))
    plt.bar(cont_cat.keys(), cont_cat.values(), color='skyblue')
    plt.yscale("log")
    plt.title("Distribuição de Categorias (Log Scale)")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "distribuicao_categorias_log.png"))
    plt.close()

if __name__ == "__main__":
    main()
import json
import random
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

RANDOM_SEED = 42
INPUT_PATH = "train.json"
OUT_DIR = "oversampling"
OUTPUT_PATH = os.path.join(OUT_DIR, "oversampling.json")

random.seed(RANDOM_SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# Pesos adaptativos (ajuste aqui)
CAT_W = {
    "others": 4,     # mais agressivo
    "price": 2,      # mais agressivo
    "location": 1.2,   # ameno
    "general": 1.2,    # ameno
    "service": 0.6,
    "structure": 0.6,
}

POL_W = {
    "NEU": 0.3,  
    "NEG": 0.3,   
    "POS": 0.1,  
}

# Para não explodir (cap por par)
MAX_MULT_OVER_BASE = 3.0  

def percentile(values, p):
    """p em [0,100]. Sem numpy."""
    if not values:
        return 0
    v = sorted(values)
    k = (len(v) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(v) - 1)
    if f == c:
        return v[f]
    return v[f] + (v[c] - v[f]) * (k - f)

# ------------------ Ler dataset ------------------
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# ------------------ Explodir em amostras unitárias ------------------
# cada sample = 1 annotation (pra oversampling por par)
samples = []
for ex_id, ex in data.items():
    text = ex.get("text", "")
    for ann in ex.get("annotations", []) or []:
        cat = ann.get("category")
        pol = ann.get("polarity")
        if not cat or not pol:
            continue
        samples.append({"text": text, "annotation": ann, "category": cat, "polarity": pol})

print("Base samples:", len(samples))

# ------------------ Agrupar por par (cat, pol) ------------------
by_pair = defaultdict(list)
for s in samples:
    by_pair[(s["category"], s["polarity"])].append(s)

pair_counts = {pair: len(lst) for pair, lst in by_pair.items()}
all_counts = list(pair_counts.values())

BASE_TARGET = int(percentile(all_counts, 75))  # alvo base sem explodir
BASE_TARGET = max(BASE_TARGET, 1)

print("BASE_TARGET (p75 dos pares):", BASE_TARGET)

# ------------------ Gerar novos samples só para pares minoritários ------------------
new_samples = list(samples)  # começa com original

for (cat, pol), lst in by_pair.items():
    cur = len(lst)
    w = CAT_W.get(cat, 1.0) * POL_W.get(pol, 1.0)

    target = int(BASE_TARGET * w)
    # cap pra não explodir
    cap = int(BASE_TARGET * MAX_MULT_OVER_BASE)
    target = min(target, cap)

    if cur >= target:
        continue

    need = target - cur
    # amostrar com reposição dentro do próprio par
    added = random.choices(lst, k=need)
    new_samples.extend(added)

# ------------------ Embaralhar ------------------
random.shuffle(new_samples)

print("Oversampled samples:", len(new_samples))

# ------------------ Salvar oversampling.json (sem indent = mais rápido) ------------------
out = {}
for i, s in enumerate(new_samples, start=1):
    out[f"os_{i:07d}"] = {"text": s["text"], "annotations": [s["annotation"]]}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)  # sem indent = muito mais rápido

print(f"✅ Salvo: {OUTPUT_PATH}")

# ------------------ Contagens p/ gráficos (dataset final) ------------------
categorias = []
polaridades = []
for ex in out.values():
    ann = ex["annotations"][0]
    categorias.append(ann["category"])
    polaridades.append(ann["polarity"])

cont_cat = Counter(categorias)
cont_pol = Counter(polaridades)

# -------- Gráfico de Categorias --------
plt.figure()
plt.bar(cont_cat.keys(), cont_cat.values())
plt.title("Distribuição de Categorias - Oversampling")
plt.xlabel("Categoria")
plt.ylabel("Frequência")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "distribuicao_categorias.png"), dpi=300)
plt.close()

# -------- Gráfico de Categorias (escala log) --------
plt.figure()
plt.bar(cont_cat.keys(), cont_cat.values())
plt.yscale("log")
plt.title("Distribuição de Categorias - Oversampling")
plt.xlabel("Categoria")
plt.ylabel("Frequência (log)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "distribuicao_categorias_log.png"), dpi=300)
plt.close()

# -------- Gráfico de Polaridade --------
plt.figure()
plt.bar(cont_pol.keys(), cont_pol.values())
plt.title("Distribuição de Polaridade - Oversampling")
plt.xlabel("Polaridade")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "distribuicao_polaridade.png"), dpi=300)
plt.close()

print("Gráficos salvos em:", OUT_DIR)

# (opcional) imprimir distribuição por par para conferir
pair_after = Counter((ex["annotations"][0]["category"], ex["annotations"][0]["polarity"]) for ex in out.values())
print("Top 10 pares (após):", pair_after.most_common(10))
print("Bottom 10 pares (após):", pair_after.most_common()[-10:])
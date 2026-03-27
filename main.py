# -*- coding: utf-8 -*-
"""
Targeted augmentation for ABSA-style JSON with multiple annotations per review.

Goals (as requested):
1) Oversampling for polarity "NEU" (minority) targeting balance by (category, polarity)
2) Data augmentation for polarity "NEG" using:
   - synonym replacement (label-preserving)
   - antonym-style replacement WITHOUT flipping label (uses "não/nada <antônimo>" templates)
3) Do NOT fully balance; only mitigate imbalance via partial targets (ratios)

Input:  /train.json
Output: /train_aug_targeted.json

Assumptions (based on your dataset):
- data is a dict: {example_id: {"text": str, "annotations": [ ... ]}}
- each annotation has:
  - "category": one of {"service","price","location","general","structure",...}
  - "polarity": "POS"|"NEG"|"NEU"
  - "sentiment": {"term": str, "location": [start,end]}
  - "aspect": {"term": str, "location":[start,end]} (optional)
"""

import json
import random
from copy import deepcopy
from collections import Counter, defaultdict

random.seed(7)

IN_PATH = "train.json"
OUT_PATH = "train_aug_targeted.json"

# -----------------------------
# Hyperparams (tune these)
# -----------------------------
NEU_TARGET_RATIO = 0.15   # bring NEU up to 15% of max(POS,NEG) per category
NEG_TARGET_RATIO = 0.35   # bring NEG up to 35% of POS per category (per category)
MAX_NEW_PER_BASE_EX = 6   # avoid overusing the same base example (helps generalization)

# Probability for NEG augmentation choices
P_NEG_SYNONYM = 0.6       # 60% synonyms, 40% antonym-template

# If a sentiment.term has multiple words and no direct mapping, we skip.
ALLOW_MULTIWORD_TERM = False

# -----------------------------
# Lightweight lexicons (not "full manual synonyms"):
# - keep this minimal: focus on common sentiment adjectives.
# - you can expand later or swap with embedding/MLM candidate generation.
# -----------------------------
NEG_SYNONYMS = {
    "ruim": ["péssimo", "horrível", "muito ruim"],
    "péssimo": ["horrível", "terrível"],
    "horrível": ["péssimo", "terrível"],
    "terrível": ["horrível", "péssimo"],
    "caro": ["muito caro", "caríssimo"],
    "lento": ["devagar", "muito lento"],
    "sujo": ["imundo", "mal limpo"],
    "grosseiro": ["rude", "mal-educado"],
    "antipático": ["rude", "sem simpatia"],
    "barulhento": ["muito barulhento", "ruidoso"],
    "pequeno": ["apertado", "minúsculo"],
}

# For "antonym augmentation" while keeping NEG label:
# Replace term with "não/nada <positive_antonym>" which keeps negative meaning.
# Example: "ruim" -> "nada bom", "sujo" -> "não limpo"
POS_ANTONYMS = {
    "ruim": ["bom", "ótimo"],
    "péssimo": ["bom", "ótimo"],
    "horrível": ["bom", "ótimo"],
    "caro": ["barato", "em conta"],
    "lento": ["rápido", "ágil"],
    "sujo": ["limpo", "impecável"],
    "grosseiro": ["educado", "gentil"],
    "antipático": ["simpático", "cordial"],
    "barulhento": ["silencioso", "tranquilo"],
    "pequeno": ["grande", "espaçoso"],
}

NEG_TEMPLATES = [
    "não {w}",
    "nada {w}",
    "nem um pouco {w}",
]

# -----------------------------
# Helpers
# -----------------------------
def norm(s: str) -> str:
    return (s or "").strip().lower()

def is_single_token(term: str) -> bool:
    term = (term or "").strip()
    return (" " not in term) and ("\t" not in term) and ("\n" not in term)

def safe_get_loc(ann, field):
    loc = ann.get(field, {}).get("location")
    if not loc or len(loc) != 2:
        return None
    return int(loc[0]), int(loc[1])

def replace_span_and_update_offsets(text, anns, ann_idx, new_phrase: str):
    """
    Replace sentiment span using its location, and shift subsequent offsets.
    Only edits the sentiment span of the chosen annotation.
    """
    a = anns[ann_idx]
    loc = safe_get_loc(a, "sentiment")
    if loc is None:
        return None

    s0, s1 = loc
    if s0 < 0 or s1 > len(text) or s0 >= s1:
        return None

    old_phrase = text[s0:s1]
    if old_phrase == new_phrase:
        return None

    new_text = text[:s0] + new_phrase + text[s1:]
    delta = len(new_phrase) - (s1 - s0)

    # Update chosen sentiment
    a["sentiment"]["term"] = new_phrase
    a["sentiment"]["location"] = [s0, s1 + delta]

    # Shift subsequent annotations (both aspect & sentiment) that start after the replaced span end
    for j, aj in enumerate(anns):
        if j == ann_idx:
            continue

        for key in ("aspect", "sentiment"):
            if key in aj and "location" in aj[key] and aj[key]["location"] and len(aj[key]["location"]) == 2:
                x0, x1 = int(aj[key]["location"][0]), int(aj[key]["location"][1])
                if x0 >= s1:
                    aj[key]["location"] = [x0 + delta, x1 + delta]

    return new_text, anns

def make_single_annotation_example(ex, ann_i):
    """
    Create a new example that contains ONLY the chosen annotation.
    This avoids accidentally increasing other polarities from the same review.
    """
    new_ex = {
        "text": ex["text"],
        "annotations": [deepcopy(ex["annotations"][ann_i])]
    }
    return new_ex

def counts_by_category_polarity(data):
    counts = defaultdict(Counter)  # counts[category][polarity]
    for ex in data.values():
        for a in ex.get("annotations", []):
            cat = a.get("category", "UNKNOWN")
            pol = a.get("polarity", None)
            if pol:
                counts[cat][pol] += 1
    return counts

def index_by_category_polarity(data):
    """
    returns idx[category][polarity] = list[(ex_id, ann_i)]
    """
    idx = defaultdict(lambda: defaultdict(list))
    for ex_id, ex in data.items():
        anns = ex.get("annotations", [])
        for i, a in enumerate(anns):
            cat = a.get("category", "UNKNOWN")
            pol = a.get("polarity", None)
            if pol:
                idx[cat][pol].append((ex_id, i))
    return idx

def propose_neg_synonym(term: str):
    t = norm(term)
    if not t:
        return None
    if (not ALLOW_MULTIWORD_TERM) and (not is_single_token(t)):
        return None
    cands = NEG_SYNONYMS.get(t)
    if not cands:
        return None
    return random.choice(cands)

def propose_neg_antonym_template(term: str):
    """
    Keep NEG label by using a negation template with a positive antonym.
    """
    t = norm(term)
    if not t:
        return None
    if (not ALLOW_MULTIWORD_TERM) and (not is_single_token(t)):
        return None
    pos_cands = POS_ANTONYMS.get(t)
    if not pos_cands:
        return None
    pos_word = random.choice(pos_cands)
    tmpl = random.choice(NEG_TEMPLATES)
    return tmpl.format(w=pos_word)

# -----------------------------
# Main augmentation routine
# -----------------------------
def compute_targets(counts):
    """
    Partial targets per category.
    """
    targets = defaultdict(dict)
    for cat, c in counts.items():
        pos = c.get("POS", 0)
        neg = c.get("NEG", 0)
        neu = c.get("NEU", 0)

        # NEU target: ratio of max(POS,NEG) in that category
        base = max(pos, neg)
        targets[cat]["NEU"] = max(neu, int(round(NEU_TARGET_RATIO * base)))

        # NEG target: ratio of POS (if POS=0, use max(base,1) to avoid always zero)
        base_pos = max(pos, 1)
        targets[cat]["NEG"] = max(neg, int(round(NEG_TARGET_RATIO * base_pos)))

        # Keep POS unchanged (we won't augment POS here)
        targets[cat]["POS"] = pos

    return targets

def augment(data):
    data_out = deepcopy(data)

    counts = counts_by_category_polarity(data_out)
    idx = index_by_category_polarity(data_out)
    targets = compute_targets(counts)

    # Track how many times each base example was used
    used_per_base = Counter()

    new_examples = {}
    new_id_counter = 0

    def new_id(base_id, tag):
        nonlocal new_id_counter
        new_id_counter += 1
        return f"{base_id}_{tag}_{new_id_counter:06d}"

    # ---- 1) Oversample NEU (by category) ----
    for cat, t in targets.items():
        cur = counts[cat].get("NEU", 0)
        need = max(0, t["NEU"] - cur)
        if need == 0:
            continue

        pool = idx[cat].get("NEU", [])
        if not pool:
            continue

        # sample with replacement, but cap per base example
        attempts = 0
        created = 0
        while created < need and attempts < need * 10:
            attempts += 1
            base_id, ann_i = random.choice(pool)
            if used_per_base[base_id] >= MAX_NEW_PER_BASE_EX:
                continue

            base_ex = data_out[base_id]
            if ann_i >= len(base_ex.get("annotations", [])):
                continue

            # make a single-annotation example to avoid increasing other polarities
            ex_new = make_single_annotation_example(base_ex, ann_i)

            nid = new_id(base_id, "neu_os")
            new_examples[nid] = ex_new

            used_per_base[base_id] += 1
            created += 1

        counts[cat]["NEU"] += created

    # ---- 2) Augment NEG (by category) with synonym / antonym-template ----
    for cat, t in targets.items():
        cur = counts[cat].get("NEG", 0)
        need = max(0, t["NEG"] - cur)
        if need == 0:
            continue

        pool = idx[cat].get("NEG", [])
        if not pool:
            continue

        attempts = 0
        created = 0
        while created < need and attempts < need * 20:
            attempts += 1
            base_id, ann_i = random.choice(pool)
            if used_per_base[base_id] >= MAX_NEW_PER_BASE_EX:
                continue

            base_ex = data_out[base_id]
            anns = base_ex.get("annotations", [])
            if ann_i >= len(anns):
                continue

            ann = anns[ann_i]
            term = ann.get("sentiment", {}).get("term", "")
            if not term:
                continue

            # choose strategy
            do_syn = (random.random() < P_NEG_SYNONYM)
            if do_syn:
                repl = propose_neg_synonym(term)
                tag = "neg_syn"
            else:
                repl = propose_neg_antonym_template(term)
                tag = "neg_ant"

            if not repl:
                continue

            # Make single-annotation example first (so offsets shifting is trivial)
            ex_new = make_single_annotation_example(base_ex, ann_i)
            text = ex_new["text"]
            anns_new = ex_new["annotations"]

            # Replace span in this single annotation
            res = replace_span_and_update_offsets(text, anns_new, 0, repl)
            if res is None:
                continue
            text2, anns2 = res
            ex_new["text"] = text2
            ex_new["annotations"] = anns2
            # polarity stays NEG

            nid = new_id(base_id, tag)
            new_examples[nid] = ex_new

            used_per_base[base_id] += 1
            created += 1

        counts[cat]["NEG"] += created

    # Merge
    data_out.update(new_examples)
    return data_out

# -----------------------------
# Run
# -----------------------------
with open(IN_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

before_counts = counts_by_category_polarity(data)

aug = augment(data)

after_counts = counts_by_category_polarity(aug)

# Print a compact report
def print_report(before, after):
    cats = sorted(set(before.keys()) | set(after.keys()))
    print("\n=== (category, polarity) annotation counts: BEFORE -> AFTER ===")
    for cat in cats:
        b = before.get(cat, Counter())
        a = after.get(cat, Counter())
        print(
            f"{cat:10s}  "
            f"POS {b.get('POS',0):4d}->{a.get('POS',0):4d}  "
            f"NEG {b.get('NEG',0):4d}->{a.get('NEG',0):4d}  "
            f"NEU {b.get('NEU',0):4d}->{a.get('NEU',0):4d}"
        )

    tot_b = Counter()
    tot_a = Counter()
    for cat in cats:
        tot_b.update(before.get(cat, Counter()))
        tot_a.update(after.get(cat, Counter()))
    print("\n=== TOTAL polarities (annotations) ===")
    print(f"POS {tot_b.get('POS',0)} -> {tot_a.get('POS',0)}")
    print(f"NEG {tot_b.get('NEG',0)} -> {tot_a.get('NEG',0)}")
    print(f"NEU {tot_b.get('NEU',0)} -> {tot_a.get('NEU',0)}")

print_report(before_counts, after_counts)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(aug, f, ensure_ascii=False, indent=2)

print("\nSaved:", OUT_PATH)
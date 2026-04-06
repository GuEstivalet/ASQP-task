import json


def gerar_dataset_filtrado(input_path, output_path):

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)


    linearized_data = []
    excluidos = 0

    for key, value in data.items():
        annotations = value.get("annotations") or []
        if not annotations:
            excluidos += 1
            continue

        text = value.get("text", "")
        target = linearizer(annotations)

        linearized_data.append(
            {
                "id": key,
                "input_text": text,
                "target_text": target,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(linearized_data, f, ensure_ascii=False, indent=4)

    print("Dataset linearizado e filtrado!")
    print(f"Exemplos válidos: {len(linearized_data)}")
    print(f"Exemplos excluídos (sem anotação): {excluidos}")


def linearizer(annotations):
    quads = []

    for ann in annotations:
        aspect = str(ann.get("aspect", {}).get("term") or "none").strip().lower()
        opinion = str(ann.get("sentiment", {}).get("term") or "none").strip().lower()
        category = str(ann.get("category") or "none").strip().lower()
        polarity = str(ann.get("polarity") or "none").strip().lower()

        quad = f"[A] {aspect} [O] {opinion} [C] {category} [P] {polarity}"
        quads.append(quad)

    return " [SSEP] ".join(quads)


# Execução
gerar_dataset_filtrado("train.json", "linearized.json")
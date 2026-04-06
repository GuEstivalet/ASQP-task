def aplicar_bt(text):
    """
    Lógica para Back Translation (Ex: PT -> EN -> PT).
    """
    return f"{text} [BT]" 

def aplicar_sr(text):
    """
    Lógica para Synonym Replacement.
    nlpaug usando o WordNet pt
    """
    # Exemplo mock (substitua pela chamada da lib)
    return f"{text} [SR]"

def gerar_datasets_experimentais(df_annot):
    """
    Gera as 5 combinações de datasets para os experimentos.
    """
    print("\n--- Iniciando Geração de Datasets (Augmentation) ---")
    
    # 1. Dataset raw
    ds_raw = df_annot.copy()
    
    # 2. BT
    print("Gerando BT...")
    ds_bt = df_annot.copy()
    ds_bt['text'] = ds_bt['text'].apply(aplicar_bt)
    
    # 3. SR
    print("Gerando SR...")
    ds_sr = df_annot.copy()
    ds_sr['text'] = ds_sr['text'].apply(aplicar_sr)
    
    # 4. BT + SR 
    print("Gerando BT + SR...")
    ds_bt_sr = ds_bt.copy()
    ds_bt_sr['text'] = ds_bt_sr['text'].apply(aplicar_sr)
    
    # 5. SR + BT 
    print("Gerando SR + BT...")
    ds_sr_bt = ds_sr.copy()
    ds_sr_bt['text'] = ds_sr_bt['text'].apply(aplicar_bt)
    
    datasets = {
        'original': ds_raw,
        'bt': ds_bt,
        'sr': ds_sr,
        'bt_sr': ds_bt_sr,
        'sr_bt': ds_sr_bt
    }
    
    for nome, ds in datasets.items():
        print(f"Dataset '{nome}' gerado com {len(ds)} amostras.")
        
    return datasets
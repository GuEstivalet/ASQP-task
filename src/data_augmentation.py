import pandas as pd

def balanced_da(df_annot, metodo="bt"):
    """
    Aplica aumento de dados para balancear cada combinação de Categoria e Polaridade.
    """
    # 1. Identificar a combinação mais frequente para servir de alvo (teto)
    contagem_pares = df_annot.groupby(['category', 'polarity']).size()
    max_samples = contagem_pares.max()
    print(max_samples)
    
    print(f"\n--- Iniciando Aumento ({metodo.upper()}) ---")
    print(f"Alvo de balanceamento: {max_samples} amostras por par (Categoria, Polaridade)")

    lista_dfs = [df_annot]

    # 2. Iterar sobre cada combinação existente no dataset
    for (cat, pol), total in contagem_pares.items():
        diferenca = max_samples - total
        
        if diferenca > 0:
            # Filtra apenas as amostras desse par específico
            df_subset = df_annot[(df_annot['category'] == cat) & (df_annot['polarity'] == pol)]
            
            # Sorteia amostras para aumentar (com reposição caso precise de mais do que tem)
            df_para_aumentar = df_subset.sample(diferenca, replace=True, random_state=42)
            
            # Aplica a transformação escolhida
            if metodo == "bt":
                df_para_aumentar['text'] = df_para_aumentar['text'].apply(aplicar_bt)
            elif metodo == "sr":
                df_para_aumentar['text'] = df_para_aumentar['text'].apply(aplicar_sr)
                
            lista_dfs.append(df_para_aumentar)
            print(f" > [{cat} | {pol}]: Adicionadas {diferenca} amostras.")

    # Concatena tudo em um novo dataset balanceado
    df_balanceado = pd.concat(lista_dfs).reset_index(drop=True)
    return df_balanceado

def gen_all_datasets_combinations(df_annot):
    """
    Gera as combinações solicitadas, todas buscando o equilíbrio.
    """
    datasets = {}
    
    # 1. RAW
    datasets['raw'] = df_annot.copy()
    
    # 2. BT 
    datasets['bt'] = balanced_da(df_annot, metodo="bt")
    
    # 3. SR 
    datasets['sr'] = balanced_da(df_annot, metodo="sr")
    
    # 4. BT + SR (maior potencial)
    print("\nGerando combinação BT + SR...")
    df_bt_base = balanced_da(df_annot, metodo="bt")
    datasets['bt_sr'] = balanced_da(df_bt_base, metodo="sr")
    
    # 5. SR + BT
    print("\nGerando combinação SR + BT...")
    df_sr_base = balanced_da(df_annot, metodo="sr")
    datasets['sr_bt'] = balanced_da(df_sr_base, metodo="bt")
    
    return datasets

def aplicar_bt(text):
    return text
    
    
def aplicar_sr(text):
    return text
import os
import sys
from src.analise_dataset import carregar_dados,verificar_nulos,processar_anotacoes,analisar_polaridade,analisar_categorias
from src.data_augmentation import gen_all_datasets_combinations
def main():
    
    # 1. Load dataset
    caminho = 'train.json'
    df_raw = carregar_dados(caminho)
    
    # 2. Verify null values
    verificar_nulos(df_raw)
    
    # 3. Processing and cleaning
    df_annot = processar_anotacoes(df_raw)
    
    # 4. Analysis
    analisar_polaridade(df_annot)
    analisar_categorias(df_annot)
    
    # 5. Experimental generation
    meus_datasets = gen_all_datasets_combinations(df_annot)
    
    
    # 6. Salvar os datasets 
    print("\n--- Salvando arquivos CSV ---")
    for nome, df_exp in meus_datasets.items():
        filename = f'dataset_{nome}.csv'
        df_exp.to_csv(filename, index=False)
        print(f"Arquivo gerado: {filename} | Total de linhas: {len(df_exp)}")

    print("\nTodos os datasets experimentais foram salvos com sucesso.")

    
if __name__ == "__main__":
    main()

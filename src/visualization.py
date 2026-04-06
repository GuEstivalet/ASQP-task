import matplotlib.pyplot as plt
import os

def salvar_grafico_barras(dados, titulo, nome_arquivo, xlabel="Categoria", ylabel="Frequência", log_scale=False):
    """
    Gera e salva um gráfico de barras a partir de um dicionário ou objeto similar.
    """
    plt.figure(figsize=(10, 6))
    
    # Extração de chaves e valores
    plt.bar(dados.keys(), dados.values())
    
    # Configurações de Escala
    if log_scale:
        plt.yscale("log")
        ylabel += " (log)"
    
    # Títulos e Eixos
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Formatação
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Salvamento
    caminho_completo = os.path.join(OUT_DIR, nome_arquivo)
    plt.savefig(caminho_completo, dpi=300)
    plt.close()
    print(f"Gráfico salvo em: {caminho_completo}")

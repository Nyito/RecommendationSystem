from rapidfuzz import fuzz

def combined_score(input_user, product_name):
    partial = fuzz.partial_ratio(input_user, product_name)
    token_sort = fuzz.token_sort_ratio(input_user, product_name)
    
    # Dar pesos diferentes a cada métrica (ajuste conforme o comportamento desejado)
    return 0.5 * partial + 0.5 * token_sort

# Exemplo de uso:
input_user = "Café"
product_name = "Café Torrado e Moído"
score = combined_score(input_user, product_name)
print(score)  # Resultado da combinação

from rapidfuzz import process

input_user = "Café"
products = ["Café Torrado e Moído", "Café Expresso", "Café Solúvel"]

best_matches = process.extract(input_user, products, scorer=fuzz.W_ratio, limit=3)
print(best_matches)  # Retorna as 3 melhores correspondências

# Definir thresholds diferentes para categorizar a qualidade da correspondência
thresholds = {
    'alta_qualidade': 80,
    'media_qualidade': 60
}

def classificar_score(score):
    if score >= thresholds['alta_qualidade']:
        return "Correspondência Forte"
    elif score >= thresholds['media_qualidade']:
        return "Correspondência Razoável"
    else:
        return "Correspondência Fraca"
    
from rapidfuzz import process, fuzz

products = ["Café Torrado e Moído", "Café Expresso", "Café Solúvel"]

def combined_score(input_user, product_name):
    w_ratio = fuzz.W_ratio(input_user, product_name)
    partial = fuzz.partial_ratio(input_user, product_name)
    token_sort = fuzz.token_sort_ratio(input_user, product_name)
    
    # Ajuste de ponderações
    return 0.5 * w_ratio + 0.3 * partial + 0.2 * token_sort

# Usando process.extract com uma função de score personalizada
input_user = "Café Moído"
best_matches = process.extract(input_user, products, scorer=combined_score, limit=3)
print(best_matches)

from rapidfuzz import process, fuzz

# Função para realizar os testes
def testar_inputs(input_user, products, divisor=None, scorer=fuzz.W_ratio, limit=3):
    # Caso haja divisor, quebrar o input em partes
    if divisor:
        inputs = [i.strip() for i in input_user.split(divisor)]
    else:
        inputs = [input_user]

    resultados = {}
    
    # Testar cada input individualmente
    for inp in inputs:
        best_matches = process.extract(inp, products, scorer=scorer, limit=limit)
        resultados[inp] = best_matches

    return resultados

# Base de dados de produtos
products = ["Café Torrado e Moído", "Café Expresso", "Açúcar Refinado", "Leite Desnatado", "Chá Verde"]

# Cenários de teste
cenarios = [
    {"input": "Café", "divisor": None},  # Input de uma palavra
    {"input": "Café; Açúcar; Leite", "divisor": ";"},  # Múltiplos inputs de uma palavra
    {"input": "Café Torrado e Moído", "divisor": None},  # Input de muitas palavras
    {"input": "Café Torrado; Açúcar Refinado; Leite Desnatado", "divisor": ";"},  # Múltiplos inputs de muitas palavras
    {"input": "Cafe", "divisor": None},  # Input com erro de digitação
]

# Testar cada cenário
for cenario in cenarios:
    input_test = cenario["input"]
    divisor = cenario["divisor"]
    
    print(f"Testando input: '{input_test}'")
    resultados = testar_inputs(input_test, products, divisor)
    
    for input_user, matches in resultados.items():
        print(f"Melhores correspondências para '{input_user}':")
        for match, score, _ in matches:
            print(f" - {match}: {score}")
    print("\n" + "-"*50 + "\n")


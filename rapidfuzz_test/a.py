from rapidfuzz import fuzz, process, utils
import pandas as pd

dataset = pd.read_csv('./data/CleanData.csv')

print(dataset['Product Name'][6875])

SCORERS = {
    "ratio": fuzz.ratio,
    "partial_ratio": fuzz.partial_ratio,
    "token_sort_ratio": fuzz.token_sort_ratio,
    "partial_token_sort_ratio": fuzz.partial_token_sort_ratio,
    "token_set_ratio": fuzz.token_set_ratio,
    "partial_token_set_ratio": fuzz.partial_token_set_ratio,
    "Q_ratio": fuzz.QRatio,
    "W_ratio": fuzz.WRatio,
}

def one_word_one_string(user_input, products, scorers, limit=3):

    teste1 = pd.DataFrame(columns=['input', 'scorer_name', 'scorer_value', 'product_name'])
    for input in user_input:
        for key, value in scorers.items():
            matches = process.extract(input, products, scorer=value, limit=limit, processor=utils.default_process)
            df_aux = pd.DataFrame({
                'input':[input] * len(matches),
                'scorer_name':[key] * len(matches), 
                'scorer_value':[match[1] for match in matches], 
                'product_name':[match[0] for match in matches]
            })
            teste1 = pd.concat([teste1, df_aux], ignore_index=True)
    return  teste1

def n_word_one_string(user_input, products, scorers, limit=3):
    
    teste2 = pd.DataFrame(columns=['input', 'scorer_name', 'scorer_value', 'product_name'])
    for input in user_input:
        for key, value in scorers.items():
            matches = process.extract(input, products, scorer=value, limit=limit, processor=utils.default_process)
            df_aux = pd.DataFrame({
                'input':[input] * len(matches),
                'scorer_name':[key] * len(matches), 
                'scorer_value':[match[1] for match in matches], 
                'product_name':[match[0] for match in matches]
            })
            teste2 = pd.concat([teste2, df_aux], ignore_index=True)
    return  teste2

def one_word_n_string(user_input, products, scorers, limit=3):

    teste3 = pd.DataFrame(columns=['input', 'scorer_name', 'scorer_value', 'product_name'])
    for input in user_input:
        for key, value in scorers.items():
            matches = process.extract(input, products, scorer=value, limit=limit, processor=utils.default_process)
            df_aux = pd.DataFrame({'input':input, 'scorer_name':key, 
                    'scorer_value':[matches[0][1], matches[1][1], matches[2][1]], 
                    'product_name':[matches[0][0], matches[1][0], matches[2][0]],
                    },index=[0])
            teste3 = pd.concat([teste3, df_aux], ignore_index=True)
    return  teste3

def n_word_n_string(user_input, products, scorers, limit=3):

    teste4 = pd.DataFrame(columns=['input', 'scorer_name', 'scorer_value', 'product_name'])
    for input in user_input:
        for key, value in scorers.items():
            matches = process.extract(input, products, scorer=value, limit=limit, processor=utils.default_process)
            df_aux = pd.DataFrame({'input':input, 'scorer_name':key, 
                    'scorer_value':[matches[0][1], matches[1][1], matches[2][1]], 
                    'product_name':[matches[0][0], matches[1][0], matches[2][0]],
                    },index=[0])
            teste4 = pd.concat([teste4, df_aux], ignore_index=True)
    return  teste4

products_list = dataset["Product Name"]

limit = 2

lista_df = []

scenarios = [
    ["Fantail"],
    ["Non Fiction Educational Games"],
    ["Learninc", "Turqooise", "Fection"],
    ["Fantaik Bools", "Junior Books", "Fiktion Action Ganes"]
]

df1 = one_word_one_string(user_input=scenarios[2], products=products_list, scorers=SCORERS, limit=limit)
df2 = n_word_one_string(user_input=scenarios[3], products=products_list, scorers=SCORERS, limit=limit)
# df3 = one_word_n_string(user_input=scenarios[2], products=products_list, scorers=SCORERS)
# df4 = n_word_n_string(user_input=scenarios[3], products=products_list, scorers=SCORERS)

lista_df.append(df1)
lista_df.append(df2)
# lista_df.append(df1)
# lista_df.append(df1)

df_final = pd.concat(lista_df, ignore_index=True)

print(df_final)

df_final.to_csv('./rapidfuzz_test/check2.csv', index=False)
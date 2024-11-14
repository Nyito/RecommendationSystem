## Developed algorithm libraries
import pandas as pd
import numpy as np
import nltk
import gensim
from gensim.models import Word2Vec
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from rapidfuzz import fuzz, process, utils


## References libraries
# import turicreate as tc
# from sklearn.model_selection import train_test_split
# import sys
# sys.path.append("..")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to load the dataset
def load_database():
    dataset = pd.read_csv("../data/AmazonData.csv")
    return dataset

# Function to process the dataset
def data_manipulation(dataset):
    # Excluding columns that we dont use
    cols = [0,2,3,5,6,8,9,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
    dataset.drop(dataset.columns[cols], axis =1, inplace=True)
    dataset.dropna(inplace = True)

    # Splitting Category in 3 parts
    new = dataset["Category"].str.split("|", n = 3, expand = True)
    
    # making the first category called Main Category
    dataset["Main Category"]= new[0] 
    
    # making the second category called sub_category 
    dataset["Sub Category"]= new[1]

    # making the third category called side_category 
    dataset["Side Category"]= new[2]

    # making the last column consist of the remaining categories
    dataset["Other Category"]= new[3]

    # Dropping old category columns and the remaining categories 
    dataset.drop(columns =["Category"], inplace = True)

    # Setting Column Selling Price as float value
    # Database Price and weight treatment
    dataset.rename(columns = {'Uniq Id':'Id','Shipping Weight':'Shipping Weight(Pounds)', 'Selling Price':'Selling Price($)'}, inplace = True)

    # Removing units from Price and Weight
    dataset['Shipping Weight(Pounds)'] = dataset['Shipping Weight(Pounds)'].str.strip('ounces')
    dataset['Shipping Weight(Pounds)'] = dataset['Shipping Weight(Pounds)'].str.strip('pounds')
    dataset['Selling Price($)'] = dataset['Selling Price($)'].str.replace('$', '')

    # Removing rows with Total Price invalid
    indexes = dataset[dataset['Selling Price($)'] == 'Total price:'].index
    dataset.drop(indexes, inplace=True)

    # Removing rows with '-' character
    dataset['Selling Price($)'] = dataset['Selling Price($)'].str.replace(',', '', regex=False)
    indexes = dataset[dataset['Selling Price($)'].str.contains('-', na=False)].index
    dataset.drop(indexes, inplace=True)

    # Removing rows with '&' character
    indexes = dataset[dataset['Selling Price($)'].str.contains('&', na=False)].index
    dataset.drop(indexes, inplace=True)

    # Removing rows with 'Currently' character
    indexes = dataset[dataset['Selling Price($)'].str.contains('Currently', na=False)].index
    dataset.drop(indexes, inplace=True)

    # Removing rows with 'from' character
    indexes = dataset[dataset['Selling Price($)'].str.contains('from', na=False)].index
    dataset.drop(indexes, inplace=True)

    # Adjusting values with wrong format
    dataset['Selling Price($)'] = dataset['Selling Price($)'].str.split(' ').str[0]
    dataset['Selling Price($)'] = dataset['Selling Price($)'].astype(float)

    # Setting Column Shipping Weight as float value
    indexes = dataset[dataset['Shipping Weight(Pounds)'].str.contains(r'\. ', na=False)].index

    dataset.at[1619, 'Shipping Weight(Pounds)']
    dataset.drop(1619, inplace=True)
    dataset['Shipping Weight(Pounds)'] = dataset['Shipping Weight(Pounds)'].str.replace(',', '', regex=False)
    dataset['Shipping Weight(Pounds)'] = dataset['Shipping Weight(Pounds)'].astype(float)

    return dataset
def save_data_manipulation(dataset):
    dataset.to_csv('../data/CleanData.csv', index=False)


# Requisito #02
def load_data():
    dataset = load_database()
    dataset = data_manipulation(dataset)
    return dataset

# Data split to sklearn
from sklearn.model_selection import train_test_split
def split_data_sets(dataset):
    train, test = train_test_split(dataset, test_size=0.2)
    return train, test

# Save data (optional)
def save_data_sets(train, test):
    train.to_csv("../data/train.csv", index=False)
    test.to_csv("../data/test.csv", index=False)

# Requisitos #03 (Our Recommendation Algorithm )

# Load data for training model
def load_data_parameters():
    # Aqui seria possível implementar a leitura dos parâmetros para o modelo
    return {'vector_size': 100, 'window': 5, 'min_count': 1,'workers': 4}

# Processing text function
def preprocess_text(text):
    text = text.replace('[^a-zA-Z]',' ').lower()
    stop_re = '\\b'+'\\b|\\b'.join(nltk.corpus.stopwords.words('english'))+'\\b'
    text = text.replace(stop_re, '')
    text = text.split()

    # Add lemmatization using WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in text]
    return lemmatized_text

# Treinamento do modelo base com Word2Vec
def train_word2vec_model(data_parameters, dataset):

    model = Word2Vec(sentences=dataset["Processed Product Name"], vector_size=data_parameters['vector_size'],window=data_parameters['window'], min_count=data_parameters['min_count'], workers=data_parameters['workers'])
    return model

# vectorizing text function
def vectorize_product(product_name, model):
    words = [word for word in product_name if word in model.wv]
    if len(words) > 0:
        return np.mean([model.wv[word] for word in words], axis=0)
    else:
        return np.zeros(model.wv.vector_size)
    
# Morfologic and Syntatic Filters
def prototype_rapid_fuzz_filter(user_input, products, number_of_rec):
    list_of_rec = []
    token_set_ratio_match = process.extract(user_input, products, scorer=fuzz.token_set_ratio, limit=1, processor=utils.default_process)
    partial_ratio_matches = process.extract(user_input, products, scorer=fuzz.token_set_ratio, limit=number_of_rec, processor=utils.default_process)

    if token_set_ratio_match[0][1] == 100:
        list_of_rec.extend(token_set_ratio_match)
        for match in partial_ratio_matches:
            if match[0] != token_set_ratio_match[0][0]:
                list_of_rec.append(match)
    
    else:
        for match in partial_ratio_matches:
            list_of_rec.append(match)

    return list_of_rec

def rapid_fuzz_rec_to_df(recommendations, dataframe):
    sorted_indeces = [match[2] for match in recommendations]
    reordered_df = dataframe.iloc[sorted_indeces].reset_index(drop=True)
    return reordered_df


# Product Recommendation function
def product_recommendation(product_vector, dataset, top_n=5):
    # Calcular similaridades cosseno
    similarities = dataset["Product Vector"].apply(lambda x: cosine_similarity([product_vector], [x])[0][0])
    
    # Ordenar por similaridade e pegar os top_n produtos
    top_indices = similarities.nlargest(top_n).index
    
    # Retornar o DataFrame com os produtos recomendados, mas mantendo os nomes originais
    return dataset.loc[top_indices, dataset.columns != 'Product Vector']


# Category Filter 
def category_filter(dataset, selected_product):

    main_category_input = selected_product['Main Category']
    sub_category_input = selected_product['Sub Category']
    side_category_input = selected_product['Side Category']
    other_category_input = selected_product['Other Category']

    # Create a new column to calculate the score
    dataset['score'] = 0

    # Raises the score if categories match
    dataset.loc[dataset['Main Category'] == main_category_input, 'score'] += 1
    dataset.loc[dataset['Sub Category'] == sub_category_input, 'score'] += 1
    dataset.loc[dataset['Side Category'] == side_category_input, 'score'] += 1
    dataset.loc[dataset['Other Category'] == other_category_input, 'score'] += 1

    # Sort the database based on the score
    category_filter = dataset.sort_values(by='score', ascending=False)

    max_score = category_filter['score'].max()

    # Filter rows with the maximum score
    category_filter = category_filter[category_filter['score'] == max_score]

    # Removes the new column
    category_filter = category_filter.drop(columns='score')

    # return the sorted database
    if category_filter.empty:
        print('No recommendaation found for this product.')
    else:
        return category_filter

def name_based_filter(dataset, product_name):

    parameters = load_data_parameters()
    # Applying text preprocess in dataset
    dataset["Processed Product Name"] = dataset["Product Name"].apply(preprocess_text)

    model = train_word2vec_model(parameters,dataset)

    # Applying vectorizing function in dataset
    dataset["Product Vector"] = dataset["Processed Product Name"].apply(lambda x: vectorize_product(x, model))

    # Pré-processando o nome do produto fornecido pelo usuário
    processed_product_name = preprocess_text(product_name)

    # Vetorizando o nome do produto fornecido (passar o modelo aqui também)
    product_vector = vectorize_product(processed_product_name, model)

    recommendation = product_recommendation(product_vector, dataset, dataset.shape[0])
    return recommendation

def save_main_model(model):
    # Create model pasta
    model.save("../data/main_model.model")

def load_main_model():
    # Create model pasta
    return Word2Vec.load("../data/main_model.model")


## MAIN TO RUN CODE
# pensar em forma de testar c�digo no dataset
dataset_1 = load_data()
dataset_2 = load_data()


# Streamlit layout
# Função principal do Streamlit para exibir o sistema de recomendação
st.sidebar.title("Product Recommendation System")
option = st.sidebar.radio("Select input method:", ("Select input from database", "Type product name"))

# Variáveis de saída
list_word2vec = []
list_rapidfuzz = []


if option == "Select input from database":
    product_line = st.sidebar.number_input("Type the product's number:", min_value=1, max_value=len(dataset_1), step=1) - 1
    
    if 0 <= product_line < len(dataset_1):
        selected_product = dataset_1.iloc[product_line]
        product_name = selected_product['Product Name']

        # Filter based on product name using Word2Vec and RapidFuzz
        category_filtered = category_filter(dataset_1, selected_product)
        name_based_filter_result = name_based_filter(category_filtered, product_name)
        #st.write("Word2Vec with Category Filter:")
        #st.write(name_based_filter_result['Product Name'][:5])

        list_word2vec = name_based_filter_result['Product Name'].tolist()
        list_rapidfuzz = prototype_rapid_fuzz_filter(user_input=product_name, products=dataset_2["Product Name"], number_of_rec=dataset_2.shape[0])

    else:
        st.write("Invalid line number.")

elif option == "Type product name":
    product_name = st.sidebar.text_input("Type the product's name:")
    
    if product_name:
        # Filter based on product name using Word2Vec and RapidFuzz
        name_based_filter_result = name_based_filter(dataset_1, product_name)
        #st.write("Word2Vec Filter:")
        #st.write(name_based_filter_result['Product Name'][:5])

        list_word2vec = name_based_filter_result['Product Name'].tolist()
        list_rapidfuzz = prototype_rapid_fuzz_filter(user_input=product_name, products=dataset_2["Product Name"], number_of_rec=dataset_2.shape[0])

# Generate final recommendation list
final_list = []
for i in range(len(list_word2vec)):
    for j in range(len(list_rapidfuzz)):
        if list_word2vec[i] == list_rapidfuzz[j][0]:
            final_list.append([list_word2vec[i], i + j])


# Título e exibição das recomendações no lado direito
st.header("Algorithm Recommendations")
if option == "Select input from database":
    st.markdown("<h2 style='font-weight:bold;'>Selected Product:</h2>", unsafe_allow_html=True)
    st.write("")  # This creates an empty line
    st.write(product_name)

    #st.write("Top recommendations based on selected product:")
#elif option == "Type product name":
    #st.write("Top recommendations based on product name:")

final_list.sort(key=lambda x: x[1])
st.markdown("<h2 style='font-weight:bold;'>Final Recommendation:</h2>", unsafe_allow_html=True)
for idx, item in enumerate(final_list[:10], start=1):
    st.write(f"{idx} - Product: {item[0]}, Score: {item[1]}")


### To run and create a page on local host, use the command: streamlit run app.py on terminal
## Make sure you are "main" folder, if you're not, then enter before running streamlit run app.py on terminal
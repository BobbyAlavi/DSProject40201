import json
import re
import os
import codecs
import pandas
from sklearn.metrics.pairwise import cosine_similarity



def advanced_tokenizer(text):
    pattern = r'\w+|[^\w\s]'
    return re.findall(pattern, text)

def tokenize_json(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Extract the text data (modify this based on the structure of your JSON)
    text_data = data['text_field']  # Replace 'text_field' with the actual key

    # Tokenize the text data
    tokens = advanced_tokenizer(text_data)
    return tokens

json_file_path = r"C:\Users\Asus\Desktop\DSProject\data.json"

tokens = tokenize_json(json_file_path)
print(tokens)

# Replace this with the path to the directory containing your documents
directory_path = r'C:\Users\Asus\Desktop\DSProject\data'

# Initialize an empty list to hold the text of each document
documents = []

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):  # Check if the file is a text file
        file_path = os.path.join(directory_path, filename)  # Get the full file path
        with codecs.open(file_path, 'r', encoding='utf-8', errors='ignore') as file:  # Open the file
            documents.append(file.read())  # Read the file and append its contents to the list

from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming 'documents' is now a list of your preprocessed text documents
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Convert matrix to an array for easier viewing
tfidf_array = tfidf_matrix.toarray()
print(tfidf_array)

# Get the feature names (words/terms from your corpus)
feature_names = vectorizer.get_feature_names_out()


# first_document_vector = tfidf_matrix[0]
# df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["TF-IDF"])
# df.sort_values(by=["TF-IDF"],ascending=False)


from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(query_vector, tfidf_matrix)

from difflib import SequenceMatcher

def similar_words(query, document):
    query_words = query.split()  # Assuming query is a string of words
    document_words = document.split()  # Assuming document is a string of words
    similarity_threshold = 0.8  # Set a threshold for similarity (adjust as needed)
    
    for query_word in query_words:
        for document_word in document_words:
            match_ratio = SequenceMatcher(None, query_word, document_word).ratio()
            if match_ratio > similarity_threshold:
                print(f"Query word: {query_word}, Document word: {document_word}, Similarity: {match_ratio}")

def query_document_similarity(query_vector, tfidf_matrix, query, document):
    # Calculate cosine similarities
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
    
    # Find similar words
    similar_words(query, document)
    
    return cosine_similarities




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import codecs



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


# Assuming 'documents' is now a list of your preprocessed text documents
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Convert matrix to an array for easier viewing
tfidf_array = tfidf_matrix.toarray()
print(tfidf_array)

# Get the feature names (words/terms from your corpus)
feature_names = vectorizer.get_feature_names_out()


# Convert sparse matrix to dense matrix for dimensionality reduction
tfidf_dense = tfidf_matrix.todense()

# Step 2: Apply Dimensionality Reduction
pca = PCA(n_components=2)
reduced_vectors_pca = pca.fit_transform(tfidf_dense)

# Step 3: Visualize the Results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(reduced_vectors_pca[:, 0], reduced_vectors_pca[:, 1])
plt.title('2D PCA of TF-IDF Vectors')

plt.show()


from sklearn.cluster import KMeans

# Number of clusters
k = 5  # Adjust this based on your specific needs

# Apply K-Means to your reduced vectors (choose either PCA or t-SNE reduced vectors)
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(reduced_vectors_pca)  # or reduced_vectors_tsne

plt.scatter(reduced_vectors_pca[:, 0], reduced_vectors_pca[:, 1], c=clusters, cmap='viridis')  # or reduced_vectors_tsne
plt.title('Document Clusters')
plt.show()


# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(reduced_vectors_pca[:, 0], reduced_vectors_pca[:, 1], c=clusters, cmap='viridis')

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, marker='x')

plt.title('Clustered Documents in 2D Space')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.colorbar(label='Cluster')
plt.show()
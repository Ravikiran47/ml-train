
# ==============================
# AI/ML Demo Program
# ==============================

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from nltk.stem import PorterStemmer

# ------------------------------
# Dataset
# ------------------------------
sentences = [
    "I love machine learning",
    "Machine learning is interesting",
    "Python is easy to learn",
    "I enjoy coding in Python"
]

labels = [1, 1, 0, 0]   # Dummy labels for Neural Network

# ------------------------------
# 1. Unsupervised Learning
# ------------------------------
print("\n===== K-Means Clustering =====")

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(sentences)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

print("Clusters:", kmeans.labels_)

# ------------------------------
# 2. Reinforcement Learning
# ------------------------------
print("\n===== Reinforcement Learning =====")

Q = np.zeros((2, 2))
reward = 10
learning_rate = 0.5

Q[0][1] = Q[0][1] + learning_rate * reward

print("Q Table:")
print(Q)

# ------------------------------
# 3. Stemming
# ------------------------------
print("\n===== Stemming =====")

stemmer = PorterStemmer()

for word in ["playing", "running", "studies"]:
    print(word, "->", stemmer.stem(word))

# ------------------------------
# 4. Lemmatization (Simple Version)
# ------------------------------
print("\n===== Lemmatization =====")

def simple_lemmatize(word):
    if word.endswith("ies"):
        return word[:-3] + "y"
    elif word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    elif word.endswith("ing"):
        return word[:-3]
    else:
        return word

for word in ["cars", "studies", "running"]:
    print(word, "->", simple_lemmatize(word))

# ------------------------------
# 5. Bag of Words
# ------------------------------
print("\n===== Bag of Words =====")

bow = CountVectorizer()
bow_matrix = bow.fit_transform(sentences)

print("Words:")
print(bow.get_feature_names_out())

print("Matrix:")
print(bow_matrix.toarray())

# ------------------------------
# 6. TF-IDF
# ------------------------------
print("\n===== TF-IDF =====")

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(sentences)

print("Words:")
print(tfidf.get_feature_names_out())

print("Matrix:")
print(tfidf_matrix.toarray())

# ------------------------------
# 7. Word2Vec (Simple Simulation)
# ------------------------------
print("\n===== Word2Vec =====")

unique_words = list(set(" ".join(sentences).lower().split()))

for word in unique_words:
    vector = np.random.rand(5)
    print(word, "->", vector)

# ------------------------------
# 8. Neural Network
# ------------------------------
print("\n===== Neural Network =====")

nn = MLPClassifier(max_iter=1000, random_state=42)

nn.fit(tfidf_matrix, labels)

prediction = nn.predict(tfidf_matrix)

print("Predictions:", prediction)y
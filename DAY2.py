
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

sentences = [
    "I love machine learning",
    "Machine learning is interesting",
    "Python is easy to learn",
    "I enjoy coding in Python"
]

labels = [1, 1, 0, 0]   # Dummy labels for Neural Network

#Unsupervised Learning
print("\n K-Means Clustering")

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(sentences)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

print("Clusters:", kmeans.labels_)

#Reinforcement Learning
print("\nReinforcement Learning ")

Q = np.zeros((2, 2))
reward = 10
learning_rate = 0.5

Q[0][1] = Q[0][1] + learning_rate * reward

print("Q Table:")
print(Q)

#Stemming
print("\nStemming")

stemmer = PorterStemmer()

for word in ["playing", "running", "studies"]:
    print(word, "->", stemmer.stem(word))

#Lemmatization 
print("\nLemmatization")

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
    
# Bag of Words
print("\n Bag of Words")

bow = CountVectorizer()
bow_matrix = bow.fit_transform(sentences)

print("Words:")
print(bow.get_feature_names_out())

print("Matrix:")
print(bow_matrix.toarray())

# TF-IDF
print("\n===== TF-IDF =====")

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(sentences)

print("Words:")
print(tfidf.get_feature_names_out())

print("Matrix:")
print(tfidf_matrix.toarray())

# Word2Vec 

print("\n===== Word2Vec =====")

unique_words = list(set(" ".join(sentences).lower().split()))

for word in unique_words:
    vector = np.random.rand(5)
    print(word, "->", vector)


#  Neural Network
print("\n Neural Network ")

nn = MLPClassifier(max_iter=1000, random_state=42)

nn.fit(tfidf_matrix, labels)

prediction = nn.predict(tfidf_matrix)

print("Predictions:", prediction)
# K-Means Clustering Plot

plt.figure(figsize=(6,5))

X_array = X.toarray()

plt.scatter(X_array[:,0], X_array[:,1],c=kmeans.labels_,cmap="viridis",s=100)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()



# Reinforcement Learning Plot (Q-Table)

plt.figure(figsize=(5,4))

plt.imshow(Q, cmap="Blues")
plt.colorbar()

plt.xticks([0,1],["Action 0","Action 1"])
plt.yticks([0,1],["State 0","State 1"])

plt.title("Q-Table")
plt.show()



# Bag of Words Plot

plt.figure(figsize=(8,5))

word_counts = bow_matrix.toarray().sum(axis=0)

plt.bar(bow.get_feature_names_out(), word_counts)

plt.title("Bag of Words")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)

plt.show()



# TF-IDF Plot

plt.figure(figsize=(8,5))

tfidf_scores = tfidf_matrix.toarray().mean(axis=0)

plt.bar(tfidf.get_feature_names_out(), tfidf_scores)

plt.title("TF-IDF Scores")
plt.xlabel("Words")
plt.ylabel("Average TF-IDF")
plt.xticks(rotation=45)

plt.show()



#Word2Vec Plot

plt.figure(figsize=(8,5))

words = unique_words
values = [np.random.rand() for _ in words]

plt.bar(words, values)

plt.title("Word2Vec (Simulated)")
plt.xlabel("Words")
plt.ylabel("Vector Value")
plt.xticks(rotation=45)

plt.show()



# Neural Network Predictions

plt.figure(figsize=(6,5))

plt.plot(labels, marker="o", label="Actual")
plt.plot(prediction, marker="x", label="Predicted")

plt.title("Neural Network Prediction")
plt.xlabel("Sentence")
plt.ylabel("Class")
plt.legend()
plt.grid(True)

plt.show()
1. !pip install nltk
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
text = "The quick brown fox jumps over the lazy dog"
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
words = word_tokenize(text)
filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
stemmed_words = [stemmer.stem(word) for word in filtered_words]
lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in filtered_words]
print("Original Words:", words)
print("Filtered Words (No Stop Words):", filtered_words)
print("Stemmed Words:", stemmed_words)
print("Lemmatized Words:", lemmatized_words)










2. !pip install scikit-learn  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
documents = [
    "Apples are sweet and crispy.",
    "Bananas are rich in potassium and sweet.",
    "Oranges are citrus fruits and juicy.",
    "Apples and bananas are popular fruits."
]
print("Bag of Words (BoW) Representation:")
count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform(documents)
print("Vocabulary:", count_vectorizer.get_feature_names_out())
print("\nBoW Matrix:")
print(bow_matrix.toarray())
print("\nTF-IDF Representation:")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print("\nCosine Similarity between Document 1 and Document 2:", cosine_sim[0][0])








3. !pip install gensim
import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
sentences = [
    "This is the first sentence in the document.",
    "This is the second sentence.",
    "Yet another sentence.",
    "And the final sentence.",
    "Word embeddings are useful for various NLP tasks.",
    "Word2Vec is a popular technique for generating word embeddings.",
    "Gensim library provides an implementation of Word2Vec.",
    "We will explore Word2Vec and other word embedding techniques."
]
tokenized_sentences = [sentence.lower().split() for sentence in sentences]
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=10)
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
vector = model.wv['sentence']
print("Vector representation of 'sentence':", vector)
similar_words = model.wv.most_similar('sentence', topn=3)
print("Words similar to 'sentence':", similar_words)
words = list(model.wv.key_to_index) 
X = model.wv[words]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
plt.figure(figsize=(12, 8))
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Word Embeddings Visualization using PCA")
plt.show()
4. def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i  
    for j in range(n + 1):
        dp[0][j] = j  
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,    
                    dp[i][j - 1] + 1,    
                    dp[i - 1][j - 1] + 1 
                )
    return dp[m][n], dp
def print_dp_table(s1, s2, dp):
    print("   " + "  ".join(f"{c}" for c in " " + s2))
    for i, row in enumerate(dp):
        row_str = " ".join(f"{cell:2}" for cell in row)
        prefix = s1[i - 1] if i > 0 else " "
        print(f"{prefix} {row_str}")
s1 = "kitten"
s2 = "sitting"
distance, dp_table = levenshtein_distance(s1, s2)
print(f"Levenshtein Distance between '{s1}' and '{s2}': {distance}\n")
print("DP Table:")
print_dp_table(s1, s2, dp_table)

5. import nltk
from nltk.util import ngrams
from collections import Counter
from nltk.tokenize import word_tokenize
nltk.download('punkt')
text = "Natural language processing enables computers to understand human language."
tokens = word_tokenize(text)
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))
bigrams = generate_ngrams(tokens, 2)
bigram_counts = Counter(bigrams)
print("Bigrams:\n", bigram_counts)
trigrams = generate_ngrams(tokens, 3)
trigram_counts = Counter(trigrams)
print("\nTrigrams:\n", trigram_counts)
n = 4  
ngrams_general = generate_ngrams(tokens, n)
ngram_counts_general = Counter(ngrams_general)
print("\nN-Grams (n={}):\n".format(n), ngram_counts_general)












6. from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
documents = [
    "AI is transforming industries.",
    "Natural language processing empowers machines to understand humans.",
    "AI and NLP have a great impact on technology."
]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
for index, doc in enumerate(documents):
    print(f"\nDocument {index + 1}: {doc}")
    similarities = [(i + 1, score) for i, score in enumerate(cosine_sim[index]) if i != index]
    for other_doc_index, similarity_score in similarities:
        print(f" - Similarity with Document {other_doc_index}: {similarity_score:.2f}")
















7. from collections import defaultdict
import math
train = [
    [('the', 'DET'), ('cat', 'NOUN'), ('sat', 'VERB')],
    [('a', 'DET'), ('dog', 'NOUN'), ('barked', 'VERB')],
    [('the', 'DET'), ('dog', 'NOUN'), ('ran', 'VERB')]
]
T = defaultdict(lambda: defaultdict(int))
E = defaultdict(lambda: defaultdict(int))
tags = defaultdict(int)
for s in train:
    p = '<s>'
    for w, t in s:
        T[p][t] += 1
        E[t][w] += 1
        tags[t] += 1
        p = t
    T[p]['</s>'] += 1
def log_norm(d):
    return {k: {kk: math.log(v / sum(d[k].values())) for kk, v in d[k].items()} for k in d}
T = log_norm(T)
E = log_norm(E)
tagset = list(tags)
def viterbi(s):
    V = [{}]
    P = {}
    for t in tagset:
        V[0][t] = T['<s>'].get(t, -1e9) + E[t].get(s[0], math.log(1e-6))
        P[t] = [t]
    for i in range(1, len(s)):
        V.append({})
        newP = {}
        for ct in tagset:
            prob, pt = max(
                ((V[i - 1][pt] + T[pt].get(ct, -1e9) + E[ct].get(s[i], math.log(1e-6))), pt)
                for pt in tagset
            )
            V[i][ct] = prob
            newP[ct] = P[pt] + [ct]
        P = newP
    final = max(V[-1], key=V[-1].get)
    return P[final]
test = ['the', 'dog', 'sat']
print("Predicted Tags:", list(zip(test, viterbi(test))))


















8. import nltk
from nltk import PCFG
from nltk.parse import ViterbiParser
grammar = PCFG.fromstring("""
S -> NP VP [1.0]
NP -> Det N [0.5] | N [0.5]
VP -> V NP [0.7] | V [0.3]
Det -> 'the' [1.0]
N -> 'dog' [0.5] | 'cat' [0.5]
V -> 'chased' [0.5] | 'slept' [0.5]
""")
parser = ViterbiParser(grammar)
sentence = ['the', 'dog', 'chased', 'the', 'cat']
for tree in parser.parse(sentence):
    print(tree)
    tree.pretty_print()















9. !pip install spacy
!python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load("en_core_web_sm")
def extract_phrases(text):
    doc = nlp(text)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    verb_phrases = []
    for token in doc:
        if token.pos_ == "VERB":
            start = token.i
            end = len(doc)
            for tok in doc[start:]:
                if tok.dep_ in ("punct",) and tok.text in (".", ";"):
                    end = tok.i
                    break
            span = doc[start:end + 1]
            verb_phrases.append(span.text.strip())
    return noun_phrases, verb_phrases
text = "The quick brown fox jumps over the lazy dog near the park."
noun_chunks, verb_chunks = extract_phrases(text)
print("Extracted Noun Phrases:")
for chunk in noun_chunks:
    print("-", chunk)
print("\nExtracted Verb Phrases:")
for chunk in verb_chunks:
    print("-", chunk)




10. 
import spacy
nlp = spacy.load("en_core_web_sm")
text = "Apple was founded by Steve Jobs and is headquartered in Cupertino, California. The iPhone 15 was released in 2023."
doc = nlp(text)
print("Named Entities:")
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")
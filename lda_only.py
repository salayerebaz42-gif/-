import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv("all_bugs_topics.csv")
X = df['clean_title']
y = df['topic']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

cv = CountVectorizer(max_features=1000)
X_train_counts = cv.fit_transform(X_train)
X_test_counts  = cv.transform(X_test)

lda = LatentDirichletAllocation(n_components=5, random_state=42)
X_train_vec = lda.fit_transform(X_train_counts)
X_test_vec  = lda.transform(X_test_counts)

models = {
    "Naive Bayes":    MultinomialNB(),
    "SVM":            SVC(kernel='linear', random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), random_state=42),
    "Random Forest":  RandomForestClassifier(n_estimators=100, random_state=42),
}

print("\n" + "="*55)
print("  النتائج - LDA فقط")
print("="*55)

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"  {name:<18} Acc={acc:.2%}  F1={f1:.2%}")

print("="*55)
print("تم بنجاح")
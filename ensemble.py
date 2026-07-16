import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv("all_bugs_topics.csv")
X = df['clean_title']
y = df['topic']

tfidf = TfidfVectorizer(max_features=1000)
X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42)

# تدريب النماذج الثلاثة الأفضل
nb  = MultinomialNB()
svm = SVC(kernel='linear', random_state=42, probability=True)
nn  = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)

nb.fit(X_train, y_train)
svm.fit(X_train, y_train)
nn.fit(X_train, y_train)

# Soft Voting
prob_nb  = nb.predict_proba(X_test)
prob_svm = svm.predict_proba(X_test)
prob_nn  = nn.predict_proba(X_test)

prob_avg = (prob_nb + prob_svm + prob_nn) / 3
y_pred_ensemble = nb.classes_[np.argmax(prob_avg, axis=1)]

y_test_vals = y_test.values

print("\n" + "="*60)
print("  مقارنة النماذج مع Ensemble")
print("="*60)

def show(name, y_pred):
    acc = accuracy_score(y_test_vals, y_pred)
    pre = precision_score(y_test_vals, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test_vals, y_pred, average='weighted', zero_division=0)
    f1  = f1_score(y_test_vals, y_pred, average='weighted', zero_division=0)
    print(f"  {name:<25} Acc={acc:.2%}  F1={f1:.2%}")

show("Naive Bayes",          nb.predict(X_test))
show("SVM",                  svm.predict(X_test))
show("Neural Network",       nn.predict(X_test))
show("Ensemble (NB+SVM+NN)", y_pred_ensemble)

print("="*60)
print("\n✓ تم بنجاح")
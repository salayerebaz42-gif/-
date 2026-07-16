import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv("all_bugs_topics.csv")
X = df['clean_title']
y = df['topic']

# TF-IDF على كل البيانات أولاً
tfidf = TfidfVectorizer(max_features=1000)
X_vec = tfidf.fit_transform(X)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42)

models = {
    "Naive Bayes":    MultinomialNB(),
    "SVM":            SVC(kernel='linear', random_state=42),
  "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), random_state=42),
    "Random Forest":  RandomForestClassifier(n_estimators=100, random_state=42),
}

print("\n" + "="*55)
print("  النتائج - TF-IDF فقط")
print("="*55)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"  {name:<18} Acc={acc:.4f}  P={pre:.4f}  R={rec:.4f}  F1={f1:.4f}")

print("="*55)
print("تم بنجاح")
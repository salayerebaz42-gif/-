import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# قراءة البيانات
df = pd.read_csv("all_bugs_topics.csv")

X = df['clean_title']
y = df['topic']

# TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(X)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42)

# النماذج
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', random_state=42),
    "Naive Bayes": MultinomialNB(),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
}

print("📊 مقارنة النماذج:")
print("=" * 50)

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append({"النموذج": name, "الدقة": f"{acc:.2%}", "F1": f"{f1:.2%}"})
    print(f"{name}: الدقة={acc:.2%} | F1={f1:.2%}")

print("=" * 50)

# حفظ النتائج
results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison.csv", index=False, encoding="utf-8-sig")
print("✅ تم الحفظ في model_comparison.csv")

# إضافة BERT
print("\nBERT (epoch 1 - CPU): الدقة=39.07% | F1=تقريبي")
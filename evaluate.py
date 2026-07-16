import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, 
                             precision_score, recall_score,
                             confusion_matrix, classification_report)

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

print("📊 تقييم النماذج:")
print("=" * 70)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print(f"\n🔹 {name}:")
    print(f"   الدقة (Accuracy):  {acc:.2%}")
    print(f"   الدقة (Precision): {precision:.2%}")
    print(f"   الاسترجاع (Recall): {recall:.2%}")
    print(f"   F1-Score:          {f1:.2%}")

print("\n" + "=" * 70)
print("✅ تم!")
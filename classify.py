import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# قراءة البيانات
df = pd.read_csv("all_bugs_topics.csv")

# إعداد البيانات
X = df['clean_title']
y = df['topic']

# TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(X)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42)

print(f"بيانات التدريب: {X_train.shape[0]}")
print(f"بيانات الاختبار: {X_test.shape[0]}")

# نموذج Random Forest
print("\n📊 Random Forest:")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"الدقة: {accuracy_score(y_test, y_pred_rf):.2%}")

# نموذج SVM
print("\n📊 SVM:")
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print(f"الدقة: {accuracy_score(y_test, y_pred_svm):.2%}")

print("\n✅ تم!")
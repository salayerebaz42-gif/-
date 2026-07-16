import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("all_bugs_topics.csv")
X = df['clean_title']
y = df['topic']

tfidf = TfidfVectorizer(max_features=1000)
X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42)

X_text_train, X_text_test, _, _ = train_test_split(
    X, y, test_size=0.2, random_state=42)

models = {
    "Naive Bayes":    MultinomialNB(),
    "SVM":            SVC(kernel='linear', random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), random_state=42),
    "Random Forest":  RandomForestClassifier(n_estimators=100, random_state=42),
}

preds = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds[name] = model.predict(X_test)

y_test_vals = y_test.values
X_test_texts = X_text_test.values

print("\n" + "="*60)
print("  توزيع الأخطاء بحسب الموضوع")
print("="*60)

error_data = []
for topic in sorted(df['topic'].unique()):
    row = {"Topic": f"Topic {topic}"}
    for name in models:
        mask = (y_test_vals == topic)
        errors = sum(preds[name][mask] != y_test_vals[mask])
        row[name] = errors
    error_data.append(row)

error_df = pd.DataFrame(error_data)
print(error_df.to_string(index=False))

print("\n" + "="*60)
print("  إجمالي الأخطاء لكل نموذج")
print("="*60)
for name in models:
    total = sum(preds[name] != y_test_vals)
    print(f"  {name:<18} أخطاء={total}/{len(y_test_vals)}")

print("\n" + "="*60)
print("  أمثلة على أخطاء Naive Bayes")
print("="*60)
count = 0
for i in range(len(y_test_vals)):
    if preds["Naive Bayes"][i] != y_test_vals[i] and count < 5:
        print(f"  النص:    {X_test_texts[i]}")
        print(f"  الصحيح:  Topic {y_test_vals[i]}")
        print(f"  المتوقع: Topic {preds['Naive Bayes'][i]}")
        print(f"  {'-'*45}")
        count += 1

print("\n✓ تم بنجاح")
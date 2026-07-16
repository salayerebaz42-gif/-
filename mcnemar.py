import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

df = pd.read_csv("all_bugs_topics.csv")
X = df['clean_title']
y = df['topic']

tfidf = TfidfVectorizer(max_features=1000)
X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42)

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

def mcnemar_test(name1, name2):
    p1 = (preds[name1] == y_test_vals)
    p2 = (preds[name2] == y_test_vals)
    b = sum(p1 & ~p2)
    c = sum(~p1 & p2)
    table = [[sum(p1 & p2), b],
             [c, sum(~p1 & ~p2)]]
    result = mcnemar(table, exact=False, correction=True)
    sig = "معنادار ✓" if result.pvalue < 0.05 else "غير معنادار ✗"
    print(f"  {name1:<18} vs {name2:<18} | p={result.pvalue:.4f} | {sig}")

print("\n" + "="*65)
print("  نتائج آزمون McNemar")
print("="*65)
mcnemar_test("Naive Bayes",    "SVM")
mcnemar_test("Naive Bayes",    "Neural Network")
mcnemar_test("Naive Bayes",    "Random Forest")
mcnemar_test("SVM",            "Neural Network")
mcnemar_test("SVM",            "Random Forest")
mcnemar_test("Neural Network", "Random Forest")
print("="*65)
print("\n✓ تم بنجاح")
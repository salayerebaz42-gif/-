import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("all_bugs_topics.csv")

firefox = df[df['project'] == 'Mozilla Firefox']
eclipse = df[df['project'] == 'Eclipse JDT']
vscode  = df[df['project'] == 'VS Code']

models = {
    "Naive Bayes":    MultinomialNB(),
    "SVM":            SVC(kernel='linear', random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), random_state=42),
    "Random Forest":  RandomForestClassifier(n_estimators=100, random_state=42),
}

def run(train_df, test_df, scenario):
    print(f"\n{'='*50}")
    print(f"  {scenario}")
    print(f"{'='*50}")
    results = []
    tfidf = TfidfVectorizer(max_features=1000)
    X_train = tfidf.fit_transform(train_df['clean_title'])
    X_test  = tfidf.transform(test_df['clean_title'])
    y_train = train_df['topic']
    y_test  = test_df['topic']
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        print(f"  {name:<18} Acc={acc:.2%}  F1={f1:.2%}")
        results.append({"Scenario": scenario, "Model": name,
                        "Accuracy": round(acc*100,2), "F1": round(f1*100,2)})
    return results

all_results = []

train1 = pd.concat([firefox, eclipse])
all_results += run(train1, vscode, "Train: Firefox+Eclipse  |  Test: VSCode")

train2 = pd.concat([firefox, vscode])
all_results += run(train2, eclipse, "Train: Firefox+VSCode   |  Test: Eclipse")

train3 = pd.concat([eclipse, vscode])
all_results += run(train3, firefox, "Train: Eclipse+VSCode   |  Test: Firefox")

results_df = pd.DataFrame(all_results)
results_df.to_csv("cross_project_results.csv", index=False, encoding="utf-8-sig")
print("\n✓ تم الحفظ في cross_project_results.csv")
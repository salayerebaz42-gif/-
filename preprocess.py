import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ─── تحميل البيانات ───────────────────────────────────────────
df = pd.read_csv("all_bugs_topics.csv")
X = df['clean_title']
y = df['topic']

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─── النماذج ──────────────────────────────────────────────────
models = {
    "Naive Bayes":    MultinomialNB(),
    "SVM":            SVC(kernel='linear', random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), random_state=42),
    "Random Forest":  RandomForestClassifier(n_estimators=100, random_state=42),
}

# ─── دالة التقييم ─────────────────────────────────────────────
def evaluate(X_tr, X_te, y_tr, y_te, mode_name):
    print(f"\n{'='*55}")
    print(f"  الوضع: {mode_name}")
    print(f"{'='*55}")
    results = []
    for name, model in models.items():
        # Naive Bayes لا يقبل قيماً سالبة → نتحقق
        if name == "Naive Bayes":
            model.fit(abs(X_tr) if hasattr(X_tr, 'toarray') else X_tr, y_tr)
            y_pred = model.predict(abs(X_te) if hasattr(X_te, 'toarray') else X_te)
        else:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        pre = precision_score(y_te, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_te, y_pred, average='weighted', zero_division=0)
        f1  = f1_score(y_te, y_pred, average='weighted', zero_division=0)

        print(f"  {name:<18} Acc={acc:.2%}  P={pre:.2%}  R={rec:.2%}  F1={f1:.2%}")
        results.append({
            "Mode":      mode_name,
            "Model":     name,
            "Accuracy":  round(acc * 100, 2),
            "Precision": round(pre * 100, 2),
            "Recall":    round(rec * 100, 2),
            "F1-Score":  round(f1  * 100, 2),
        })
    return results

all_results = []

# ══════════════════════════════════════════════════════════════
# الوضع 1: TF-IDF فقط
# ══════════════════════════════════════════════════════════════
print("\n>>> جاري حساب TF-IDF فقط...")
tfidf = TfidfVectorizer(max_features=1000)
X_tfidf_train = tfidf.fit_transform(X_train_text)
X_tfidf_test  = tfidf.transform(X_test_text)

all_results += evaluate(X_tfidf_train, X_tfidf_test, y_train, y_test, "TF-IDF فقط")

# ══════════════════════════════════════════════════════════════
# الوضع 2: LDA فقط
# ══════════════════════════════════════════════════════════════
print("\n>>> جاري حساب LDA فقط...")
cv = CountVectorizer(max_features=1000)
X_counts_train = cv.fit_transform(X_train_text)
X_counts_test  = cv.transform(X_test_text)

lda = LatentDirichletAllocation(n_components=5, random_state=42)
X_lda_train = lda.fit_transform(X_counts_train)
X_lda_test  = lda.transform(X_counts_test)

all_results += evaluate(X_lda_train, X_lda_test, y_train, y_test, "LDA فقط")

# ══════════════════════════════════════════════════════════════
# الوضع 3: TF-IDF + LDA معاً
# ══════════════════════════════════════════════════════════════
print("\n>>> جاري حساب TF-IDF + LDA معاً...")
tfidf2 = TfidfVectorizer(max_features=1000)
X_tfidf2_train = tfidf2.fit_transform(X_train_text)
X_tfidf2_test  = tfidf2.transform(X_test_text)

cv2 = CountVectorizer(max_features=1000)
X_counts2_train = cv2.fit_transform(X_train_text)
X_counts2_test  = cv2.transform(X_test_text)

lda2 = LatentDirichletAllocation(n_components=5, random_state=42)
X_lda2_train = lda2.fit_transform(X_counts2_train)
X_lda2_test  = lda2.transform(X_counts2_test)

X_combined_train = hstack([X_tfidf2_train, csr_matrix(X_lda2_train)])
X_combined_test  = hstack([X_tfidf2_test,  csr_matrix(X_lda2_test)])

all_results += evaluate(X_combined_train, X_combined_test, y_train, y_test, "TF-IDF + LDA")

# ══════════════════════════════════════════════════════════════
# جدول النتائج النهائي
# ══════════════════════════════════════════════════════════════
print("\n\n" + "="*55)
print("  جدول Ablation Study - النتائج الكاملة")
print("="*55)

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

results_df.to_csv("ablation_study_results.csv", index=False, encoding="utf-8-sig")
print("\n✓ تم حفظ النتائج في: ablation_study_results.csv")

# ══════════════════════════════════════════════════════════════
# ملخص: بهترین وضع لكل نموذج
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  ملخص: أفضل وضع لكل نموذج (حسب F1-Score)")
print("="*55)
best = results_df.loc[results_df.groupby("Model")["F1-Score"].idxmax()]
print(best[["Model", "Mode", "F1-Score"]].to_string(index=False))
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# قراءة البيانات
df = pd.read_csv("all_bugs_preprocessed.csv")
print(f"عدد التقارير: {len(df)}")

# تطبيق TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_title'])

print(f"✅ حجم المصفوفة: {X.shape}")
print(f"✅ عدد الميزات: {X.shape[1]}")
print("أهم الكلمات:")
print(vectorizer.get_feature_names_out()[:20])
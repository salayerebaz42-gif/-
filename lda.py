import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# قراءة البيانات
df = pd.read_csv("all_bugs_preprocessed.csv")

# تحويل النصوص
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_title'])

# تطبيق LDA
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

# عرض الموضوعات
print("📊 الموضوعات المكتشفة:")
words = vectorizer.get_feature_names_out()
for i, topic in enumerate(lda.components_):
    top_words = [words[j] for j in topic.argsort()[-10:]]
    print(f"\nالموضوع {i+1}: {', '.join(top_words)}")

# تخصيص كل تقرير لموضوع
df['topic'] = lda.transform(X).argmax(axis=1) + 1
df.to_csv("all_bugs_topics.csv", index=False)
print("\n✅ تم الحفظ في all_bugs_topics.csv")
print(df['topic'].value_counts())
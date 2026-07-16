import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# تحميل المكتبات
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# قراءة البيانات
df = pd.read_csv("all_bugs.csv")
print(f"عدد التقارير: {len(df)}")

# دالة تنظيف النص
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isna(text):
        return ""
    # حذف HTML
    text = re.sub(r'<.*?>', '', str(text))
    # حذف رموز خاصة
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # تحويل لحروف صغيرة
    text = text.lower()
    # Tokenization
    words = text.split()
    # حذف Stop words + Lemmatization
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# تطبيق التنظيف
df['clean_title'] = df['title'].apply(clean_text)

# معالجة القيم المفقودة
df = df[df['clean_title'].str.len() > 0]
df = df.reset_index(drop=True)

# حفظ النتيجة
df.to_csv("all_bugs_preprocessed.csv", index=False, encoding="utf-8")
print(f"بعد المعالجة: {len(df)} تقرير")
print("✅ تم الحفظ في all_bugs_preprocessed.csv")
print(df[['title', 'clean_title']].head())
from torch.optim import AdamW

# إعداد التدريب
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"جهاز التدريب: {device}")

model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# تدريب epoch واحد
print("جاري التدريب...")
model.train()
for batch in train_loader:
    input_ids, attention_mask, labels = [b.to(device) for b in batch]
    outputs = model(input_ids=input_ids, 
                   attention_mask=attention_mask, 
                   labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# تقييم النموذج
print("جاري التقييم...")
model.eval()
preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds.extend(outputs.logits.argmax(dim=1).cpu().numpy())

print(f"✅ دقة BERT: {accuracy_score(y_test, preds):.2%}")
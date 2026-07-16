import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

# ==============================
# 1. قراءة البيانات
# ==============================
df = pd.read_csv("all_bugs_topics.csv")
df['topic'] = df['topic'] - 1

X = df['clean_title'].tolist()
y = df['topic'].tolist()

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"بيانات التدريب: {len(X_train)}")
print(f"بيانات الاختبار: {len(X_test)}")

# ==============================
# 2. تجهيز BERT
# ==============================
print("جاري تحميل BERT...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=32,   # تقليل الطول لتسريع CPU
        return_tensors='pt'
    )

train_enc = tokenize(X_train)
test_enc = tokenize(X_test)

train_dataset = TensorDataset(
    train_enc['input_ids'],
    train_enc['attention_mask'],
    torch.tensor(y_train)
)

test_dataset = TensorDataset(
    test_enc['input_ids'],
    test_enc['attention_mask'],
    torch.tensor(y_test)
)

# ==============================
# 3. بناء النموذج
# ==============================
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=5
)

# ⚠️ تشغيل على CPU فقط
device = torch.device('cpu')
print(f"جهاز التدريب: {device}")

model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# ==============================
# 4. التدريب (Epoch واحد)
# ==============================
print("جاري التدريب...")
model.train()

for i, batch in enumerate(train_loader):
    input_ids, attention_mask, labels = [b.to(device) for b in batch]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    loss = outputs.loss
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if i % 20 == 0:
        print(f"Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")

# ==============================
# 5. التقييم
# ==============================
print("جاري التقييم...")
model.eval()

preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds.extend(outputs.logits.argmax(dim=1).cpu().numpy())

# ==============================
# 6. النتائج
# ==============================
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average='weighted')
recall = recall_score(y_test, preds, average='weighted')
f1 = f1_score(y_test, preds, average='weighted')

print("=" * 50)
print(f"✅ Accuracy:  {accuracy:.2%}")
print(f"✅ Precision: {precision:.2%}")
print(f"✅ Recall:    {recall:.2%}")
print(f"✅ F1-Score:  {f1:.2%}")
print("=" * 50)import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

# ==============================
# 1. قراءة البيانات
# ==============================
df = pd.read_csv("all_bugs_topics.csv")
df['topic'] = df['topic'] - 1

X = df['clean_title'].tolist()
y = df['topic'].tolist()

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"بيانات التدريب: {len(X_train)}")
print(f"بيانات الاختبار: {len(X_test)}")

# ==============================
# 2. تجهيز BERT
# ==============================
print("جاري تحميل BERT...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=32,   # تقليل الطول لتسريع CPU
        return_tensors='pt'
    )

train_enc = tokenize(X_train)
test_enc = tokenize(X_test)

train_dataset = TensorDataset(
    train_enc['input_ids'],
    train_enc['attention_mask'],
    torch.tensor(y_train)
)

test_dataset = TensorDataset(
    test_enc['input_ids'],
    test_enc['attention_mask'],
    torch.tensor(y_test)
)

# ==============================
# 3. بناء النموذج
# ==============================
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=5
)

# ⚠️ تشغيل على CPU فقط
device = torch.device('cpu')
print(f"جهاز التدريب: {device}")

model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# ==============================
# 4. التدريب (Epoch واحد)
# ==============================
print("جاري التدريب...")
model.train()

for i, batch in enumerate(train_loader):
    input_ids, attention_mask, labels = [b.to(device) for b in batch]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    loss = outputs.loss
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if i % 20 == 0:
        print(f"Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")

# ==============================
# 5. التقييم
# ==============================
print("جاري التقييم...")
model.eval()

preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds.extend(outputs.logits.argmax(dim=1).cpu().numpy())

# ==============================
# 6. النتائج
# ==============================
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average='weighted')
recall = recall_score(y_test, preds, average='weighted')
f1 = f1_score(y_test, preds, average='weighted')

print("=" * 50)
print(f"✅ Accuracy:  {accuracy:.2%}")
print(f"✅ Precision: {precision:.2%}")
print(f"✅ Recall:    {recall:.2%}")
print(f"✅ F1-Score:  {f1:.2%}")
print("=" * 50)
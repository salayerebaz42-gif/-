import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

# تحميل البيانات
df = pd.read_csv("all_bugs_topics.csv")
df['topic'] = df['topic'] - 1
X = df['clean_title'].tolist()
y = df['topic'].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True,
                    max_length=64, return_tensors='pt')

train_enc = tokenize(X_train)
test_enc = tokenize(X_test)

train_dataset = TensorDataset(
    train_enc['input_ids'],
    train_enc['attention_mask'],
    torch.tensor(y_train))

test_dataset = TensorDataset(
    test_enc['input_ids'],
    test_enc['attention_mask'],
    torch.tensor(y_test))

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# التدريب مع تسجيل النتائج
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    train_loss = total_loss/len(train_loader)

    # تقييم
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=labels)
            val_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = val_loss/len(test_loader)
    accuracy = correct/total * 100
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {accuracy:.2f}%")
import pandas as pd

df = pd.read_csv("eclipse_bugs.csv.csv")
print(f"قبل التنظيف: {len(df)} سجل")

df = df.dropna(subset=["short_desc"])
df = df.drop_duplicates(subset=["bug_id"])
df = df.reset_index(drop=True)

df.to_csv("eclipse_clean.csv", index=False, encoding="utf-8")
print(f"بعد التنظيف: {len(df)} سجل")
print("تم!")
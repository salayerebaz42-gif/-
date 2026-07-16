import pandas as pd

df = pd.read_csv("mozilla_bugs.csv.csv")
print(f"قبل التنظيف: {len(df)} سجل")
df = df.dropna(subset=["short_desc"])
df = df.drop_duplicates(subset=["bug_id"])
df = df.reset_index(drop=True)
df.to_csv("mozilla_clean.csv", index=False, encoding="utf-8")
print(f"بعد التنظيف: {len(df)} سجل")
print("تم!")
#  الكود الجوه  لاضهار الجدول الاخطاء    
import pandas as pd

df = pd.read_csv("mozilla_bugs.csv.csv")

# عرض كل الأخطاء
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print(df[["bug_id", "short_desc", "bug_type", "bug_status"]].to_string())
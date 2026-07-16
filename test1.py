import pandas as pd

mozilla = pd.read_csv("mozilla_clean.csv")
eclipse = pd.read_csv("eclipse_clean.csv")
vscode = pd.read_csv("bugs_clean.csv")

# التوزيع الزمني Mozilla
mozilla['changeddate'] = pd.to_datetime(mozilla['changeddate'])
mozilla['year'] = mozilla['changeddate'].dt.year
print("📊 التوزيع الزمني Mozilla:")
print(mozilla['year'].value_counts().sort_index())

print("=" * 50)

# التوزيع الزمني Eclipse
eclipse['changeddate'] = pd.to_datetime(eclipse['changeddate'])
eclipse['year'] = eclipse['changeddate'].dt.year
print("📊 التوزيع الزمني Eclipse:")
print(eclipse['year'].value_counts().sort_index())
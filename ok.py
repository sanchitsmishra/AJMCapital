import pandas as pd

df = pd.read_csv("data/raw/loan_prediction.csv")

print(df["Loan_Status"].value_counts())
print(df["Loan_Status"].value_counts(normalize=True))
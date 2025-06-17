import pandas as pd
import csv

crime_data = pd.read_csv("data_source/crime_district.csv")

transaction_data = pd.read_excel("data_source/Open Transaction Data.xlsx")

transaction_data.fillna(method='ffill', inplace=True)

print(transaction_data.head())
print(crime_data.head())


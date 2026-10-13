import pandas as pd

df = pd.read_csv("hotel_bookings.csv")

print("rows, columns:", df.shape)
print("\ncolumn names and types:")
print(df.dtypes)
print("\nnulls per column (only columns that have them):")
print(df.isna().sum()[lambda s: s > 0])
print("\ncancellation rate:", df["is_canceled"].mean().round(4))
print("\ndate range:", df["arrival_date_year"].min(), "to", df["arrival_date_year"].max())
print("\nnumeric summary:")
print(df.describe())

import pandas as pd
df = pd.read_csv("hotel_bookings.csv")

# Is Portugal actually the biggest market?
print(df["country"].value_counts().head())

# Does the top country cancel more than the rest?
top = df["country"].value_counts().idxmax()
print("most common country:", top)
print("cancel rate, that country:", df[df["country"] == top]["is_canceled"].mean().round(3))
print("cancel rate, everyone else:", df[df["country"] != top]["is_canceled"].mean().round(3))

import pandas as pd
df = pd.read_csv("hotel_bookings.csv")

counts = df["country"].value_counts()
top10 = counts.head(10)

print("top 10 countries by booking count:")
print(top10)

print("\neach as % of all rows:")
print((top10 / len(df) * 100).round(2))

print("\ntop 10 combined:", round(top10.sum() / len(df) * 100, 2), "%")
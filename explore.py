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
import pandas as pd
from tabulate import tabulate
import numpy as np

df = pd.read_csv('titanic_raw.csv')

# filling a null values using fillna()
# df["Cabin"].fillna("Unknown", inplace = True)
df['Age'].fillna(value=round(df.Age.mean()), inplace=True)
df['Sex'].replace('male', 0, inplace=True)
df['Sex'].replace('female', 1, inplace=True)
df.drop('Ticket', axis=1, inplace=True)
df.drop('Name', axis=1, inplace=True)
df.drop('Embarked', axis=1, inplace=True)
df.drop('Cabin', axis=1, inplace=True)

# Categorizing different price classes :
bins = [-1, 10, 30, 50, 100, np.inf]
labels = [1, 2, 3, 4, 5]
df['Pricerange'] = pd.cut(df['Fare'], bins, labels=labels)
df.drop('Fare', axis=1, inplace=True)

df.drop('PassengerId', axis=1, inplace=True)

temp_cols=df.columns.tolist()
new_cols=temp_cols[1:] + temp_cols[0:1]
df=df[new_cols]

# print(tabulate(df, headers='keys'))
df.to_csv('titanic.csv', index=False)


# ============================================
# TASK 2 - Titanic Dataset EDA
# ============================================


# --------------------------------------------
# STEP 1: Import Libraries
# --------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------
# STEP 2: Load Dataset
# --------------------------------------------

df = pd.read_csv("Titanic_Dataset.csv")


# --------------------------------------------
# STEP 3: Preview Dataset
# --------------------------------------------

print("Dataset Preview:\n")
print(df.head())


# --------------------------------------------
# STEP 4: Dataset Information
# --------------------------------------------

print("\nDataset Information:\n")
print(df.info())


# --------------------------------------------
# STEP 5: Check Missing Values
# --------------------------------------------

print("\nMissing Values:\n")
print(df.isnull().sum())


# --------------------------------------------
# STEP 6: Data Cleaning
# --------------------------------------------

# Fill missing Age values with mean age
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill missing Embarked values with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column because too many missing values
df.drop(columns='Cabin', inplace=True)


# --------------------------------------------
# STEP 7: Verify Missing Values Again
# --------------------------------------------

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())


# --------------------------------------------
# STEP 8: Survival Count
# --------------------------------------------

plt.figure(figsize=(6,4))

sns.countplot(x='Survived', data=df)

plt.title("Survival Count")

plt.show()


# --------------------------------------------
# STEP 9: Gender vs Survival
# --------------------------------------------

plt.figure(figsize=(6,4))

sns.countplot(x='Sex', hue='Survived', data=df)

plt.title("Gender vs Survival")

plt.show()


# --------------------------------------------
# STEP 10: Passenger Class vs Survival
# --------------------------------------------

plt.figure(figsize=(6,4))

sns.countplot(x='Pclass', hue='Survived', data=df)

plt.title("Passenger Class vs Survival")

plt.show()


# --------------------------------------------
# STEP 11: Age Distribution
# --------------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(df['Age'], bins=20)

plt.title("Age Distribution")

plt.xlabel("Age")

plt.show()


# --------------------------------------------
# STEP 12: Correlation Heatmap
# --------------------------------------------

# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(10,6))

sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()


# --------------------------------------------
# STEP 13: Final Insights
# --------------------------------------------

print("""
Insights:
1. Female passengers had higher survival rates.
2. Passengers in higher classes survived more.
3. Younger passengers formed a large portion of the dataset.
4. Data cleaning is important before analysis.
5. Visualizations help identify trends and patterns easily.
""")
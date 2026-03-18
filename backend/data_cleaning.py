import pandas as pd
import numpy as np


def clean_data(df):
    """
    Perform smart data cleaning operations
    """

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Handle missing values
    numeric_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns

    # Fill numeric missing values using median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values using most frequent value
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Remove outliers using IQR method
    for col in numeric_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df
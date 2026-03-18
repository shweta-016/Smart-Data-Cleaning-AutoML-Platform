from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd


def preprocess_data(df):
    """
    Encode categorical variables and split dataset
    """

    df = df.copy()

    # Encode categorical columns
    for col in df.select_dtypes(include="object").columns:

        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Last column assumed as target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Iterable, Tuple


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["customerID"], errors="ignore")

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace(["", " ", "nan"], np.nan)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(method="ffill").fillna(0.0)

    return df


def encode_categorical(df: pd.DataFrame, columns: Iterable[str] = None) -> pd.DataFrame:
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=["object"]).columns.tolist()

    encoder = LabelEncoder()
    for col in columns:
        df[col] = encoder.fit_transform(df[col].astype(str))

    return df


def scale_numeric(df: pd.DataFrame, columns: Iterable[str] = None) -> pd.DataFrame:
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()

    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])

    return df


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"MonthlyCharges", "tenure"}.issubset(df.columns):
        df["MonthlyChargeWithTenure"] = df["MonthlyCharges"] * df["tenure"]

    contract_map = {0: 1, 1: 12, 2: 24}
    if "Contract" in df.columns:
        df["ContractLength"] = df["Contract"].map(contract_map).fillna(0).astype(int)

    def contract_tenure_risk(row):
        contract = row.get("Contract", 0)
        tenure = row.get("tenure", 0)

        if contract == 0:
            if tenure <= 6:
                return "High"
            if 7 <= tenure <= 12:
                return "Medium"
            return "Low"
        if contract == 1:
            if tenure <= 6:
                return "Medium"
            if 7 <= tenure <= 12:
                return "Low"
            return "Very Low"
        if contract == 2:
            if tenure <= 12:
                return "Low"
            return "Very Low"
        return "Low"

    if {"Contract", "tenure"}.issubset(df.columns):
        df["ContractTenureRisk"] = df.apply(contract_tenure_risk, axis=1)
        df["ContractTenureRisk"] = df["ContractTenureRisk"].replace(
            {"Very Low": 0, "Low": 1, "Medium": 2, "High": 3}
        )

    return df


def drop_uninformative_features(df: pd.DataFrame, drop_columns: Iterable[str] = None) -> pd.DataFrame:
    if drop_columns is None:
        drop_columns = ["PhoneService", "gender", "StreamingTV", "StreamingMovies", "MultipleLines"]
    return df.drop(columns=[c for c in drop_columns if c in df.columns], errors="ignore")


def split_features_target(df: pd.DataFrame, target: str = "Churn") -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target].copy()
    return X, y

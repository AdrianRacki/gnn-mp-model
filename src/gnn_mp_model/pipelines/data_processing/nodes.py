"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.6
"""
import typing as t

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split


def _drop_columns_main(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(
        columns=[
            "CASRN",
            "EXTERNALID",
            "N",
            "NAME",
            "ARTICLEID",
            "PUBMEDID",
            "PAGE",
            "TABLE",
            "UNIT {Melting Point}",
            "Decomposition",
            "UNIT {Decomposition}",
            "Glass-transition temperature (Tg)",
            "UNIT {Glass-transition temperature (Tg)}",
            "Melting Point {measured, converted}",
            "UNIT {Melting Point}.1",
            "Decomposition {measured, converted}",
            "UNIT {Decomposition}.1",
            "Glass-transition temperature (Tg) {measured, converted}",
            "UNIT {Glass-transition temperature (Tg)}.1",
            "weight loss",
            "UNIT {weight loss}",
            "Solvent",
        ]
    )
    return df


def _drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df["Melting Point"] = df["Melting Point"].replace("-", np.nan)
    df = df.dropna()
    return df


def _canonize_smiles(smiles: str) -> str:
    smiles = str(smiles)
    canon_smiles = Chem.CanonSmiles(smiles)
    return canon_smiles


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"SMILES": "smiles", "Melting Point": "MP"})
    return df


def _replace_with_mean(value: float) -> float:
    try:
        return float(value)
    except ValueError:
        parts = value.split(" ")
        num1 = float(parts[0])
        num2 = float(parts[2])
        value = (num1 + num2) / 2
        return value
    return value


def _drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["Unnamed: 0", "error"])
    return df


# Nodes
def preprocess_main_database(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_columns_main(df)
    df = _drop_missing_values(df)
    df["Melting Point"] = df["Melting Point"].apply(_replace_with_mean)
    df["SMILES"] = df["SMILES"].apply(_canonize_smiles)
    df = _rename_columns(df)
    return df


def preprocess_ilt_database(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_columns(df)
    df["MP"].astype(float)
    df["smiles"] = df["smiles"].apply(_canonize_smiles)
    df = df[["smiles", "MP"]]
    return df


def merge_data(main: pd.DataFrame, ilt: pd.DataFrame) -> pd.DataFrame:
    frames = [main, ilt]
    merged_df = pd.concat(frames)
    merged_df = merged_df.drop_duplicates(subset="smiles")
    merged_df = merged_df.where(merged_df["MP"] < 250).dropna()  # noqa: PLR2004
    merged_df = merged_df.where(merged_df["MP"] > 0).dropna()  # noqa: PLR2004
    return merged_df


def random_data_split(df: pd.DataFrame, split_ratio: float) -> t.Tuple:
    df_train, df_test = train_test_split(df, test_size=split_ratio, random_state=42)
    return df_train, df_test

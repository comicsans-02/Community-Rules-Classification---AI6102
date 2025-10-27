import re
import pandas as pd
from typing import List, Optional
from sklearn.model_selection import train_test_split
from cleantext import clean

def clean_text(text: Optional[str]) -> str:
    """
    Basic text cleaning function to turn text to lowercase, remove URLs, remove special characters and normalize whitespaces
    """
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text) # URLs
    text = re.sub(r"[^a-z0-9\s']", '', text)              
    text = re.sub(r"\s+", ' ', text).strip()
    return text

def cleaner(text: Optional[str]) -> str:
    """
    Uses the package clean-text to customise and clean input as per parameters
    """
    return clean(
        text,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
        no_line_breaks=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        lang="en",
    )

def preprocess_dataframe(df: pd.DataFrame, text_columns: List[str], label_column: Optional[str] = None) -> pd.DataFrame:
    """
    Clean multiple text columns and create a combined text column for modeling
    """
    # Clean only columns that exist
    for col in [c for c in text_columns if c in df.columns]:
        df[col] = df[col].astype(str).apply(cleaner)

    # Drop rows with missing `rule_violation` value
    if label_column and label_column in df.columns:
        df = df.dropna(subset=[label_column])

    return df

def combine_comment_rule(df: pd.DataFrame, comment_col: str = 'comment_text', rule_col: str = 'rule_text', new_col: str = 'combined_text') -> pd.DataFrame:
    """
    Combine comment text and rule text into a single column for modeling.
    """
    if comment_col not in df.columns or rule_col not in df.columns:
        raise ValueError(f"Columns {comment_col} or {rule_col} not found in DataFrame.")
    
    # Combine with separator for clarity
    df[new_col] = df[comment_col].astype(str) + " [SEP] " + df[rule_col].astype(str)
    
    return df


def split_data(df: pd.DataFrame, label_column: str, test_size: float = 0.2, random_state: int = 42):
    """
    Split into train and validation sets.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        df['combined_text'], 
        df[label_column],
        test_size=test_size,
        stratify=df[label_column],
        random_state=random_state
    )

    return X_train, X_val, y_train, y_val

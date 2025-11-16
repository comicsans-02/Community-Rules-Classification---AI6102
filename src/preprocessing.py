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

def write_positive_negative_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert binary rule violation labels (0/1) into FastText-compatible string labels ('positive'/'negative')
    """
    df['label'] = df['rule_violation'].apply(lambda x: 'positive' if x == 1 else 'negative')
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

# Augment training data with positive/negative examples as additional rows
# This follows the same pattern as the ridge regression notebook

def get_augmented_dataframe(train_df_clean, train_df_original, test_df_original):
    """
    Extract positive/negative examples from train and test sets,
    create new rows, remove example columns, and return augmented dataframe.
    """
    # Start with the main body column (remove example columns from train_df_clean)
    core_columns = ['body', 'rule', 'subreddit', 'rule_violation']
    flatten = []
    flatten.append(train_df_clean[core_columns].copy())
    
    # Process train dataset examples
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col_name = f"{violation_type}_example_{i}"
            
            if col_name in train_df_original.columns:
                sub_dataset = train_df_original[[col_name, "rule", "subreddit"]].copy()
                sub_dataset = sub_dataset.rename(columns={col_name: "body"})
                sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
                
                # Remove null/empty entries
                sub_dataset.dropna(subset=['body'], inplace=True)
                sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]
                
                if not sub_dataset.empty:
                    # Preprocess this sub-dataset
                    columns_to_clean = ["body", "rule"]
                    sub_dataset_clean = preprocess_dataframe(sub_dataset, columns_to_clean, label_column="rule_violation")
                    flatten.append(sub_dataset_clean[core_columns])
    
    # Process test dataset examples  
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col_name = f"{violation_type}_example_{i}"
            
            if col_name in test_df_original.columns:
                sub_dataset = test_df_original[[col_name, "rule", "subreddit"]].copy()
                sub_dataset = sub_dataset.rename(columns={col_name: "body"})
                sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
                
                # Remove null/empty entries
                sub_dataset.dropna(subset=['body'], inplace=True)
                sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]
                
                if not sub_dataset.empty:
                    # Preprocess this sub-dataset
                    columns_to_clean = ["body", "rule"]
                    sub_dataset_clean = preprocess_dataframe(sub_dataset, columns_to_clean, label_column="rule_violation")
                    flatten.append(sub_dataset_clean[core_columns])
    
    # Concatenate all dataframes
    dataframe = pd.concat(flatten, axis=0)
    dataframe = dataframe.drop_duplicates(subset=['body', 'rule', 'subreddit'], ignore_index=True)
    dataframe.drop_duplicates(subset=['body', 'rule'], keep='first', inplace=True)
    
    # Shuffle
    return dataframe.sample(frac=1, random_state=42).reset_index(drop=True)

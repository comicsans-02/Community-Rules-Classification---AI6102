import re
import pandas as pd
from typing import List, Optional
from sklearn.model_selection import train_test_split
from cleantext import clean

PLACEHOLDER_TOKENS = {"nan", "<url>", "<email>", "<phone>", "<NUM>", "<NUMBER>", "<PHONE>"}

# def clean_text(text: Optional[str]) -> str:
#     """
#     Basic text cleaning function to turn text to lowercase, remove URLs, remove special characters and normalize whitespaces
#     """
#     if pd.isnull(text):
#         return ""
#     text = text.lower()
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text) # URLs
#     text = re.sub(r"[^a-z0-9\s']", '', text)              
#     text = re.sub(r"\s+", ' ', text).strip()
#     return text

# def get_augmented_dataframe(train_df_clean, train_df_original, test_df_original):
#     """
#     Extract positive/negative examples from train and test sets,
#     create new rows, remove example columns, and return augmented dataframe.
#     """
#     # Start with the main body column (remove example columns from train_df_clean)
#     core_columns = ['body', 'rule', 'subreddit', 'rule_violation']
#     flatten = []
#     flatten.append(train_df_clean[core_columns].copy())
    
#     # Process train dataset examples
#     for violation_type in ["positive", "negative"]:
#         for i in range(1, 3):
#             col_name = f"{violation_type}_example_{i}"
            
#             if col_name in train_df_original.columns:
#                 sub_dataset = train_df_original[[col_name, "rule", "subreddit"]].copy()
#                 sub_dataset = sub_dataset.rename(columns={col_name: "body"})
#                 sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
                
#                 # Remove null/empty entries
#                 sub_dataset.dropna(subset=['body'], inplace=True)
#                 sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]
                
#                 if not sub_dataset.empty:
#                     # Preprocess this sub-dataset
#                     columns_to_clean = ["body", "rule"]
#                     sub_dataset_clean = preprocess_dataframe(sub_dataset, columns_to_clean, label_column="rule_violation")
#                     flatten.append(sub_dataset_clean[core_columns])
    
#     # Process test dataset examples  
#     for violation_type in ["positive", "negative"]:
#         for i in range(1, 3):
#             col_name = f"{violation_type}_example_{i}"
            
#             if col_name in test_df_original.columns:
#                 sub_dataset = test_df_original[[col_name, "rule", "subreddit"]].copy()
#                 sub_dataset = sub_dataset.rename(columns={col_name: "body"})
#                 sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
                
#                 # Remove null/empty entries
#                 sub_dataset.dropna(subset=['body'], inplace=True)
#                 sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]
                
#                 if not sub_dataset.empty:
#                     # Preprocess this sub-dataset
#                     columns_to_clean = ["body", "rule"]
#                     sub_dataset_clean = preprocess_dataframe(sub_dataset, columns_to_clean, label_column="rule_violation")
#                     flatten.append(sub_dataset_clean[core_columns])
    
#     # Concatenate all dataframes
#     dataframe = pd.concat(flatten, axis=0)
#     dataframe = dataframe.drop_duplicates(subset=['body', 'rule', 'subreddit'], ignore_index=True)
#     dataframe.drop_duplicates(subset=['body', 'rule'], keep='first', inplace=True)
    
#     # Shuffle
#     return dataframe.sample(frac=1, random_state=42).reset_index(drop=True)


# This follows the same pattern as the ridge regression notebook
def get_augmented_dataframe(train_df_original: pd.DataFrame, test_df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Keep duplicates unless entire row (including source) is identical
    Ensures 'body' and 'rule' columns exist and are strings (may be empty)
    """
    core_cols = ["body", "rule", "subreddit", "rule_violation", "source"]
    all_rows = []

    # Ensure columns exist in originals
    for df in [train_df_original, test_df_original]:
        if "rule" not in df.columns:
            df["rule"] = ""
        if "subreddit" not in df.columns:
            df["subreddit"] = ""

    # Add original train rows (as-is)
    df_main = train_df_original.copy()
    df_main["source"] = "original"
    # ensure types
    df_main["body"] = df_main["body"].astype(str).fillna("").astype(str)
    df_main["rule"] = df_main["rule"].astype(str).fillna("").astype(str)
    df_main["subreddit"] = df_main["subreddit"].astype(str).fillna("").astype(str)
    df_main["rule_violation"] = df_main["rule_violation"].astype(int)
    all_rows.append(df_main[core_cols])

    # Helper to extract examples from a dataframe
    def extract_examples(source_df, source_label):
        rows = []
        for violation_type in ["positive", "negative"]:
            for i in range(1, 3):
                col_name = f"{violation_type}_example_{i}"
                if col_name not in source_df.columns:
                    continue
                sub = source_df[[col_name, "rule", "subreddit"]].copy()
                sub = sub.rename(columns={col_name: "body"})
                # coerce types and fill missing bodies/rules with empty string
                sub["body"] = sub["body"].astype(str).fillna("").astype(str)
                sub["rule"] = sub["rule"].astype(str).fillna("").astype(str)
                sub["subreddit"] = sub["subreddit"].astype(str).fillna("").astype(str)
                sub["rule_violation"] = 1 if violation_type == "positive" else 0
                sub["source"] = f"{source_label}_aug_{violation_type}"
                # drop rows where body is empty (no useful augmentation)
                sub = sub[sub["body"].str.strip().str.len() > 0]
                if not sub.empty:
                    rows.append(sub[core_cols])
        return rows

    # Add examples from train and test
    all_rows.extend(extract_examples(train_df_original, "train"))
    all_rows.extend(extract_examples(test_df_original, "test"))

    # Concatenate
    df = pd.concat(all_rows, ignore_index=True)

    # Drop only exact duplicates where every field matches including source
    df = df.drop_duplicates(subset=core_cols).reset_index(drop=True)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def combine_comment_rule(df: pd.DataFrame, comment_col: str = "body", rule_col: str = "rule", new_col: str = "combined_text") -> pd.DataFrame:
    """
    Safely combine comment and rule into combined_text
    Remove rows that produce meaningless combined_text (empty or only placeholders)
    """
    if comment_col not in df.columns or rule_col not in df.columns:
        raise ValueError(f"Columns {comment_col} or {rule_col} not found in DataFrame.")

    df = df.copy()

    # Ensure string and strip
    df[comment_col] = df[comment_col].astype(str).fillna("").apply(lambda s: s.strip())
    df[rule_col] = df[rule_col].astype(str).fillna("").apply(lambda s: s.strip())

    # Build combined string
    df[new_col] = (df[comment_col].fillna("") + " [SEP] " + df[rule_col].fillna("")).str.strip()

    # Define function to detect "meaningful" combined_text
    def is_meaningful(s: str) -> bool:
        if not isinstance(s, str):
            return False
        s_stripped = s.strip()
        if len(s_stripped) < 3:
            return False
        # remove the separator and check remaining content length
        s_no_sep = s_stripped.replace("[SEP]", "").replace("  ", " ").strip()
        if len(s_no_sep) < 2:
            return False
        # exclude strings that are only placeholder tokens
        s_low = s_no_sep.lower()
        if s_low in {"", "nan", "<url>", "<email>", "<phone>"}:
            return False
        # if string only contains punctuation or bracket tokens, reject
        if re.fullmatch(r"[\W_]+", s_no_sep):
            return False
        return True

    meaningful_mask = df[new_col].apply(is_meaningful)
    df = df[meaningful_mask].reset_index(drop=True)

    return df

def cleaner(text: Optional[str]) -> str:
    """
    Clean input text robustly:
    - If input is None/NaN-like -> return empty string
    - Use clean-text but map obvious placeholder tokens to empty
    - Normalize whitespace and strip
    """
    if text is None:
        return ""

    # Sometimes pandas NaN cast to str gives 'nan' -> treat as missing
    if isinstance(text, float) and pd.isna(text):
        return ""
    t = str(text).strip()

    if t.lower() in {"nan", "none", "null", ""}:
        return ""

    # Apply clean-text and be conservative: return empty if result contains only placeholder tokens
    cleaned = clean(
        t,
        fix_unicode=True,
        to_ascii=True,
        lower=False, # keep upper/lower case as is
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

    if cleaned is None:
        return ""

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Map placeholder returns to empty
    cleaned_lower = cleaned.lower()
    if any(tok in cleaned_lower for tok in ["<url>", "<email>", "<phone>"]):
        # if the whole text is only a placeholder token, treat as empty
        if re.fullmatch(r"(<url>|<email>|<phone>)+", cleaned_lower):
            return ""

    # If cleaned equals literal 'nan' or similar, return empty
    if cleaned_lower in {"nan", "none", "null", ""}:
        return ""

    return cleaned

def preprocess_dataframe(df: pd.DataFrame, text_columns: List[str], label_column: Optional[str] = None) -> pd.DataFrame:
    """
    Clean specified text columns
    Remove rows that have no useful text in any of the specified text_columns after cleaning
    """
    df = df.copy()

    cleaned_cols = []
    for col in text_columns:
        if col in df.columns:
            # apply cleaner (which returns "" for missing/nan-like)
            df[col] = df[col].apply(cleaner)
            cleaned_cols.append(col)

    # drop rows with missing label if requested
    if label_column and label_column in df.columns:
        df = df.dropna(subset=[label_column])
        # ensure integer labels 0/1
        try:
            df[label_column] = df[label_column].astype(int)
        except Exception:
            pass

    # Remove rows where ALL cleaned_cols are empty after cleaning
    if cleaned_cols:
        mask_any = df[cleaned_cols].apply(lambda r: any(str(v).strip() != "" for v in r), axis=1)
        df = df[mask_any].reset_index(drop=True)

    return df

def split_data(df, label_column='rule_violation', test_size=0.2):
    """
    Split into train and validation dataset
    """
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df[label_column], random_state=42
    )
    return train_df, val_df

# For baseline Fasttext model
def write_positive_negative_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert binary rule violation labels (0/1) into FastText-compatible string labels ('positive'/'negative')
    """
    df['label'] = df['rule_violation'].apply(lambda x: '1' if x == 1 else '0')
    return df

def map_fasttext_labels(df, label_col="rule_violation"):
    df[label_col] = df[label_col].map({
        "positive": 1,
        "negative": 0,
        1: 1,
        0: 0
    })
    return df

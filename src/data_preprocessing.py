import pandas as pd
import re 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.utils.logger import setup_logger

# Download necessary NTLK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Initialize logger
logger = setup_logger("Data Preprocessing", "logs/preprocessing.log")

def clean_text(text):
    """
    Comprehensive text cleaning:
    - Convert to lowercase
    - Remove special characters
    - Tokenize
    - Remove stopwords
    """

    # Explicit multi-stage checking
    if pd.isna(text):  # First, check if truly NaN
        return ""
    if not isinstance(text, str):  # Second, check type
        return str(text)
    
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)

    # Remove stopwords. Keep question words as they can inform the question being asked.
    stop_words = set(stopwords.words('english')) - {"what", "why", "how", "who", "when", "where"}
    tokens = [word for word in tokens if word not in stop_words]
    
    # Rejoin tokens
    cleaned_text = ' '.join(tokens)

    return cleaned_text.strip()

def inspect_dataset(df: pd.DataFrame):
    """
    Inspect the dataset for basic information, missing values, and types.
    """
    logger.info(f"Dataset Shape: {df.shape}")
    logger.info(f"Missing Values:\n{df.isnull().sum()}")
    logger.info(f"Data Types:\n{df.dtypes}")

    nan_rows = df[df.isnull().any(axis=1)]
    logger.info(f"Sample Rows with NaN:\n{nan_rows}")


def standardize_column_type(df, column_name):
    """
    Standardize column values to guaranteed string type.
    """
    df[column_name] = df[column_name].fillna('').astype(str).str.strip()
    return df


def preprocess_dataset(input_path: str, output_path: str, sample_size=50000):
    """
    Preprocess the Quora Question Pairs dataset:
    - Load dataset
    - Clean and normalize text
    - Save processed dataset
    """
    logger.info("Starting dataset preprocessing...")

    logger.info("Loading dataset...")
    df = pd.read_csv(input_path)

    logger.info("Inspecting raw dataset...")
    inspect_dataset(df)

    logger.info("Cleaning dataset...")
    df.dropna(subset=["question1", "question2"], inplace=True)
    df.drop_duplicates(subset=["question1", "question2"], inplace=True)
    logger.info(f"Dataset Shape after cleaning: {df.shape}")

    logger.info("Sampling dataset...")
    sampled_df = df.sample(n=min(sample_size, len(df)), random_state=42)

    logger.info("Applying text preprocessing...")
    sampled_df["cleaned_q1"] = sampled_df["question1"].apply(clean_text)
    sampled_df["cleaned_q2"] = sampled_df["question2"].apply(clean_text)

    logger.info("Standardising column types...")
    sampled_df = standardize_column_type(sampled_df, 'cleaned_q1')
    sampled_df = standardize_column_type(sampled_df, 'cleaned_q2')
    
    logger.info("Inspecting processed dataset...")
    inspect_dataset(sampled_df)

    logger.info(f"Saving processed dataset to {output_path}...")
    sampled_df.to_pickle(output_path)
    logger.info("Dataset preprocessing complete.")

    return sampled_df



if __name__ == "__main__":

    preprocess_dataset(
        "/home/isgr/text-similarity/data/raw/train.csv",
        "/home/isgr/text-similarity/data/processed/train.pkl",
        sample_size=1000
        )
    
    # preprocess_dataset(
    #     "/home/isgr/text-similarity/data/raw/test.csv",
    #     "/home/isgr/text-similarity/data/processed/test.pkl",
    #     sample_size=1000
    #     )
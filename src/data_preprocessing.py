import pandas as pd
import re 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger("Data Preprocessing", "logs/preprocessing.log")

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def clean_text(text):
    """
    Comprehensive text cleaning:
    - Convert to lowercase
    - Remove special characters
    - Tokenize
    - Remove stopwords (keeping question words)
    """
    # Handle non-string input
    if pd.isna(text):
        return ""
    if not isinstance(text, str):
        return str(text)
    
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords but keep question words
    stop_words = set(stopwords.words('english')) - {"what", "why", "how", "who", "when", "where"}
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens).strip()

def inspect_dataset(df):
    """
    Inspect the dataset and log key information.
    """
    logger.info(f"Dataset Shape: {df.shape}")
    logger.info(f"Missing Values:\n{df.isnull().sum()}")
    logger.info(f"Data Types:\n{df.dtypes}")

    # Log sample of rows with NaN if any exist
    nan_rows = df[df.isnull().any(axis=1)]
    if not nan_rows.empty:
        logger.info(f"Sample Rows with NaN:\n{nan_rows.head()}")

def standardize_column(df, column_name):
    """
    Clean up a column's values and ensure string type.
    """
    return df[column_name].fillna('').astype(str).str.strip()

def preprocess_dataset(input_path, output_path, sample_size=50000):
    """
    Preprocess the Quora Question Pairs dataset:
    1. Load and inspect data
    2. Clean and normalize text
    3. Sample if requested
    4. Save processed dataset
    """
    logger.info("Starting dataset preprocessing...")

    # Load dataset
    logger.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)
    inspect_dataset(df)

    # Clean dataset
    logger.info("Removing missing values and duplicates...")
    df.dropna(subset=["question1", "question2"], inplace=True)
    df.drop_duplicates(subset=["question1", "question2"], inplace=True)
    logger.info(f"Dataset shape after cleaning: {df.shape}")

    # Sample if requested
    if sample_size:
        logger.info(f"Sampling {sample_size} rows...")
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    # Clean text
    logger.info("Cleaning question text...")
    df["cleaned_q1"] = df["question1"].apply(clean_text)
    df["cleaned_q2"] = df["question2"].apply(clean_text)

    # Standardize columns
    logger.info("Standardizing column types...")
    df["cleaned_q1"] = standardize_column(df, "cleaned_q1")
    df["cleaned_q2"] = standardize_column(df, "cleaned_q2")

    # Final inspection
    logger.info("Final dataset inspection:")
    inspect_dataset(df)

    # Save processed dataset
    logger.info(f"Saving processed dataset to {output_path}")
    df.to_pickle(output_path)
    logger.info("Dataset preprocessing complete.")

    return df

if __name__ == "__main__":
    # Process training data
    preprocess_dataset(
        input_path="data/raw/train.csv",
        output_path="data/processed/train.pkl",
        sample_size=1000
    )
    
    # Uncomment to process test data
    # preprocess_dataset(
    #     input_path="data/raw/test.csv",
    #     output_path="data/processed/test.pkl",
    #     sample_size=1000
    # )
import pandas as pd
import re 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.utils.logger import setup_logger
from src.utils.config import load_config

config =load_config()

# Initialize logger
logger = setup_logger("Data Preprocessing", config['logging']['file']['preprocessing'])

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def clean_text(text):
    """
    Clean text based on configuration settings.
    """
    text_config = config['preprocessing']['text_cleaning']

    # Handle non-string input
    if pd.isna(text):
        return ""
    if not isinstance(text, str):
        return str(text)
    
    # Apply configured transformations
    if text_config['convert_lowercase']:
        text = text.lower()

    if text_config['remove_punctuation']:
        text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)

    if text_config['remove_stopwords']:
        # Get question words from config
        question_words = set(text_config['question_words']) if text_config['keep_question_words'] else set()
        stop_words = set(stopwords.words('english')) - question_words
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

def preprocess_dataset(
    input_path=None, 
    output_path=None, 
    sample_size=None
):
    """
    Preprocess the dataset using configuration settings.
    """
    # Use config values if not specified
    input_path = input_path or config['paths']['raw']['train']
    output_path = output_path or config['paths']['processed']['train']
    sample_size = sample_size or config['preprocessing']['sample_size']

    logger.info(f"Starting dataset preprocessing from {input_path}")

    # Load dataset
    df = pd.read_csv(input_path)
    inspect_dataset(df)

    # Get column names from config
    input_cols = config['columns']['input']
    processed_cols = config['columns']['processed']

    # Clean dataset
    logger.info("Removing missing values and duplicates...")
    df.dropna(subset=[input_cols['first_question'], input_cols['second_question']], inplace=True)
    df.drop_duplicates(subset=[input_cols['first_question'], input_cols['second_question']], inplace=True)
    logger.info(f"Dataset shape after cleaning: {df.shape}")

    # Sample if specified
    if sample_size:
        logger.info(f"Sampling {sample_size} rows...")
        df = df.sample(
            n=min(sample_size, len(df)), 
            random_state=config['preprocessing']['random_seed']
        )

    # Clean text
    logger.info("Cleaning question text...")
    df[processed_cols['first_question']] = df[input_cols['first_question']].apply(clean_text)
    df[processed_cols['second_question']] = df[input_cols['second_question']].apply(clean_text)

    # Standardize columns
    logger.info("Standardizing column types...")
    df[processed_cols['first_question']] = standardize_column(df, processed_cols['first_question'])
    df[processed_cols['second_question']] = standardize_column(df, processed_cols['second_question'])

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
    preprocess_dataset()
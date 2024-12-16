import pandas as pd
from sentence_transformers import SentenceTransformer
from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger("Embeddings", "logs/embeddings.log")


def generate_embeddings(processed_df_path: str, embedding_path: str, column1:str="cleaned_q1", column2:str="cleaned_q2", model_name: str = "all-MiniLM-L6-v2") -> pd.DataFrame:
    """
    Generate embeddings for preprocessed questions
    
    Args:
    - processed_df_path: Path to processed dataset
    - embedding_path: Path to save the embeddings
    - model_name: Sentence Transformer model to use
    
    Returns:
    - DataFrame with embeddings and labels
    """
    
    logger.info(f"Loading processed dataset from {processed_df_path}...")
    # Load the processed dataset
    df = pd.read_pickle(processed_df_path)

    # Check and log data types in columns
    logger.info(f"Verifying data types of {column1} and {column2} columns...")
    logger.debug(f"Data types in cleaned_q1: {df[column1].dtype}")
    logger.debug(f"Data types in cleaned_q2: {df[column2].dtype}")
    logger.debug(f"Types in Columns:\n{df.dtypes}")
    
    # Check for missing values
    logger.info("Checking for missing values in the dataset...")
    nan_rows = df[df.isnull().any(axis=1)]
    if not nan_rows.empty:
        logger.warning(f"Found NaN values in the dataset: \n{nan_rows}")
    
    logger.info("Initializing the Sentence Transformer model...")
    model = SentenceTransformer(model_name)

    logger.info(f"Generating embeddings for {column1} and {column2}...")
    q1_embeddings = model.encode(df[column1].tolist())
    q2_embeddings = model.encode(df[column2].tolist())

    logger.debug(f"Generated embeddings for question 1. Example: {q1_embeddings[:1]}")
    logger.debug(f"Generated embeddings for question 2. Example: {q2_embeddings[:1]}")

    logger.info("Creating DataFrame with embeddings and labels...")
    embedding_df = pd.DataFrame({
        "question1": df["question1"],
        "question2": df["question2"],
        "q1_embeddings": list(q1_embeddings),
        "q2_embeddings": list(q2_embeddings),   
    })

    logger.info(f"Sample of the generated embedding DataFrame:\n{embedding_df.head(5)}")

    logger.info(f"Saving embeddings to {embedding_path}...")
    embedding_df.to_pickle(embedding_path)

    logger.info(f"Embeddings saved successfully to {embedding_path}")
    return embedding_df


if __name__ == "__main__":
    logger.info("Starting the embedding generation process...")
    
    # Adjust paths to your actual file locations
    embeddings = generate_embeddings(
        "/home/isgr/text-similarity/data/processed/train.pkl",
        "/home/isgr/text-similarity/data/embeddings/train_embeddings.pkl"
    )
    
    # Example for generating test embeddings (uncomment if needed)
    # embeddings = generate_embeddings(
    #     "/home/isgr/text-similarity/data/processed/test.pkl",
    #     "/home/isgr/text-similarity/data/embeddings/test_embeddings.pkl"
    # )

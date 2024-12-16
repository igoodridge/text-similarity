import pandas as pd
from sentence_transformers import SentenceTransformer
from src.utils.logger import setup_logger
from src.utils.config import load_config

# Load configuration
config = load_config()

# Initialize logger using config
logger = setup_logger("Embeddings", config['logging']['file']['embeddings'])

def generate_embeddings(
    processed_df_path=None,
    embedding_path=None,
    model_name=None,
    column1=None,
    column2=None
) -> pd.DataFrame:
    """
    Generate embeddings for preprocessed questions using configured settings.
    
    Args:
    - processed_df_path: Path to processed dataset (optional, uses config if not provided)
    - embedding_path: Path to save embeddings (optional, uses config if not provided)
    - model_name: Sentence Transformer model to use (optional, uses config if not provided)
    - column1: Name of first question column (optional, uses config if not provided)
    - column2: Name of second question column (optional, uses config if not provided)
    
    Returns:
    - DataFrame with embeddings and labels
    """
    # Use config values if not provided
    processed_df_path = processed_df_path or config['paths']['processed']['train']
    embedding_path = embedding_path or config['paths']['embeddings']['train']
    model_name = model_name or config['model']['name']
    
    # Get column names from config
    processed_cols = config['columns']['processed']
    embedding_cols = config['columns']['embeddings']
    column1 = column1 or processed_cols['first_question']
    column2 = column2 or processed_cols['second_question']
    
    logger.info(f"Loading processed dataset from {processed_df_path}")
    df = pd.read_pickle(processed_df_path)

    # Verify data types and log information
    logger.info(f"Verifying data types of {column1} and {column2} columns")
    logger.debug(f"Data types in {column1}: {df[column1].dtype}")
    logger.debug(f"Data types in {column2}: {df[column2].dtype}")
    logger.debug(f"Types in Columns:\n{df.dtypes}")
    
    # Check for missing values
    logger.info("Checking for missing values in the dataset")
    nan_rows = df[df.isnull().any(axis=1)]
    if not nan_rows.empty:
        logger.warning(f"Found {len(nan_rows)} rows with NaN values")
        logger.debug(f"Sample of rows with NaN:\n{nan_rows.head()}")
    
    # Initialize model with configured settings
    logger.info(f"Initializing Sentence Transformer model: {model_name}")
    model = SentenceTransformer(
        model_name,
    )

    # Generate embeddings with batch processing
    logger.info(f"Generating embeddings for {column1} and {column2}")
    batch_size = config['model'].get('batch_size', 32)
    
    q1_embeddings = model.encode(
        df[column1].tolist(),
        batch_size=batch_size,
        show_progress_bar=True
    )
    q2_embeddings = model.encode(
        df[column2].tolist(),
        batch_size=batch_size,
        show_progress_bar=True
    )

    logger.debug(f"Generated embeddings shape: {q1_embeddings.shape}")

    # Create DataFrame with original questions and embeddings
    logger.info("Creating DataFrame with embeddings")
    input_cols = config['columns']['input']
    embedding_df = pd.DataFrame({
        input_cols['first_question']: df[input_cols['first_question']],
        input_cols['second_question']: df[input_cols['second_question']],
        embedding_cols['first_question']: list(q1_embeddings),
        embedding_cols['second_question']: list(q2_embeddings)
    })

    # Log sample information
    logger.info(f"Generated embedding DataFrame shape: {embedding_df.shape}")
    logger.debug(f"Sample of embedding values:\n{embedding_df[embedding_cols['first_question']].iloc[0][:5]}")

    # Save embeddings
    logger.info(f"Saving embeddings to {embedding_path}")
    embedding_df.to_pickle(embedding_path)
    logger.info("Embeddings saved successfully")

    return embedding_df

if __name__ == "__main__":
    logger.info("Starting the embedding generation process")
    
    # Generate training embeddings
    embeddings = generate_embeddings()
    
    # Optionally generate test embeddings
    # test_embeddings = generate_embeddings(
    #     processed_df_path=config['paths']['processed']['test'],
    #     embedding_path=config['paths']['embeddings']['test']
    # )
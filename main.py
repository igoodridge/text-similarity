from pathlib import Path
import pandas as pd
from src.data_preprocessing import preprocess_dataset
from src.embeddings import generate_embeddings
from src.similarity_search import SimilarityEngine, QuestionSimilarityComparator
from src.utils.logger import setup_logger
from src.utils.config import load_config

# Load configuration
config = load_config()
logger = setup_logger("Main Runner", config['logging']['file']['app'])

def run_pipeline():
    """Run the complete data processing and model initialization pipeline."""
    try:
        # Step 1: Preprocess Dataset
        logger.info("Starting dataset preprocessing...")
        preprocess_dataset(
            input_path=config['paths']['raw']['train'],
            output_path=config['paths']['processed']['train'],
            sample_size=config['preprocessing']['sample_size']
        )
        logger.info("Dataset preprocessing completed.")

        # Step 2: Generate Embeddings
        logger.info("Starting embeddings generation...")
        generate_embeddings(
            processed_df_path=config['paths']['processed']['train'],
            embedding_path=config['paths']['embeddings']['train'],
            model_name=config['model']['name']
        )
        logger.info("Embeddings generation completed.")

        # Step 3: Initialize Similarity Search Components
        logger.info("Testing similarity search functionality...")
        # Load the data and initialize components
        df = pd.read_pickle(config['paths']['processed']['train'])
        embeddings_df = pd.read_pickle(config['paths']['embeddings']['train'])
        
        from src.similarity_search import load_embeddings_from_dataframe
        embeddings, questions = load_embeddings_from_dataframe(
            embeddings_df
        )
        
        # Test similarity engine
        engine = SimilarityEngine(embeddings, questions)
        test_query = "What is machine learning?"
        results = engine.find_similar_questions(
            test_query,
            top_k=config['similarity']['default_top_k'],
            method=config['similarity']['default_method']
        )
        logger.info(f"Sample search results for '{test_query}': {results}")

    except Exception as e:
        logger.error(f"An error occurred during the pipeline: {e}")
        raise

if __name__ == "__main__":
    run_pipeline()
from src.data_preprocessing import preprocess_dataset
from src.embeddings import generate_embeddings
from src.similarity_search import SimilaritySearcher
from src.utils.logger import setup_logger

logger = setup_logger("Main Runner", "logs/main.log")

def run_pipeline():
    try:
        # Step 1: Preprocess Dataset
        logger.info("Starting dataset preprocessing...")
        preprocess_dataset(
            input_path="data/raw/train.csv",
            output_path="data/processed/train.pkl",
            sample_size=1000
        )
        logger.info("Dataset preprocessing completed.")

        # Step 2: Generate Embeddings
        logger.info("Starting embeddings generation...")
        generate_embeddings(
            processed_df_path="data/processed/train.pkl",
            embedding_path="data/embeddings/train_embeddings.pkl",
            model_name="all-MiniLM-L6-v2"
        )
        logger.info("Embeddings generation completed.")

        # Step 3: Initialize Similarity Searcher and Evaluate
        logger.info("Initializing similarity searcher and running evaluation...")
        searcher = SimilaritySearcher(
            questions_path="data/processed/train.pkl",
            embeddings_path="data/embeddings/train_embeddings.pkl",
            column1="q1_embeddings",
            column2="q2_embeddings"
        )
        question_similarity = searcher.compute_question_similarity(threshold=0.7)
        metrics = searcher.evaluate_similarity()
        logger.info(f"Similarity model evaluation metrics: {metrics}")

    except Exception as e:
        logger.error(f"An error occurred during the pipeline: {e}")

if __name__ == "__main__":
    run_pipeline()

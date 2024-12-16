from src.utils.logger import setup_logger
import argparse
from src.data_preprocessing import preprocess_data
from src.embeddings import generate_embeddings
from src.similarity_search import SimilaritySearcher
import logging
import os

def main():
    # Set up logger
    os.makedirs("logs", exist_ok=True)
    logger = setup_logger(log_file="logs/app.log", log_level=logging.INFO)

    logger.info("Starting the Text Similarity System")

    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description="Text Similarity Search System")
        parser.add_argument("--data", required=True, help="Path to input dataset")
        parser.add_argument("--output", required=True, help="Path to save results")
        args = parser.parse_args()

        # Workflow
        logger.info("Preprocessing data...")
        preprocessed_data = preprocess_data(args.data)

        logger.info("Generating embeddings...")
        embedding_file = generate_embeddings(preprocessed_data, model_name="all-MiniLM-L6-v2")

        logger.info("Performing similarity search...")
        searcher = SimilaritySearcher(embedding_file)
        query = input("Enter your query: ")
        results = searcher.find_similar_questions(query, top_k=5)
        logger.info(f"Search Results: {results}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()

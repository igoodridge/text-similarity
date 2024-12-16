import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from sentence_transformers import SentenceTransformer
from src.utils.logger import setup_logger
from src.data_preprocessing import clean_text

# Initialize logger
logger = setup_logger("Similarity Search", "logs/sim_search.log")


class SimilaritySearcher:
    """
    A class to compute question similarities, evaluate performance, and find similar questions.

    Args:
        questions_path (str): Path to the pickled questions dataset.
        embeddings_path (str): Path to the pickled embeddings dataset.
        column1 (str): Name of the first embeddings column.
        column2 (str): Name of the second embeddings column.
    """

    def __init__(self, questions_path: str, embeddings_path: str, column1: str, column2: str):
        logger.info("Loading questions dataset and embeddings.")
        self.questions_df = pd.read_pickle(questions_path)
        self.embeddings_df = pd.read_pickle(embeddings_path)

        # Ensure embeddings match dataframe lengths
        assert len(self.questions_df) == len(self.embeddings_df), "Question data and embeddings length mismatch"

        self.column1 = column1
        self.column2 = column2

        # Compute similarities and store results in a DataFrame
        self.result_df = self.compute_question_similarity()

    def compute_question_similarity(self, threshold: float = 0.7) -> pd.DataFrame:
        """
        Compute similarity between two columns of embeddings and append results to the original DataFrame.

        Args:
            threshold (float): Similarity threshold for classification.

        Returns:
            pd.DataFrame: DataFrame with similarity scores and a duplicate flag.
        """
        logger.info("Computing similarity between embedding columns.")

        # Convert embeddings to numpy arrays
        embeddings1 = np.stack(self.embeddings_df[self.column1].values)
        embeddings2 = np.stack(self.embeddings_df[self.column2].values)

        # Validate input shapes
        assert embeddings1.shape == embeddings2.shape, "Embedding columns must have identical shape."

        # Compute cosine similarity for each pair
        similarities = np.diagonal(cosine_similarity(embeddings1, embeddings2))

        # Binary classification based on threshold
        binary_similarities = (similarities > threshold).astype(int)

        # Append similarity metrics to a copy of the original DataFrame
        result_df = self.questions_df.copy()
        result_df['similarity_score'] = similarities
        result_df['is_duplicate_modelled'] = binary_similarities

        logger.info("Similarity computation complete.")
        return result_df

    def evaluate_similarity(self) -> dict:
        """
        Evaluate similarity performance using precision, recall, F1-score, accuracy, and ROC-AUC.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        logger.info("Evaluating similarity performance.")

        df = self.result_df
        similarity_scores = df['similarity_score']
        given_labels = df["is_duplicate"].to_list()
        predicted_labels = df["is_duplicate_modelled"].to_list()

        # Calculate evaluation metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            given_labels, predicted_labels, average="binary"
        )
        accuracy = accuracy_score(given_labels, predicted_labels)
        roc_auc = roc_auc_score(given_labels, similarity_scores)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
        }

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def find_similar_questions(self, query_text: str, model_name: str = "all-MiniLM-L6-v2", top_k: int = 5) -> list:
        """
        Find the top-k similar questions to the query text based on cosine similarity.

        Args:
            query_text (str): The user's question to compare.
            model_name (str): The pre-trained model to use for embeddings.
            top_k (int): The number of similar questions to return.

        Returns:
            list: A list of dictionaries containing the top-k most similar questions and their similarity scores.
        """
        logger.info(f"Finding top {top_k} similar questions for query: '{query_text}'.")

        # Combine questions from both columns into a single list
        all_questions = pd.concat(
            [self.questions_df["question1"], self.questions_df["question2"]]
        ).reset_index(drop=True)

        # Combine embeddings into a single list
        all_embeddings = pd.concat(
            [self.embeddings_df[self.column1], self.embeddings_df[self.column2]]
        ).reset_index(drop=True)

        # Ensure embeddings are in the correct shape
        all_embeddings = np.stack(all_embeddings.values)
        assert len(all_questions) == len(all_embeddings), "Question list and embeddings list must be the same length."

        # Initialize the model and process the query
        model = SentenceTransformer(model_name)
        cleaned_query_text = clean_text(query_text)
        query_embedding = model.encode([cleaned_query_text]).reshape(1, -1)

        # Compute cosine similarities
        logger.info("Computing cosine similarities between query and all questions.")
        similarities = cosine_similarity(query_embedding, all_embeddings)[0]

        # Find the top-k most similar questions
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [
            {"question": all_questions.iloc[i], "similarity_score": similarities[i]}
            for i in top_k_indices
        ]

        logger.info("Top-k similar questions retrieved.")
        return results


def main():
    # Create a SimilaritySearcher instance
    searcher = SimilaritySearcher(
        questions_path="/home/isgr/text-similarity/data/processed/train.pkl",
        embeddings_path="/home/isgr/text-similarity/data/embeddings/train_embeddings.pkl",
        column1="q1_embeddings",
        column2="q2_embeddings",
    )

    # Evaluate performance
    metrics = searcher.evaluate_similarity()
    print("Evaluation Metrics:", metrics)

    # Example query
    query = "What is machine learning?"
    similar_questions = searcher.find_similar_questions(query_text=query, top_k=5)
    print("Top Similar Questions:", similar_questions)


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Union
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.utils.logger import setup_logger
from src.data_preprocessing import clean_text

class SimilarityEngine:
    """
    A modular similarity search engine supporting multiple methods.
    
    Responsibilities:
    - Manage embedding models
    - Perform similarity computations
    - Support different similarity search techniques
    """
    
    def __init__(self, 
                 embeddings: np.ndarray, 
                 questions: List[str], 
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the similarity engine with pre-computed embeddings.
        
        Args:
            embeddings (np.ndarray): Precomputed embeddings for questions
            questions (List[str]): Corresponding original questions
            model_name (str): Sentence transformer model to use
        """
        self.logger = setup_logger("SimilarityEngine", "logs/similarity_engine.log")
        self.model = SentenceTransformer(model_name)
        
        # Normalize embeddings for consistent similarity computation
        faiss.normalize_L2(embeddings)
        
        self.embeddings = embeddings
        self.questions = questions
        
        # Prepare FAISS index for efficient search
        self._prepare_faiss_index()
    
    def _prepare_faiss_index(self):
        """
        Prepare a FAISS index for efficient similarity search.
        """
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(self.embeddings)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query into an embedding vector.
        
        Args:
            query (str): Input query to encode
        
        Returns:
            np.ndarray: Encoded query embedding
        """
        cleaned_query = clean_text(query)
        query_embedding = self.model.encode([cleaned_query])
        faiss.normalize_L2(query_embedding)
        return query_embedding
    
    def find_similar_questions(
        self, 
        query: str, 
        top_k: int = 5, 
        method: str = 'faiss'
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Find similar questions using specified method.
        
        Args:
            query (str): Input query
            top_k (int): Number of similar questions to return
            method (str): Similarity search method ('cosine' or 'faiss')
        
        Returns:
            List of dictionaries with similar questions and their scores
        """
        query_embedding = self.encode_query(query)
        
        if method == 'cosine':
            return self._cosine_similarity_search(query_embedding, top_k)
        elif method == 'faiss':
            return self._faiss_similarity_search(query_embedding, top_k)
        else:
            raise ValueError(f"Unsupported similarity method: {method}")
    
    def _cosine_similarity_search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Perform cosine similarity search.
        
        Args:
            query_embedding (np.ndarray): Encoded query
            top_k (int): Number of similar questions to return
        
        Returns:
            List of similar questions with their scores
        """
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {
                "question": self.questions[idx], 
                "similarity_score": similarities[idx]
            } 
            for idx in top_indices
        ]
    
    def _faiss_similarity_search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Perform FAISS similarity search.
        
        Args:
            query_embedding (np.ndarray): Encoded query
            top_k (int): Number of similar questions to return
        
        Returns:
            List of similar questions with their scores
        """
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        return [
            {
                "question": self.questions[indices[0][i]], 
                "similarity_score": 1 / (1 + distances[0][i])  # Convert distance to similarity
            } 
            for i in range(top_k)
        ]

class QuestionSimilarityComparator:
    """
    Specialized class for comparing two specific questions.
    
    Responsibilities:
    - Compute similarity between two specific questions
    - Provide detailed similarity metrics
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the comparator with a sentence transformer model.
        
        Args:
            model_name (str): Sentence transformer model to use
        """
        self.logger = setup_logger("QuestionSimilarityComparator", "logs/comparator.log")
        self.model = SentenceTransformer(model_name)
    
    def compute_similarity(self, question1: str, question2: str) -> Dict[str, float]:
        """
        Compute similarity between two questions.
        
        Args:
            question1 (str): First question
            question2 (str): Second question
        
        Returns:
            Dictionary of similarity metrics
        """
        # Clean questions
        cleaned_q1 = clean_text(question1)
        cleaned_q2 = clean_text(question2)
        
        # Encode questions
        emb1 = self.model.encode([cleaned_q1])
        emb2 = self.model.encode([cleaned_q2])
        
        # Compute cosine similarity
        similarity_score = cosine_similarity(emb1, emb2)[0][0]
        
        return {
            "raw_similarity": similarity_score,
            "is_similar": similarity_score > 0.7,
            "normalized_score": min(max(similarity_score, 0), 1)
        }

def load_embeddings_from_dataframe(df: pd.DataFrame, column1: str, column2: str):
    """
    Utility function to load embeddings from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column1 (str): First embedding column
        column2 (str): Second embedding column
    
    Returns:
        Tuple of embeddings and questions
    """
    # Combine embeddings and questions
    all_embeddings = pd.concat([df[column1], df[column2]]).reset_index(drop=True)
    all_questions = pd.concat([df["question1"], df["question2"]]).reset_index(drop=True)
    
    return np.stack(all_embeddings.values), all_questions.tolist()

def main():
    # Example usage
    import pandas as pd
    
    # Load preprocessed data
    df = pd.read_pickle("data/processed/train.pkl")
    embeddings, questions = load_embeddings_from_dataframe(
        df, "q1_embeddings", "q2_embeddings"
    )
    
    # Initialize similarity engine
    engine = SimilarityEngine(embeddings, questions)
    
    # Find similar questions
    query = "What is machine learning?"
    similar_questions_faiss = engine.find_similar_questions(query, method='faiss')
    similar_questions_cosine = engine.find_similar_questions(query, method='cosine')
    
    # Compare two specific questions
    comparator = QuestionSimilarityComparator()
    similarity_metrics = comparator.compute_similarity(
        "What is machine learning?", 
        "Explain the concept of machine learning"
    )
    
    print("FAISS Similar Questions:", similar_questions_faiss)
    print("Cosine Similar Questions:", similar_questions_cosine)
    print("Similarity Metrics:", similarity_metrics)

if __name__ == "__main__":
    main()
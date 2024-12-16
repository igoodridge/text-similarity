import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Union
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.data_preprocessing import clean_text

# Load configuration
config = load_config()

class SimilarityEngine:
    """A modular similarity search engine supporting multiple methods."""
    
    def __init__(self, 
                 embeddings: np.ndarray, 
                 questions: List[str], 
                 model_name: str = None):
        """Initialize the similarity engine with pre-computed embeddings."""
        self.logger = setup_logger("SimilarityEngine", config['logging']['file']['similarity'])
        
        # Use model name from config if not provided
        self.model_name = model_name or config['model']['name']
        self.model = SentenceTransformer(self.model_name)
        
        # Normalize embeddings if specified in config
        if config['similarity']['faiss'].get('normalize_vectors', True):
            faiss.normalize_L2(embeddings)
        
        self.embeddings = embeddings
        self.questions = questions
        
        # Initialize similarity settings from config
        self.similarity_config = config['similarity']
        
        # Prepare FAISS index
        self._prepare_faiss_index()
    
    def _prepare_faiss_index(self):
        """Prepare a FAISS index for efficient similarity search."""
        dimension = self.embeddings.shape[1]
        index_type = self.similarity_config['faiss']['index_type']
        
        if index_type == 'IndexFlatL2':
            self.faiss_index = faiss.IndexFlatL2(dimension)
        else:
            self.logger.warning(f"Unsupported index type: {index_type}, using IndexFlatL2")
            self.faiss_index = faiss.IndexFlatL2(dimension)
            
        self.faiss_index.add(self.embeddings)
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query into an embedding vector."""
        cleaned_query = clean_text(query)
        query_embedding = self.model.encode(
            [cleaned_query],
            batch_size=config['model'].get('batch_size', 32)
        )
        
        if config['similarity']['faiss'].get('normalize_vectors', True):
            faiss.normalize_L2(query_embedding)
        return query_embedding
    
    def find_similar_questions(
        self, 
        query: str, 
        top_k: int = None, 
        method: str = None
    ) -> List[Dict[str, Union[str, float]]]:
        """Find similar questions using specified method."""
        # Use config defaults if not specified
        top_k = top_k or self.similarity_config['default_top_k']
        method = method or self.similarity_config['default_method']
        
        # Validate top_k
        max_top_k = self.similarity_config['max_top_k']
        if top_k > max_top_k:
            self.logger.warning(f"Requested top_k {top_k} exceeds maximum {max_top_k}")
            top_k = max_top_k
        
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
        """Perform cosine similarity search."""
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
        """Perform FAISS similarity search."""
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        return [
            {
                "question": self.questions[indices[0][i]], 
                "similarity_score": 1 / (1 + distances[0][i])
            } 
            for i in range(top_k)
        ]

class QuestionSimilarityComparator:
    """Specialized class for comparing two specific questions."""
    
    def __init__(self, model_name: str = None):
        """Initialize the comparator with a sentence transformer model."""
        self.logger = setup_logger("QuestionSimilarityComparator", config['logging']['file']['similarity'])
        self.model_name = model_name or config['model']['name']
        self.model = SentenceTransformer(self.model_name)
        self.similarity_threshold = config['similarity']['cosine']['threshold']
    
    def compute_similarity(self, question1: str, question2: str) -> Dict[str, float]:
        """Compute similarity between two questions."""
        # Clean questions
        cleaned_q1 = clean_text(question1)
        cleaned_q2 = clean_text(question2)
        
        # Encode questions using batch processing
        batch_size = config['model'].get('batch_size', 32)
        embeddings = self.model.encode(
            [cleaned_q1, cleaned_q2],
            batch_size=batch_size
        )
        
        # Compute cosine similarity
        similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return {
            "raw_similarity": similarity_score,
            "is_similar": similarity_score > self.similarity_threshold,
            "normalized_score": min(max(similarity_score, 0), 1)
        }

def load_embeddings_from_dataframe(df: pd.DataFrame, column1: str = None, column2: str = None):
    """Utility function to load embeddings from a DataFrame."""
    # Use column names from config if not provided
    embedding_cols = config['columns']['embeddings']
    input_cols = config['columns']['input']
    
    column1 = column1 or embedding_cols['first_question']
    column2 = column2 or embedding_cols['second_question']
    
    # Combine embeddings and questions
    all_embeddings = pd.concat([df[column1], df[column2]]).reset_index(drop=True)
    all_questions = pd.concat(
        [df[input_cols['first_question']], df[input_cols['second_question']]]
    ).reset_index(drop=True)
    
    return np.stack(all_embeddings.values), all_questions.tolist()

def main():
    """Example usage of similarity search functionality."""
    logger = setup_logger("Similarity Search Main", config['logging']['file']['similarity'])
    
    # Load preprocessed data
    try:
        df = pd.read_pickle(config['paths']['processed']['train'])
        embeddings, questions = load_embeddings_from_dataframe(df)
        
        # Initialize similarity engine
        engine = SimilarityEngine(embeddings, questions)
        
        # Find similar questions
        query = "What is machine learning?"
        similar_questions_faiss = engine.find_similar_questions(
            query, 
            method='faiss'
        )
        similar_questions_cosine = engine.find_similar_questions(
            query, 
            method='cosine'
        )
        
        # Compare two specific questions
        comparator = QuestionSimilarityComparator()
        similarity_metrics = comparator.compute_similarity(
            "What is machine learning?", 
            "Explain the concept of machine learning"
        )
        
        logger.info("FAISS Similar Questions: %s", similar_questions_faiss)
        logger.info("Cosine Similar Questions: %s", similar_questions_cosine)
        logger.info("Similarity Metrics: %s", similarity_metrics)
        
    except Exception as e:
        logger.error(f"Error in similarity search main: {e}")

if __name__ == "__main__":
    main()
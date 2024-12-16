from pathlib import Path
from typing import Optional
import pandas as pd
import yaml

from src.data_preprocessing import QuestionPairPreprocessor, TextCleaningConfig
from src.embeddings import generate_embeddings
from src.similarity_search import SimilarityEngine, QuestionSimilarityComparator
from src.utils.logger import setup_logger

logger = setup_logger("Main Runner", "logs/main.log")

class Pipeline:
    """
    Orchestrates the complete text similarity pipeline.
    
    Handles data preprocessing, embedding generation, and similarity model initialization.
    """
    
    def __init__(self, config_path: Optional[str] = "config/pipeline_config.yaml"):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to pipeline configuration file
        """
        self.logger = logger
        self.config = self._load_config(config_path) if config_path else {}
        self._initialize_paths()
        
        # Initialize components
        self.preprocessor = None
        self.similarity_engine = None
        self.similarity_comparator = None
    
    def _load_config(self, config_path: str) -> dict:
        """Load pipeline configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def _initialize_paths(self):
        """Initialize pipeline paths from config or defaults."""
        self.paths = {
            'raw_data': Path(self.config.get('paths', {}).get('raw_data', 'data/raw')),
            'processed_data': Path(self.config.get('paths', {}).get('processed_data', 'data/processed')),
            'embeddings': Path(self.config.get('paths', {}).get('embeddings', 'data/embeddings')),
            'models': Path(self.config.get('paths', {}).get('models', 'models'))
        }
        
        # Create directories if they don't exist
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def initialize_components(self):
        """Initialize all pipeline components."""
        self.logger.info("Initializing pipeline components...")
        
        # Initialize preprocessor with config
        text_config = TextCleaningConfig(
            **self.config.get('text_cleaning', {})
        )
        self.preprocessor = QuestionPairPreprocessor(text_config)
        
        # Initialize similarity components if data exists
        processed_path = self.paths['processed_data'] / 'train.pkl'
        embeddings_path = self.paths['embeddings'] / 'train_embeddings.pkl'
        
        if processed_path.exists() and embeddings_path.exists():
            self._initialize_similarity_components(processed_path, embeddings_path)
    
    def _initialize_similarity_components(self, processed_path: Path, embeddings_path: Path):
        """Initialize similarity search components."""
        # Load processed data and embeddings
        df = pd.read_pickle(processed_path)
        embeddings_df = pd.read_pickle(embeddings_path)
        
        # Prepare embeddings and questions for similarity engine
        embeddings, questions = self._prepare_similarity_data(df, embeddings_df)
        
        # Initialize similarity components
        model_name = self.config.get('model', {}).get('name', "all-MiniLM-L6-v2")
        self.similarity_engine = SimilarityEngine(embeddings, questions, model_name)
        self.similarity_comparator = QuestionSimilarityComparator(model_name)
    
    def _prepare_similarity_data(self, df: pd.DataFrame, embeddings_df: pd.DataFrame):
        """Prepare data for similarity search components."""
        from src.similarity_search import load_embeddings_from_dataframe
        return load_embeddings_from_dataframe(
            embeddings_df, 
            "q1_embeddings", 
            "q2_embeddings"
        )
    
    def run_pipeline(self, sample_size: Optional[int] = None):
        """
        Run the complete pipeline.
        
        Args:
            sample_size: Optional size to sample from dataset
        """
        try:
            self.logger.info("Starting pipeline execution...")
            
            # Step 1: Preprocess Dataset
            train_path = self.paths['raw_data'] / 'train.csv'
            processed_path = self.paths['processed_data'] / 'train.pkl'
            
            self.logger.info("Starting dataset preprocessing...")
            df = self.preprocessor.preprocess_dataset(
                train_path,
                processed_path,
                sample_size=sample_size
            )
            
            # Step 2: Generate Embeddings
            embeddings_path = self.paths['embeddings'] / 'train_embeddings.pkl'
            self.logger.info("Starting embeddings generation...")
            embeddings_df = generate_embeddings(
                processed_path,
                embeddings_path,
                model_name=self.config.get('model', {}).get('name', "all-MiniLM-L6-v2")
            )
            
            # Step 3: Initialize similarity components
            self._initialize_similarity_components(processed_path, embeddings_path)
            
            self.logger.info("Pipeline execution completed successfully.")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise

def main():
    """Main execution function."""
    try:
        # Initialize and run pipeline
        pipeline = Pipeline()
        pipeline.initialize_components()
        pipeline.run_pipeline(sample_size=1000)
        
        # Example usage of similarity components
        if pipeline.similarity_engine and pipeline.similarity_comparator:
            # Test similarity search
            query = "What is machine learning?"
            similar_questions = pipeline.similarity_engine.find_similar_questions(
                query, top_k=5, method='faiss'
            )
            logger.info(f"Similar questions to '{query}':")
            for q in similar_questions:
                logger.info(f"- {q['question']} (score: {q['similarity_score']:.3f})")
            
            # Test question comparison
            q1 = "What is machine learning?"
            q2 = "Can you explain machine learning?"
            similarity = pipeline.similarity_comparator.compute_similarity(q1, q2)
            logger.info(f"Similarity between questions: {similarity}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
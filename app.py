from flask import Flask, render_template, request
from pathlib import Path
import pandas as pd
from src.similarity_search import SimilarityEngine, QuestionSimilarityComparator
from src.data_preprocessing import preprocess_dataset
from src.embeddings import generate_embeddings
from src.utils.logger import setup_logger

app = Flask(__name__)
logger = setup_logger("Flask App", "logs/app.log")

class DataManager:
    """Manages data processing and embeddings generation."""
    
    def __init__(self):
        self.paths = {
            'raw': Path("data/raw/train.csv"),
            'processed': Path("data/processed/train.pkl"),
            'embeddings': Path("data/embeddings/train_embeddings.pkl")
        }
    
    def check_data_exists(self):
        """Check if all necessary data files exist."""
        return all(path.exists() for path in self.paths.values())
    
    def process_data(self, sample_size=50000):
        """Process raw data and generate embeddings."""
        logger.info("Starting data processing pipeline...")
        
        # Create directories if they don't exist
        for path in self.paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process dataset
        logger.info("Processing dataset...")
        preprocess_dataset(
            str(self.paths['raw']),
            str(self.paths['processed']),
            sample_size=sample_size
        )
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        generate_embeddings(
            str(self.paths['processed']),
            str(self.paths['embeddings'])
        )
        
        logger.info("Data processing complete.")

class SimilarityApp:
    """Manages similarity search functionality."""
    
    def __init__(self):
        self.similarity_engine = None
        self.similarity_comparator = None
        self.data_manager = DataManager()
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize similarity search components."""
        # Check if data exists, if not process it
        if not self.data_manager.check_data_exists():
            logger.info("Required data files not found. Starting data processing...")
            self.data_manager.process_data()
        
        # Load data
        df = pd.read_pickle(self.data_manager.paths['processed'])
        embeddings_df = pd.read_pickle(self.data_manager.paths['embeddings'])
        
        # Initialize components
        from src.similarity_search import load_embeddings_from_dataframe
        embeddings, questions = load_embeddings_from_dataframe(
            embeddings_df, "q1_embeddings", "q2_embeddings"
        )
        
        self.similarity_engine = SimilarityEngine(embeddings, questions)
        self.similarity_comparator = QuestionSimilarityComparator()
        
        logger.info("Similarity components initialized successfully")

# Initialize similarity application
similarity_app = SimilarityApp()

@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")

@app.route("/find_similar", methods=["POST"])
def find_similar():
    """Handle similar questions search request."""
    query_text = request.form["query"]
    top_k = int(request.form.get('top_k', 5))
    method = request.form.get('similarity_method', 'faiss')
    
    similar_questions = similarity_app.similarity_engine.find_similar_questions(
        query_text, top_k, method
    )
    
    return render_template(
        "index.html",
        query=query_text,
        similar_questions=similar_questions,
        selected_method=method
    )

@app.route("/compare_questions", methods=["POST"])
def compare_questions():
    """Handle question comparison request."""
    question1 = request.form["question1"]
    question2 = request.form["question2"]
    
    similarity_metrics = similarity_app.similarity_comparator.compute_similarity(
        question1, question2
    )
    
    return render_template(
        "index.html",
        question1=question1,
        question2=question2,
        similarity_metrics=similarity_metrics
    )

if __name__ == "__main__":
    app.run(debug=True)
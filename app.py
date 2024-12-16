from flask import Flask, render_template, request
from pathlib import Path
import pandas as pd
from src.similarity_search import SimilarityEngine, QuestionSimilarityComparator, load_embeddings_from_dataframe
from src.utils.logger import setup_logger
from src.utils.config import load_config

# Load configuration
config = load_config()
logger = setup_logger("Flask App", config['logging']['file']['app'])

app = Flask(__name__)

class DataManager:
    """Manages data processing and embeddings generation."""
    
    def __init__(self):
        self.config = config
        self.paths = config['paths']
    
    def check_data_exists(self):
        """Check if all necessary data files exist."""
        required_files = [
            self.paths['processed']['train'],
            self.paths['embeddings']['train']
        ]
        return all(Path(path).exists() for path in required_files)

class SimilarityApp:
    """Manages similarity search functionality."""
    
    def __init__(self):
        self.similarity_engine = None
        self.similarity_comparator = None
        self.data_manager = DataManager()
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize similarity search components."""
        if not self.data_manager.check_data_exists():
            raise FileNotFoundError(
                "Required data files not found. Please run the pipeline first."
            )
        
        # Load data
        embeddings_df = pd.read_pickle(self.data_manager.paths['embeddings']['train'])
        
        # Initialize components
        embeddings, questions = load_embeddings_from_dataframe(embeddings_df)
        
        self.similarity_engine = SimilarityEngine(
            embeddings, 
            questions, 
            model_name=config['model']['name']
        )
        self.similarity_comparator = QuestionSimilarityComparator(
            model_name=config['model']['name']
        )
        
        logger.info("Similarity components initialized successfully")

# Initialize similarity application
try:
    similarity_app = SimilarityApp()
except Exception as e:
    logger.error(f"Failed to initialize similarity app: {e}")
    raise

@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")

@app.route("/find_similar", methods=["POST"])
def find_similar():
    """Handle similar questions search request."""
    query_text = request.form["query"]
    top_k = int(request.form.get('top_k', config['similarity']['default_top_k']))
    method = request.form.get('similarity_method', config['similarity']['default_method'])
    
    # Validate top_k
    top_k = min(top_k, config['similarity']['max_top_k'])
    
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
    app.run(
        host=config['app']['host'],
        port=config['app']['port'],
        debug=config['app']['debug']
    )
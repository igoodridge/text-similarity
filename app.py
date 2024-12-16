from flask import Flask, render_template, request, jsonify
from src.similarity_search import SimilaritySearcher
from src.embeddings import generate_embeddings
from src.utils.logger import setup_logger

app = Flask(__name__)

logger = setup_logger("Flask App", "logs/app.log")

searcher=None

# Initialize SimilaritySearcher instance
searcher = SimilaritySearcher(
    questions_path="data/processed/train.pkl",
    embeddings_path="data/embeddings/train_embeddings.pkl",
    column1="q1_embeddings",
    column2="q2_embeddings",
)

# Home route
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle query and find similar questions
@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        global searcher
        if not searcher:
            # Lazily initialize the SimilaritySearcher
            logger.info("Initializing SimilaritySearcher...")
            searcher = SimilaritySearcher(
                questions_path="data/processed/train.pkl",
                embeddings_path="data/embeddings/train_embeddings.pkl",
                column1="q1_embeddings",
                column2="q2_embeddings"
            )
        query_text = request.form["query"]
        top_k = int(request.form.get('top_k', 5))  # Get number of similar questions to return

        similar_questions = searcher.find_similar_questions(query_text, top_k)

        return render_template("index.html", query=query_text, similar_questions=similar_questions)

    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        return render_template('error.html', error=str(e)), 500   
if __name__ == "__main__":
    app.run(debug=True)

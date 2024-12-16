# Question Similarity Search System

A web application that finds similar questions using advanced embedding techniques and allows comparing the similarity between pairs of questions.

## Overview

This system processes question data, generates embeddings using sentence transformers, and provides two main functionalities:
1. Finding similar questions to a given query
2. Computing similarity between two specific questions

The system supports two similarity search methods:
- FAISS (Facebook AI Similarity Search)
- Cosine Similarity

## Project Structure
```
.
├── data/
│   ├── raw/           # Raw question data
│   ├── processed/     # Cleaned and preprocessed data
│   └── embeddings/    # Generated embeddings
├── src/
│   ├── data_preprocessing.py  # Data cleaning and preprocessing
│   ├── embeddings.py         # Embedding generation
│   ├── similarity_search.py  # Similarity computation
│   └── utils/
│       └── logger.py        # Logging configuration
├── static/
│   └── styles.css          # Web app styling
├── templates/
│   └── index.html          # Main web interface
├── app.py                  # Flask application
├── main.py                # Pipeline execution
├── pyproject.toml         # Poetry project configuration
└── poetry.lock           # Poetry dependency lock file
```

## Setup

1. **Install Poetry** (if not already installed)
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   For Windows, run in PowerShell:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```

2. **Install Dependencies**
   ```bash
   poetry install
   ```
   This will create a virtual environment and install all required dependencies.

3. **Prepare Data**
   - Place your question pairs dataset in `data/raw/train.csv`
   - The dataset should have columns: `question1`, `question2`

## Configuration

The project uses `pyproject.toml` for dependency management and project configuration:

```toml
[tool.poetry]
name = "question-similarity-search"
version = "0.1.0"
description = "A system for finding similar questions and computing question similarity"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.8"
flask = "^2.0.1"
faiss-cpu = "^1.7.2"
sentence-transformers = "^2.2.0"
numpy = "^1.21.0"
pandas = "^1.3.0"
nltk = "^3.6.3"
scikit-learn = "^1.0.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0.0"
black = "^22.0.0"
flake8 = "^4.0.0"
```

## Running the Application

1. **Start the Flask App**
   ```bash
   poetry run python app.py
   ```
   - The app will automatically process data and generate embeddings if needed
   - Access the web interface at `http://localhost:5000`

2. **Using the Web Interface**
   - Search for similar questions by entering a query
   - Compare two specific questions for similarity
   - Choose between FAISS and cosine similarity methods

## Features

### Similar Question Search
- Enter a question query
- Select number of similar questions to return (default: 5)
- Choose similarity method (FAISS or Cosine)
- View similarity scores for each match

### Question Comparison
- Input two questions
- Get detailed similarity metrics
- See if questions are considered similar based on threshold

## Technical Details

### Data Processing
- Text cleaning and normalization
- Stopword removal (preserving question words)
- Duplicate removal
- Optional dataset sampling

### Embeddings
- Uses Sentence Transformers for embedding generation
- Default model: 'all-MiniLM-L6-v2'
- Embeddings are cached for performance

### Similarity Search
- FAISS for efficient similarity search
- Cosine similarity alternative
- Configurable similarity thresholds

## Development

1. **Adding Dependencies**
   ```bash
   poetry add package-name
   ```

2. **Adding Development Dependencies**
   ```bash
   poetry add --group dev package-name
   ```

3. **Updating Dependencies**
   ```bash
   poetry update
   ```

4. **Export Requirements** (if needed)
   ```bash
   poetry export -f requirements.txt --output requirements.txt
   ```

## Notes

- First run may take longer due to data processing and embedding generation
- FAISS method is recommended for larger datasets
- System automatically handles missing data files

## Future Improvements

Potential areas for enhancement:
- File upload interface for custom datasets
- Additional similarity metrics
- Batch processing capabilities
- Advanced filtering options
- Result explanations
- Performance optimization for larger datasets

## Support

For issues or questions, please:
1. Check the logs in the `logs/` directory
2. Ensure all dependencies are correctly installed using `poetry install`
3. Verify data file format and location
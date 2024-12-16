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
├── config/
│   └── config.yaml         # Application configuration
├── data/
│   ├── raw/               # Raw question data
│   ├── processed/         # Cleaned and preprocessed data
│   └── embeddings/        # Generated embeddings
├── src/
│   ├── data_preprocessing.py  # Data cleaning and preprocessing
│   ├── embeddings.py         # Embedding generation
│   ├── similarity_search.py  # Similarity computation
│   └── utils/
│       ├── config.py         # Configuration loading utilities
│       └── logger.py         # Logging configuration
├── static/
│   └── styles.css          # Web app styling
├── templates/
│   ├── index.html          # Main web interface
│   └── error.html          # Error page
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

3. **Configure the Application**
   - Copy `config/config.yaml.example` to `config/config.yaml`
   - Adjust settings as needed for your environment

4. **Prepare Data**
   - Place your question pairs dataset in `data/raw/train.csv`
   - The dataset should have columns: `question1`, `question2`

## Configuration

The application uses a YAML configuration file for all settings. Key configuration sections:

### App Settings
```yaml
app:
  name: "Question Similarity Search"
  debug: true
  host: "0.0.0.0"
  port: 5000
```

### Data Paths
```yaml
paths:
  data_dir: "data"
  raw:
    train: "data/raw/train.csv"
    test: "data/raw/test.csv"
  processed:
    train: "data/processed/train.pkl"
    test: "data/processed/test.pkl"
  embeddings:
    train: "data/embeddings/train_embeddings.pkl"
    test: "data/embeddings/test_embeddings.pkl"
```

### Model Configuration
```yaml
model:
  name: "all-MiniLM-L6-v2"
  batch_size: 32
  max_sequence_length: 128

similarity:
  default_method: "faiss"
  default_top_k: 5
  max_top_k: 10
  faiss:
    index_type: "IndexFlatL2"
    normalize_vectors: true
  cosine:
    threshold: 0.7
```

### Preprocessing Settings
```yaml
preprocessing:
  sample_size: 1000
  random_seed: 42
  text_cleaning:
    remove_numbers: true
    remove_punctuation: true
    convert_lowercase: true
    remove_stopwords: true
```

## Running the Application

1. **Process Data and Generate Embeddings**
   ```bash
   poetry run python main.py
   ```
   This will:
   - Process the raw data
   - Generate embeddings
   - Initialize similarity search components

2. **Start the Flask App**
   ```bash
   poetry run python app.py
   ```
   - Access the web interface at `http://localhost:5000` (or configured host/port)

## Features

### Similar Question Search
- Enter a question query
- Select number of similar questions to return (configurable, default: 5)
- Choose similarity method (FAISS or Cosine)
- View similarity scores for each match

### Question Comparison
- Input two questions
- Get detailed similarity metrics
- See if questions are considered similar based on configured threshold

## Development

### Adding Dependencies
```bash
poetry add package-name
```

### Updating Configuration
1. Edit `config/config.yaml`
2. Restart the application to apply changes

### Modifying Preprocessing
1. Adjust preprocessing settings in config.yaml
2. Run main.py to reprocess data

## Technical Details

### Data Processing
- Text cleaning and normalization (configurable)
- Stopword removal (preserving question words)
- Configurable sample size and random seed

### Embeddings
- Uses Sentence Transformers
- Configurable model and batch size
- Cached for performance

### Similarity Search
- FAISS for efficient similarity search
- Configurable similarity thresholds
- Adjustable top-k results

## Notes

- First run may take longer due to data processing and embedding generation
- FAISS method is recommended for larger datasets
- Configuration changes require application restart
- Check logs in configured log directory for troubleshooting

## Support

For issues or questions:
1. Check the configuration in `config.yaml`
2. Review logs in the configured log directory
3. Ensure all dependencies are correctly installed
4. Verify data file format and location
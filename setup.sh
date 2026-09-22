#!/bin/bash

# Check Python 3.11 is active (pyproject.toml is pinned against it -
# other versions aren't verified and may not resolve the same way)
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
if [ "$PYTHON_VERSION" != "3.11" ]; then
    echo "Python 3.11 required, found ${PYTHON_VERSION:-none}. Activate a Python 3.11 environment first."
    exit 1
fi

echo "✅ Python 3.11 found"

# Check Docker is installed and running - needed for Welsh POS tagging
# (see proffiliadur/pos_tagger.py, which calls into this container via
# `docker exec` rather than starting a fresh container per call)
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Please install Docker Desktop first:"
    echo "   https://www.docker.com/products/docker-desktop/"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "Docker is installed but not running. Please start Docker Desktop."
    exit 1
fi

echo "✅ Docker found"

# Start the persistent CyTag container.
CYTAG_IMAGE="ghcr.io/ucrel/cytag:2.0.1"
CYTAG_CONTAINER="cytag-service"

if docker ps --format '{{.Names}}' | grep -qx "$CYTAG_CONTAINER"; then
    echo "✅ CyTag service container already running"
elif docker ps -a --format '{{.Names}}' | grep -qx "$CYTAG_CONTAINER"; then
    docker start "$CYTAG_CONTAINER" > /dev/null
    echo "✅ CyTag service container started"
else
    docker run -d --name "$CYTAG_CONTAINER" \
        --platform linux/amd64 \
        -e PYTHONIOENCODING=utf-8 \
        -e LANG=C.UTF-8 \
        -e LC_ALL=C.UTF-8 \
        --entrypoint tail \
        "$CYTAG_IMAGE" -f /dev/null > /dev/null
    echo "✅ CyTag service container created and started"
fi

pip install -e .
echo "✅ Dependencies installed"

# Pre-download NLTK data and the CEFR model so analysis never depends on
# network access at call time (see proffiliadur/analysis.py, which fails
# fast if the NLTK data is missing rather than downloading silently).
python3 -c "
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
"
echo "✅ NLTK data downloaded"

python3 -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
AutoTokenizer.from_pretrained('cardiffnlp/WelshCEFR-BritLLM')
AutoModelForSequenceClassification.from_pretrained('cardiffnlp/WelshCEFR-BritLLM', dtype='float16')
"
echo "✅ CEFR model downloaded"

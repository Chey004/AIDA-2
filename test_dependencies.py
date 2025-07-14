"""
Test script to verify dependencies are working correctly
"""

import sys
print(f"Python version: {sys.version}")

# Test FastAPI
try:
    import fastapi
    print(f"FastAPI version: {fastapi.__version__}")
except Exception as e:
    print(f"FastAPI import error: {e}")

# Test Pydantic
try:
    import pydantic
    print(f"Pydantic version: {pydantic.__version__}")
except Exception as e:
    print(f"Pydantic import error: {e}")

# Test Uvicorn
try:
    import uvicorn
    print(f"Uvicorn version: {uvicorn.__version__}")
except Exception as e:
    print(f"Uvicorn import error: {e}")

# Test PyTorch
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"PyTorch import error: {e}")

# Test Transformers
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except Exception as e:
    print(f"Transformers import error: {e}")

# Test NumPy
try:
    import numpy
    print(f"NumPy version: {numpy.__version__}")
except Exception as e:
    print(f"NumPy import error: {e}")

# Test scikit-learn
try:
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
except Exception as e:
    print(f"scikit-learn import error: {e}")

# Test spaCy
try:
    import spacy
    print(f"spaCy version: {spacy.__version__}")
except Exception as e:
    print(f"spaCy import error: {e}")

# Test Sentry SDK
try:
    import sentry_sdk
    print(f"Sentry SDK version: {sentry_sdk.__version__}")
except Exception as e:
    print(f"Sentry SDK import error: {e}")

print("\nDependency test completed.")
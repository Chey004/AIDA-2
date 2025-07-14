"""
Minimal working version of the AIDA-2 app
"""

import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)
file_handler = logging.FileHandler('logs/app.log')
logger.addHandler(file_handler)

logger.info("Starting minimal AIDA-2 app")

# Import core components
try:
    import numpy as np
    import torch
    import transformers
    
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Load a simple transformer model
    from transformers import AutoTokenizer, AutoModel
    
    # Initialize a simple model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    logger.info(f"Successfully loaded {model_name}")
    
    # Test the model with a simple input
    text = "Hello, this is a test of the AIDA-2 system."
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    
    logger.info(f"Model output shape: {outputs.last_hidden_state.shape}")
    logger.info("Model test successful")
    
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}")
    raise

logger.info("AIDA-2 minimal app completed successfully")
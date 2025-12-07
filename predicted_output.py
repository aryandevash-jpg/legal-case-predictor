# -*- coding: utf-8 -*-
"""
API for Legal Case Prediction Model
Provides endpoints to predict legal case verdicts and generate summaries
"""

import os
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="Legal Case Predictor API",
    description="API for predicting legal case verdicts using LegalBERT",
    version="1.0.0"
)

# Configuration
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join("content", "ljp_legalbert_model"))
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Label mapping
LABEL_MAP = {
    0: "Negative verdict (e.g., Appeal Dismissed / Not Guilty)",
    1: "Positive verdict (e.g., Appeal Allowed / Guilty)",
}

# Global variables for model, tokenizer, and summarizer
model = None
tokenizer = None
summarizer = None


# Request/Response models
class PredictionRequest(BaseModel):
    text: str = Field(..., description="Legal case text to predict", min_length=1)
    max_summary_length: Optional[int] = Field(200, description="Maximum length of summary", ge=50, le=500)
    min_summary_length: Optional[int] = Field(60, description="Minimum length of summary", ge=20, le=200)


class PredictionResponse(BaseModel):
    pred_label: int = Field(..., description="Predicted label (0 or 1)")
    pred_verdict: str = Field(..., description="Predicted verdict description")
    probability: float = Field(..., description="Confidence score for the prediction")
    all_probabilities: list = Field(..., description="Probabilities for both classes [negative, positive]")
    summary: str = Field(..., description="Summarized version of the case text")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


def load_model_and_tokenizer():
    """Load the trained model and tokenizer from disk"""
    global model, tokenizer
    
    if model is not None and tokenizer is not None:
        return
    
    try:
        model_path = MODEL_DIR
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        print(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"Loading model from {model_path}...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(DEVICE)
        model.eval()
        
        print(f"Model and tokenizer loaded successfully on {DEVICE}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def load_summarizer():
    """Load the BART summarization pipeline"""
    global summarizer
    
    if summarizer is not None:
        return
    
    try:
        print("Loading BART summarizer...")
        device_id = 0 if torch.cuda.is_available() else -1
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn",
            device=device_id
        )
        print("Summarizer loaded successfully")
    except Exception as e:
        print(f"Warning: Failed to load summarizer: {str(e)}")
        summarizer = None


def summarize_facts(text: str, max_length: int = 200, min_length: int = 60) -> str:
    """Summarize the case facts using BART"""
    global summarizer
    
    if summarizer is None:
        return "Summarization not available"
    
    try:
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )[0]["summary_text"]
        return summary
    except Exception as e:
        return f"Error during summarization: {str(e)}"


def predict_case(text: str, max_summary_length: int = 200, min_summary_length: int = 60) -> dict:
    """
    Predict the verdict for a legal case text
    
    Args:
        text: The legal case text
        max_summary_length: Maximum length for summary
        min_summary_length: Minimum length for summary
    
    Returns:
        Dictionary with prediction results
    """
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded. Please ensure the model is available.")
    
    # Tokenize input
    encodings = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))
        prob = float(probs[pred_label])
    
    # Generate summary
    summary = summarize_facts(text, max_summary_length, min_summary_length)
    
    # Return results
    return {
        "pred_label": pred_label,
        "pred_verdict": LABEL_MAP[pred_label],
        "probability": prob,
        "all_probabilities": probs.tolist(),
        "summary": summary,
    }


# Startup event - load models when API starts
@app.on_event("startup")
async def startup_event():
    """Load models when the API starts"""
    try:
        load_model_and_tokenizer()
        load_summarizer()
    except Exception as e:
        print(f"Warning: Could not load models at startup: {str(e)}")
        print("Models will be loaded on first request")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None,
        "device": DEVICE
    }


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict the verdict for a legal case
    
    - **text**: The legal case text to analyze
    - **max_summary_length**: Maximum length of the summary (default: 200)
    - **min_summary_length**: Minimum length of the summary (default: 60)
    
    Returns prediction label, verdict, probabilities, and a summary of the case.
    """
    # Ensure models are loaded
    if model is None or tokenizer is None:
        try:
            load_model_and_tokenizer()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    
    try:
        result = predict_case(
            request.text,
            request.max_summary_length,
            request.min_summary_length
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Legal Case Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


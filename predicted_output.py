import os
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="Legal Case Predictor API",
    description="API for predicting legal case verdicts using LegalBERT",
    version="1.0.0"
)

# Add CORS middleware - must be added before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Configuration
# Use Hugging Face Hub model ID or local path
MODEL_DIR = os.getenv("MODEL_DIR", "AryanJangde/legal-case-predictor-model")
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
    include_reasoning: Optional[bool] = Field(True, description="Whether to include reasoning/explanation for the verdict")


class PredictionResponse(BaseModel):
    pred_label: int = Field(..., description="Predicted label (0 or 1)")
    pred_verdict: str = Field(..., description="Predicted verdict description")
    probability: float = Field(..., description="Confidence score for the prediction")
    all_probabilities: list = Field(..., description="Probabilities for both classes [negative, positive]")
    summary: str = Field(..., description="Summarized version of the case text")
    reasoning: Optional[dict] = Field(None, description="Explanation of the verdict with key factors")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


def load_model_and_tokenizer():
    """Load the trained model and tokenizer from disk or Hugging Face Hub"""
    global model, tokenizer
    
    if model is not None and tokenizer is not None:
        return
    
    try:
        model_path = MODEL_DIR
        
        # Check if it's a local path (exists on filesystem) or Hugging Face Hub ID
        is_local = os.path.exists(model_path) and os.path.isdir(model_path)
        
        if is_local:
            print(f"Loading tokenizer from local path: {model_path}...")
        else:
            print(f"Loading tokenizer from Hugging Face Hub: {model_path}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if is_local:
            print(f"Loading model from local path: {model_path}...")
        else:
            print(f"Loading model from Hugging Face Hub: {model_path}...")
        
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
    
    # Validate input
    if not text or not text.strip():
        return "No text provided for summarization"
    
    # BART requires minimum text length - if text is too short, return as-is
    if len(text.strip().split()) < 10:
        return text.strip()[:200] + "..." if len(text) > 200 else text.strip()
    
    try:
        # Ensure max_length and min_length are reasonable
        # BART has a maximum input length of 1024 tokens
        # Truncate text if it's too long (roughly 4000 chars to be safe)
        processed_text = text.strip()
        if len(processed_text) > 4000:
            processed_text = processed_text[:4000]
        
        # Ensure min_length is less than max_length
        if min_length >= max_length:
            min_length = max(20, max_length - 50)
        
        # Ensure max_length doesn't exceed model limits
        max_length = min(max_length, 142)  # BART-large-cnn max output is 142 tokens
        
        result = summarizer(
            processed_text,
            max_length=max_length,
            min_length=min(min_length, max_length - 20),
            do_sample=False,
            truncation=True
        )
        
        if result and len(result) > 0 and "summary_text" in result[0]:
            return result[0]["summary_text"]
        else:
            return "Summarization returned empty result"
            
    except IndexError as e:
        # Handle index out of range errors
        return f"Text too short or invalid for summarization. Original text: {text[:100]}..."
    except Exception as e:
        # Return a more user-friendly error message
        error_msg = str(e)
        if "index out of range" in error_msg.lower():
            return f"Text format incompatible with summarizer. Showing first {min(200, len(text))} characters: {text[:200]}..."
        return f"Summarization error: {error_msg}"


def extract_important_tokens(text: str, encodings, model_outputs, top_k: int = 20) -> list:
    """
    Extract the most important tokens using integrated gradients
    """
    try:
        import torch.nn.functional as F
        
        # Get input IDs and ensure gradients are enabled
        input_ids = encodings["input_ids"].clone().detach().to(DEVICE)
        input_ids.requires_grad = True
        attention_mask = encodings["attention_mask"].to(DEVICE)
        
        # Forward pass
        model.eval()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Get the predicted class probability
        probs = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1)
        score = probs[0, predicted_class]
        
        # Backward pass
        model.zero_grad()
        score.backward()
        
        # Get gradients
        gradients = input_ids.grad
        
        # Reset model to eval mode
        model.eval()
        
        if gradients is None:
            return []
        
        # Calculate token importance (absolute gradient values)
        token_importance = torch.abs(gradients[0])
        
        # Get top k important tokens
        valid_length = attention_mask[0].sum().item()
        token_importance = token_importance[:valid_length]
        top_indices = torch.topk(token_importance, k=min(top_k, len(token_importance))).indices
        
        # Decode tokens and get importance scores
        important_tokens = []
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0][:valid_length])
        
        for idx in top_indices:
            idx_val = idx.item()
            if idx_val < len(tokens):
                token = tokens[idx_val]
                importance = token_importance[idx_val].item()
                # Filter out special tokens and subword prefixes
                if token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>'] and not token.startswith('##') and len(token.strip()) > 0:
                    important_tokens.append({
                        'token': token.replace('##', ''),
                        'importance': float(importance)
                    })
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for tok in important_tokens:
            if tok['token'].lower() not in seen:
                seen.add(tok['token'].lower())
                unique_tokens.append(tok)
        
        return unique_tokens[:top_k]
    except Exception as e:
        print(f"Error extracting important tokens: {str(e)}")
        # Fallback: extract keywords using simple frequency/position analysis
        words = text.lower().split()
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were'}
        important_words = [w for w in words if w not in stop_words and len(w) > 3]
        # Return most frequent words
        from collections import Counter
        word_freq = Counter(important_words)
        return [{'token': word, 'importance': count/len(important_words)} for word, count in word_freq.most_common(top_k)]


def extract_key_sentences(text: str, encodings, model_outputs, num_sentences: int = 3) -> list:
    """
    Extract key sentences that influenced the prediction
    """
    try:
        import re
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) == 0:
            return []
        
        # Score each sentence by checking its contribution
        sentence_scores = []
        
        for sentence in sentences[:10]:  # Limit to first 10 sentences for performance
            if len(sentence) < 20:
                continue
                
            # Tokenize sentence
            sent_encodings = tokenizer(
                sentence,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)
            
            # Get prediction for this sentence
            with torch.no_grad():
                sent_outputs = model(**sent_encodings)
                sent_logits = sent_outputs.logits
                sent_probs = torch.softmax(sent_logits, dim=-1)
            
            # Score based on how much it aligns with the main prediction
            main_pred = torch.argmax(model_outputs.logits, dim=-1)
            score = sent_probs[0, main_pred].item()
            
            sentence_scores.append({
                'sentence': sentence,
                'score': float(score),
                'relevance': 'high' if score > 0.6 else 'medium' if score > 0.4 else 'low'
            })
        
        # Sort by score and return top sentences
        sentence_scores.sort(key=lambda x: x['score'], reverse=True)
        return sentence_scores[:num_sentences]
        
    except Exception as e:
        print(f"Error extracting key sentences: {str(e)}")
        return []


def generate_reasoning(text: str, pred_label: int, probability: float, important_tokens: list, key_sentences: list) -> dict:
    """
    Generate human-readable reasoning for the verdict
    """
    verdict_type = "positive" if pred_label == 1 else "negative"
    confidence_level = "high" if probability > 0.7 else "medium" if probability > 0.5 else "low"
    
    # Extract key phrases from important tokens
    key_phrases = [t['token'] for t in important_tokens[:10] if len(t['token']) > 2]
    
    # Build reasoning
    reasoning = {
        "confidence_level": confidence_level,
        "confidence_score": float(probability),
        "key_factors": key_phrases[:5],  # Top 5 key factors
        "supporting_evidence": [s['sentence'] for s in key_sentences],
        "explanation": f"The model predicts a {verdict_type} verdict with {confidence_level} confidence ({probability:.1%}). "
                      f"Key factors influencing this decision include: {', '.join(key_phrases[:3])}."
    }
    
    return reasoning


def predict_case(text: str, max_summary_length: int = 200, min_summary_length: int = 60, include_reasoning: bool = True) -> dict:
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
    
    # Extract reasoning if requested
    reasoning = None
    if include_reasoning:
        try:
            # Extract important tokens and sentences
            important_tokens = extract_important_tokens(text, encodings, outputs, top_k=15)
            key_sentences = extract_key_sentences(text, encodings, outputs, num_sentences=3)
            
            # Generate reasoning
            reasoning = generate_reasoning(text, pred_label, prob, important_tokens, key_sentences)
        except Exception as e:
            print(f"Warning: Could not generate reasoning: {str(e)}")
            reasoning = {
                "explanation": "Reasoning generation unavailable. The prediction is based on the overall case text analysis.",
                "confidence_level": "medium" if prob > 0.5 else "low"
            }
    
    # Return results
    result = {
        "pred_label": pred_label,
        "pred_verdict": LABEL_MAP[pred_label],
        "probability": prob,
        "all_probabilities": probs.tolist(),
        "summary": summary,
    }
    
    if reasoning:
        result["reasoning"] = reasoning
    
    return result


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
            request.min_summary_length,
            request.include_reasoning
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Serve frontend
@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the frontend HTML file"""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return html_path
    raise HTTPException(status_code=404, detail="Frontend not found")


# API info endpoint
@app.get("/api/info")
async def api_info():
    return {
        "message": "Legal Case Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "frontend": "/"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


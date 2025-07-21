#!/usr/bin/env python3
"""
Optimized version of the User Belief Extraction API with MPS acceleration
and other performance improvements
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import torch
import re
import numpy as np
from datetime import datetime, timezone
from transformers import pipeline
import logging
import time

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Optimized User Belief Extraction API")

# Performance optimization: Check for best available device
def get_optimal_device():
    """Get the best available device for model inference"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return 0
    else:
        return -1

# Load models with optimal device
optimal_device = get_optimal_device()
logging.info(f"Using device: {optimal_device}")

# Load models with optimizations
try:
    # Use half precision on GPU/MPS for faster inference
    torch_dtype = torch.float16 if optimal_device != -1 else torch.float32
    
    logging.info("Loading models with optimizations...")
    start_time = time.time()
    
    belief_classifier = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli", 
        device=optimal_device,
        torch_dtype=torch_dtype
    )
    
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
        device=optimal_device,
        torch_dtype=torch_dtype
    )
    
    emotion_analyzer = pipeline(
        "text-classification", 
        model="j-hartmann/emotion-english-distilroberta-base", 
        device=optimal_device,
        torch_dtype=torch_dtype
    )
    
    loading_time = time.time() - start_time
    logging.info(f"Models loaded successfully in {loading_time:.2f} seconds")
    
except Exception as e:
    logging.error(f"Failed to load models with optimization: {e}")
    logging.info("Falling back to CPU...")
    
    belief_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=-1)
    emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=-1)

# Pydantic object models
class ConversationEntry(BaseModel):
    ref_conversation_id: int
    ref_user_id: int
    transaction_datetime_utc: str
    screen_name: str
    message: str

class ConversationPayload(BaseModel):
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    entries: List[ConversationEntry] = Field(..., description="List of conversation messages")

class BeliefExtraction(BaseModel):
    message: str
    screen_name: str
    timestamp: str
    beliefs: List[Dict[str, Any]]
    sentiment: Dict[str, Any]
    emotions: Dict[str, Any]
    linguistic_features: Dict[str, Any]

class ConversationInsights(BaseModel):
    conversation_id: str
    user_id: int
    analysis_timestamp: str
    message_count: int
    user_message_count: int
    conversation_duration_minutes: Optional[float]
    overall_sentiment_trend: Dict[str, Any]
    dominant_beliefs: List[Dict[str, Any]]
    confidence_indicators: Dict[str, Any]
    processing_time_seconds: float  # Add processing time tracking

class BeliefResponse(BaseModel):
    conversation_id: str
    conversation_insights: ConversationInsights
    message_analyses: List[BeliefExtraction]
    recommendations: Dict[str, List[str]]

# Optimized belief categories 
BELIEF_CATEGORIES = {
    "self_confidence": [
        "I am confident and capable",
        "I doubt my abilities",
        "I believe in myself",
        "I feel insecure about my skills"
    ],
    "self_worth": [
        "I am valuable and worthy",
        "I struggle with self-worth",
        "I deserve good things",
        "I feel unworthy or inadequate"
    ],
    "social_connection": [
        "I value meaningful relationships",
        "I feel lonely and isolated",
        "I am good at connecting with others",
        "I struggle to make connections"
    ],
    "personal_growth": [
        "I want to learn and grow",
        "I feel stuck in my situation",
        "I am open to change",
        "I resist new experiences"
    ],
    "creativity_expression": [
        "I am creative and expressive",
        "I lack creative abilities",
        "I enjoy artistic pursuits",
        "I am not an artistic person"
    ],
    "physical_wellbeing": [
        "I am physically capable",
        "I have physical limitations",
        "I value staying active",
        "I struggle with physical challenges"
    ],
    "purpose_meaning": [
        "I have a clear sense of purpose",
        "I seek meaning in life",
        "I feel lost or directionless",
        "I know what matters to me"
    ],
    "leadership_influence": [
        "I am a natural leader",
        "I prefer to follow others",
        "I can influence positive change",
        "I feel powerless to make a difference"
    ]
}

ALL_BELIEF_TOPICS = []
for category, topics in BELIEF_CATEGORIES.items():
    ALL_BELIEF_TOPICS.extend(topics)

# Optimized batch processing function
def batch_analyze_beliefs(messages: List[str], batch_size: int = 3):
    """Process multiple messages in batches for better throughput"""
    results = []
    
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i+batch_size]
        batch_results = []
        
        for message in batch:
            # Process each message (transformers doesn't support true batching for zero-shot)
            result = belief_classifier(message, ALL_BELIEF_TOPICS, multi_class=True)
            batch_results.append(result)
        
        results.extend(batch_results)
    
    return results

# Extract linguistic features
def extract_linguistic_features(text: str) -> Dict[str, Any]:
    """Extract linguistic features from text (optimized version)"""
    words = text.split()
    sentences = text.split('.')
    
    # Use compiled regex for better performance
    first_person_pattern = re.compile(r'\b(I|me|my|myself|mine)\b', re.IGNORECASE)
    certainty_pattern = re.compile(r'\b(definitely|certainly|absolutely|sure|confident|know)\b', re.IGNORECASE)
    uncertainty_pattern = re.compile(r'\b(maybe|perhaps|might|could|unsure|uncertain|think|guess)\b', re.IGNORECASE)
    past_pattern = re.compile(r'\b\w+ed\b|\bwas\b|\bwere\b|\bhad\b', re.IGNORECASE)
    future_pattern = re.compile(r'\bwill\b|\bgoing to\b|\bplan to\b|\bwant to\b', re.IGNORECASE)
    
    first_person_count = len(first_person_pattern.findall(text))
    certainty_words = certainty_pattern.findall(text)
    uncertainty_words = uncertainty_pattern.findall(text)
    past_tense = len(past_pattern.findall(text))
    future_tense = len(future_pattern.findall(text))
    
    return {
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "first_person_ratio": first_person_count / max(len(words), 1),
        "certainty_score": len(certainty_words) / max(len(words), 1),
        "uncertainty_score": len(uncertainty_words) / max(len(words), 1),
        "temporal_focus": {
            "past_indicators": past_tense,
            "future_indicators": future_tense
        }
    }

def analyze_message(message: str, screen_name: str, timestamp: str) -> BeliefExtraction:
    """Analyze a single message (optimized version)"""
    
    belief_results = belief_classifier(message, ALL_BELIEF_TOPICS, multi_class=True)
    beliefs = [
        {"label": label, "score": round(score, 3)}
        for label, score in zip(belief_results["labels"], belief_results["scores"])
        if score > 0.3
    ]
    
    sentiment_result = sentiment_analyzer(message)[0]
    sentiment = {
        "label": sentiment_result["label"],
        "score": round(sentiment_result["score"], 3)
    }
    
    emotion_result = emotion_analyzer(message)[0]
    emotions = {
        "primary_emotion": emotion_result["label"],
        "confidence": round(emotion_result["score"], 3)
    }
    
    linguistic_features = extract_linguistic_features(message)
    
    return BeliefExtraction(
        message=message,
        screen_name=screen_name,
        timestamp=timestamp,
        beliefs=beliefs,
        sentiment=sentiment,
        emotions=emotions,
        linguistic_features=linguistic_features
    )

def generate_conversation_insights(conversation_id: str, message_analyses: List[BeliefExtraction], entries: List[ConversationEntry], processing_time: float) -> ConversationInsights:
    """Generate high-level insights from conversation analysis"""
    
    user_messages = [msg for msg in message_analyses if msg.screen_name.lower() != "storybot"]
    user_entries = [entry for entry in entries if entry.screen_name.lower() != "storybot"]
    
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user messages found for analysis")
    
    user_id = user_entries[0].ref_user_id if user_entries else 0
    
    # Calculate conversation duration
    timestamps = [datetime.fromisoformat(entry.transaction_datetime_utc.replace('Z', '+00:00')) for entry in entries]
    duration_minutes = (max(timestamps) - min(timestamps)).total_seconds() / 60 if len(timestamps) > 1 else 0
    
    # Aggregate sentiment trend
    sentiments = [msg.sentiment for msg in user_messages]
    positive_count = sum(1 for s in sentiments if s["label"] in ["POSITIVE", "positive"])
    negative_count = sum(1 for s in sentiments if s["label"] in ["NEGATIVE", "negative"])
    
    # Dominant beliefs across conversation
    all_beliefs = []
    for msg in user_messages:
        all_beliefs.extend(msg.beliefs)
    
    belief_aggregation = {}
    for belief in all_beliefs:
        label = belief["label"]
        if label in belief_aggregation:
            belief_aggregation[label].append(belief["score"])
        else:
            belief_aggregation[label] = [belief["score"]]
    
    dominant_beliefs = [
        {
            "belief": belief,
            "frequency": len(scores),
            "avg_confidence": round(np.mean(scores), 3),
            "max_confidence": round(max(scores), 3)
        }
        for belief, scores in belief_aggregation.items()
    ]
    dominant_beliefs.sort(key=lambda x: (x["frequency"], x["avg_confidence"]), reverse=True)
    
    # Confidence indicators
    avg_certainty = np.mean([msg.linguistic_features["certainty_score"] for msg in user_messages])
    avg_uncertainty = np.mean([msg.linguistic_features["uncertainty_score"] for msg in user_messages])
    
    confidence_indicators = {
        "certainty_level": round(avg_certainty, 3),
        "uncertainty_level": round(avg_uncertainty, 3),
        "confidence_trend": "increasing" if avg_certainty > avg_uncertainty else "decreasing",
        "first_person_usage": round(np.mean([msg.linguistic_features["first_person_ratio"] for msg in user_messages]), 3)
    }
    
    return ConversationInsights(
        conversation_id=conversation_id,
        user_id=user_id,
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        message_count=len(entries),
        user_message_count=len(user_messages),
        conversation_duration_minutes=round(duration_minutes, 2),
        overall_sentiment_trend={
            "positive_messages": positive_count,
            "negative_messages": negative_count,
            "sentiment_ratio": round(positive_count / max(positive_count + negative_count, 1), 3)
        },
        dominant_beliefs=dominant_beliefs[:5],
        confidence_indicators=confidence_indicators,
        processing_time_seconds=round(processing_time, 2)
    )

def generate_recommendations(insights: ConversationInsights, message_analyses: List[BeliefExtraction]) -> Dict[str, List[str]]:
    """Generate recommendations for different teams based on analysis"""
    
    recommendations = {
        "content_team": [],
        "storybot_team": [],
        "community_team": [],
        "value_assessment_team": []
    }
    
    dominant_beliefs = [b["belief"] for b in insights.dominant_beliefs[:3]]
    if any("confident" in belief.lower() for belief in dominant_beliefs):
        recommendations["content_team"].append("Provide content that reinforces confidence-building themes")
    if any("struggle" in belief.lower() or "insecure" in belief.lower() for belief in dominant_beliefs):
        recommendations["content_team"].append("Curate supportive and empowering content")
    
    if insights.confidence_indicators["uncertainty_level"] > 0.1:
        recommendations["storybot_team"].append("Use more affirming and supportive language")
    if insights.overall_sentiment_trend["sentiment_ratio"] < 0.5:
        recommendations["storybot_team"].append("Focus on positive engagement strategies")
    
    if any("lonely" in belief.lower() or "connection" in belief.lower() for belief in dominant_beliefs):
        recommendations["community_team"].append("Facilitate social connections and group activities")
    if insights.user_message_count > 5:
        recommendations["community_team"].append("User shows high engagement")
    
    if insights.conversation_duration_minutes > 30:
        recommendations["value_assessment_team"].append("High-value conversation with extended engagement")
    if len(insights.dominant_beliefs) > 3:
        recommendations["value_assessment_team"].append("Rich belief expression - valuable for analysis")
    
    return recommendations

@app.post("/extract_beliefs", response_model=BeliefResponse)
async def extract_beliefs(convo: ConversationPayload):
    """
    Optimized conversation analysis with performance tracking
    """
    start_time = time.time()
    
    try:
        logging.info(f"Processing conversation {convo.conversation_id} with {len(convo.entries)} messages")
        
        # Analyze each user message
        message_analyses = []
        user_entries = [entry for entry in convo.entries if entry.screen_name.lower() != "storybot"]
        
        logging.info(f"Analyzing {len(user_entries)} user messages...")
        
        for entry in user_entries:
            analysis = analyze_message(entry.message, entry.screen_name, entry.transaction_datetime_utc)
            message_analyses.append(analysis)
        
        if not message_analyses:
            raise HTTPException(status_code=400, detail="No user messages found for analysis")
        
        processing_time = time.time() - start_time
        
        # Generate conversation-level insights
        insights = generate_conversation_insights(convo.conversation_id, message_analyses, convo.entries, processing_time)
        
        # Generate recommendations
        recommendations = generate_recommendations(insights, message_analyses)
        
        total_time = time.time() - start_time
        logging.info(f"Analysis complete for conversation {convo.conversation_id} in {total_time:.2f}s")
        
        return BeliefResponse(
            conversation_id=convo.conversation_id,
            conversation_insights=insights,
            message_analyses=message_analyses,
            recommendations=recommendations
        )
        
    except Exception as e:
        logging.error(f"Error processing conversation {convo.conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("optimize_api:app", host="0.0.0.0", port=8001, reload=True)
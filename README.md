# User Belief Extraction API

A REST API endpoint that analyzes conversation data to determine what users believe about themselves and various topics. The API uses machine learning models to extract beliefs, sentiment, emotions, and linguistic features from conversations, providing actionable insights for different teams.

## Features

- **Multi-Model Analysis**: Uses BART-MNLI for belief classification, RoBERTa for sentiment analysis, and DistilRoBERTa for emotion detection
- **Feature Extraction**: Extracts linguistic features including certainty indicators, temporal focus, and first-person usage patterns
- **Belief Categorization**: Organizes beliefs into 8 categories: self-confidence, self-worth, social connection, personal growth, creativity/expression, physical wellbeing, purpose/meaning, and leadership/influence
- **Conversation Insights**: Provides high-level analysis including sentiment trends and confidence indicators
- **Team Recommendations**: Generates specific recommendations for content, StoryBot, community, and value assessment teams
- **Comprehensive Output**: Structured JSON response designed for consumption by different teams

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 4GB RAM (for model loading)
- GPU support optional but recommended for better performance

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd lore-take-home
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained models** (happens automatically on first run):
   The API will automatically download the following models:
   - `facebook/bart-large-mnli` (~1.6GB)
   - `cardiffnlp/twitter-roberta-base-sentiment-latest` (~500MB)
   - `j-hartmann/emotion-english-distilroberta-base` (~250MB)

### Running the API

**Start the server**:
```bash
python optimize_api.py
```

The API will be available at `http://localhost:8001`

**View API documentation**:
- Interactive docs: `http://localhost:8001/docs`
- OpenAPI schema: `http://localhost:8001/redoc`

## API Usage

### Endpoint

**POST** `/extract_beliefs`

### Request Format

```json
{
  "conversation_id": "string",
  "entries": [
    {
      "ref_conversation_id": 12345,
      "ref_user_id": 678,
      "transaction_datetime_utc": "2023-10-01T10:15:00Z",
      "screen_name": "UserName",
      "message": "I'm feeling really confident about my new project."
    }
  ]
}
```

### Response Format

```json
{
  "conversation_id": "string",
  "conversation_insights": {
    "conversation_id": "string",
    "user_id": 678,
    "analysis_timestamp": "2024-01-15T10:30:00Z",
    "message_count": 10,
    "user_message_count": 5,
    "conversation_duration_minutes": 45.5,
    "overall_sentiment_trend": {
      "positive_messages": 3,
      "negative_messages": 2,
      "sentiment_ratio": 0.6
    },
    "dominant_beliefs": [
      {
        "belief": "I am confident and capable",
        "frequency": 2,
        "avg_confidence": 0.85,
        "max_confidence": 0.92
      }
    ],
    "confidence_indicators": {
      "certainty_level": 0.15,
      "uncertainty_level": 0.05,
      "confidence_trend": "increasing",
      "first_person_usage": 0.25
    }
  },
  "message_analyses": [
    {
      "message": "I'm feeling really confident about my new project.",
      "screen_name": "UserName",
      "timestamp": "2023-10-01T10:15:00Z",
      "beliefs": [
        {
          "label": "I am confident and capable",
          "score": 0.923
        }
      ],
      "sentiment": {
        "label": "POSITIVE",
        "score": 0.887
      },
      "emotions": {
        "primary_emotion": "joy",
        "confidence": 0.78
      },
      "linguistic_features": {
        "word_count": 8,
        "sentence_count": 1,
        "first_person_ratio": 0.25,
        "certainty_score": 0.0,
        "uncertainty_score": 0.0,
        "temporal_focus": {
          "past_indicators": 1,
          "future_indicators": 0
        }
      }
    }
  ],
  "recommendations": {
    "content_team": ["Provide content that reinforces confidence-building themes"],
    "storybot_team": [],
    "community_team": ["User shows high engagement - consider for community leadership roles"],
    "value_assessment_team": ["Rich belief expression - valuable for analysis"]
  }
}
```

## Testing the API

### Unit testing

Run python test file:

```python
python test_api.py
```

### Using Python Requests

Once the server has been started, either use the following python code, or use the `testing.ipynb` notebook.

```python
import requests
import json

url = "http://localhost:8001/extract_beliefs"
data = {
    "conversation_id": "test_conversation",
    "entries": [
        {
            "ref_conversation_id": 98696,
            "ref_user_id": 782,
            "transaction_datetime_utc": "2023-10-01T10:15:00Z",
            "screen_name": "TestUser",
            "message": "I feel really confident about my abilities today!"
        }
    ]
}

response = requests.post(url, json=data)
print(json.dumps(response.json(), indent=2))
```

### Expected Test Output

The API should return:
- Belief classifications for user messages (StoryBot messages are filtered out)
- Sentiment analysis (positive/negative/neutral)
- Emotion detection (joy, sadness, anger, fear, surprise, disgust, neutral)
- Linguistic feature analysis
- Conversation-level insights and trends
- Team-specific recommendations

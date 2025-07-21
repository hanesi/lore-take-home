#!/usr/bin/env python3
"""
Test script for the User Belief Extraction API
"""

import json
import sys
from typing import Dict, Any

def create_test_payload() -> Dict[str, Any]:
    """Create a test conversation payload"""
    return {
        "conversation_id": "test_conversation_123",
        "entries": [
            {
                "ref_conversation_id": 98696,
                "ref_user_id": 782,
                "transaction_datetime_utc": "2023-10-01T10:15:00Z",
                "screen_name": "ChattyPenguin",
                "message": "Hello StoryBot, I'm having a tough time with this app. My fingers aren't what they used to be. Can you help me?"
            },
            {
                "ref_conversation_id": 98696,
                "ref_user_id": 1,
                "transaction_datetime_utc": "2023-10-01T10:20:00Z",
                "screen_name": "StoryBot",
                "message": "Hello ChattyPenguin! I'm here to help. Can you tell me what issues you're experiencing with the app?"
            },
            {
                "ref_conversation_id": 98696,
                "ref_user_id": 782,
                "transaction_datetime_utc": "2023-10-01T10:25:00Z",
                "screen_name": "ChattyPenguin",
                "message": "Well, I used to be quite confident with technology, but lately I feel like I'm struggling. I want to stay connected with my family but sometimes I feel so frustrated."
            },
            {
                "ref_conversation_id": 98696,
                "ref_user_id": 1,
                "transaction_datetime_utc": "2023-10-01T10:30:00Z",
                "screen_name": "StoryBot",
                "message": "I understand how frustrating that can feel. You're doing great by reaching out for help. What specific features are giving you trouble?"
            },
            {
                "ref_conversation_id": 98696,
                "ref_user_id": 782,
                "transaction_datetime_utc": "2023-10-01T10:35:00Z",
                "screen_name": "ChattyPenguin",
                "message": "I really value connecting with others, but I'm not sure I'm good at this technology stuff anymore. Maybe I'm just too old to learn new things."
            }
        ]
    }

def test_import():
    """Test that we can import the API module"""
    try:
        from dev.self_belief_api import app, extract_beliefs
        print("✓ Successfully imported API module")
        return True
    except Exception as e:
        print(f"✗ Failed to import API module: {e}")
        return False

def test_pydantic_models():
    """Test Pydantic model validation"""
    try:
        from dev.self_belief_api import ConversationPayload, BeliefResponse
        
        # Test valid payload
        test_data = create_test_payload()
        payload = ConversationPayload(**test_data)
        print(f"✓ Successfully created ConversationPayload with {len(payload.entries)} entries")
        
        # Test that StoryBot messages are included in payload but will be filtered during processing
        storybot_messages = [entry for entry in payload.entries if entry.screen_name.lower() == "storybot"]
        user_messages = [entry for entry in payload.entries if entry.screen_name.lower() != "storybot"]
        print(f"✓ Payload contains {len(user_messages)} user messages and {len(storybot_messages)} StoryBot messages")
        
        return True
    except Exception as e:
        print(f"✗ Pydantic model test failed: {e}")
        return False

def test_feature_extraction():
    """Test linguistic feature extraction function"""
    try:
        from dev.self_belief_api import extract_linguistic_features
        
        test_message = "I am confident that I can definitely learn new things, but sometimes I think I might struggle."
        features = extract_linguistic_features(test_message)
        
        expected_keys = ["word_count", "sentence_count", "first_person_ratio", "certainty_score", "uncertainty_score", "temporal_focus"]
        for key in expected_keys:
            if key not in features:
                raise ValueError(f"Missing key: {key}")
        
        print("✓ Feature extraction function works correctly")
        print(f"  - Word count: {features['word_count']}")
        print(f"  - First person ratio: {features['first_person_ratio']:.3f}")
        print(f"  - Certainty score: {features['certainty_score']:.3f}")
        print(f"  - Uncertainty score: {features['uncertainty_score']:.3f}")
        return True
    except Exception as e:
        print(f"✗ Feature extraction test failed: {e}")
        return False

def test_belief_categories():
    """Test belief categorization"""
    try:
        from dev.self_belief_api import BELIEF_CATEGORIES, ALL_BELIEF_TOPICS
        
        print(f"✓ Loaded {len(BELIEF_CATEGORIES)} belief categories")
        print(f"✓ Total belief topics: {len(ALL_BELIEF_TOPICS)}")
        
        # Check that we have the expected categories
        expected_categories = ["self_confidence", "self_worth", "social_connection", "personal_growth"]
        for cat in expected_categories:
            if cat not in BELIEF_CATEGORIES:
                raise ValueError(f"Missing category: {cat}")
        
        print("✓ All expected belief categories present")
        return True
    except Exception as e:
        print(f"✗ Belief category test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running User Belief Extraction API Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_import),
        ("Pydantic Models Test", test_pydantic_models),
        ("Feature Extraction Test", test_feature_extraction),
        ("Belief Categories Test", test_belief_categories),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The API is ready for use.")
        print("\nTo start the API server, run:")
        print("python self_belief_api.py")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
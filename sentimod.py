import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pickle

# Load T5 tokenizer and model with CPU fallback
def load_model_and_tokenizer():
    tokenizer_loaded = T5Tokenizer.from_pretrained('saved_model')
    model_loaded = T5ForConditionalGeneration.from_pretrained('saved_model')
    model_loaded.to(torch.device('cpu'))  # Move model to CPU

    # Load classifier using pickle
    with open('saved_model/classifier.pkl', 'rb') as f:
        classifier_loaded = pickle.load(f)

    print("Model, tokenizer, and classifier loaded.")
    return tokenizer_loaded, model_loaded, classifier_loaded

def generate_summary(input_text, tokenizer, model):
    input_ids = tokenizer(f"summarize: {input_text}", return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary

def classify_sentiment(input_text, classifier):
    labels = ["Positive", "Negative", "Neutral"]
    classification_result = classifier(input_text, labels)
    predicted_sentiment = classification_result['labels'][0]
    confidence_score = int(classification_result['scores'][0] * 100)
    return predicted_sentiment, confidence_score

def generate_recommendation(sentiment, confidence_score):
    if sentiment == "Positive":
        if confidence_score > 90:
            recommendation = "Strong Buy"
        elif confidence_score > 75:
            recommendation = "Buy"
        else:
            recommendation = "Consider Buying"
    elif sentiment == "Negative":
        if confidence_score > 90:
            recommendation = "Strong Sell"
        elif confidence_score > 75:
            recommendation = "Sell"
        else:
            recommendation = "Consider Selling"
    else:
        if confidence_score > 80:
            recommendation = "Hold"
        else:
            recommendation = "Neutral"

    return recommendation

def generate_output(input_text, tokenizer, model, classifier):
    summary = generate_summary(input_text, tokenizer, model)
    sentiment, confidence_score = classify_sentiment(input_text, classifier)
    recommendation = generate_recommendation(sentiment, confidence_score)

    result = {
        "Sentiment": sentiment,
        "Confidence Score": confidence_score,
        "Summary": summary,
        "Recommendation": recommendation
    }
    return result

def test_saved_model(input_text):
    tokenizer, model, classifier = load_model_and_tokenizer()
    result = generate_output(input_text, tokenizer, model, classifier)
    print("Generated Output:", result)

# Example Input
input_text =input("Mnagamanga:")
test_saved_model(input_text)

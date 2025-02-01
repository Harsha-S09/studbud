# studbud
# Install necessary libraries
!pip install torch transformers

# Import necessary libraries
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize the BERT model and tokenizer
def initialize_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8)
    return tokenizer, model

# Generate a study plan based on user input
def generate_study_plan(study_goal, subject, strengths, weaknesses, available_time, learning_method, tokenizer, model):
    """
    Generate a study plan based on the user's input.

    Args:
        study_goal (str): User's study goal.
        subject (str): Subject to focus on.
        strengths (str): User's strengths.
        weaknesses (str): User's weaknesses.
        available_time (str): Available study time.
        learning_method (str): Preferred learning method.
        tokenizer: Pre-trained BERT tokenizer.
        model: Pre-trained BERT model.

    Returns:
        str: Generated study plan.
    """
    prompt = (f"Study goal: {study_goal}\n"
              f"Subject: {subject}\n"
              f"Strengths: {strengths}\n"
              f"Weaknesses: {weaknesses}\n"
              f"Available time: {available_time}\n"
              f"Learning method: {learning_method}")

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
   
    # Get the model output
    outputs = model(**inputs)
    logits = outputs.logits
    max_index = torch.argmax(logits).item()
   
    # Map the model's output to a study plan (example mapping)
    study_plans = [
        "Focus on core concepts and practice quizzes.",
        "Divide topics into smaller chunks and allocate time accordingly.",
        "Emphasize weak areas and review frequently.",
        "Adopt visual learning techniques like mind maps.",
        "Spend more time on practice exams and problem-solving.",
        "Combine group study sessions with self-paced learning.",
        "Allocate consistent study hours daily with breaks.",
        "Experiment with different methods to find what works best."
    ]
   
    # Return the generated study plan
    return study_plans[max_index]

# Initialize the model
print("Initializing BERT model...")
tokenizer, model = initialize_bert_model()
print("Model initialized.")

# Input data from the user
study_goal = input("Enter your study goal: ")
subject = input("Enter the subject to focus on: ")
strengths = input("Enter your strengths: ")
weaknesses = input("Enter your weaknesses: ")
available_time = input("Enter your available study time: ")
learning_method = input("Enter your preferred learning method: ")

# Generate the study plan
print("\nGenerating your personalized study plan...")
study_plan = generate_study_plan(
    study_goal, subject, strengths, weaknesses, available_time, learning_method, tokenizer, model
)
print("\nYour personalized study plan:")
print(study_plan)

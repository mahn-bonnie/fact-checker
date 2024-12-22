import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
import gdown
import os

# Define the model file and its Google Drive ID
MODEL_ID = "1ORYi8TRcLy-sL9Q_y3d0C42Eox5DX_Tu"  # Replace with your actual file ID
MODEL_PATH = './c2_new_model_weights.pt'  # Path to store the model file locally

# Download model function
def download_model(model_id, model_path):
    url = f"https://drive.google.com/uc?id={model_id}"
    if not os.path.exists(model_path):
        st.write("Downloading model...")
        gdown.download(url, model_path, quiet=False)
        st.write("Model downloaded successfully!")
    else:
        st.write("Model already exists locally.")

# Define custom model with the correct architecture
class CustomBERTForSequenceClassification(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        super(CustomBERTForSequenceClassification, self).__init__()

        # Load the base BERT model
        self.bert = BertModel.from_pretrained(model_name)

        # Add custom classification layers
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)  # First custom layer
        self.fc2 = nn.Linear(512, num_labels)  # Second custom layer for final output

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Forward pass through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Use the [CLS] token representation
        cls_token_output = outputs.last_hidden_state[:, 0, :]

        # Pass through custom layers
        x = torch.relu(self.fc1(cls_token_output))
        logits = self.fc2(x)

        return logits

# Function to load the model
def load_model(model_path):
    try:
        # Initialize the custom model architecture
        model = CustomBERTForSequenceClassification()
        
        # Load pre-trained weights
        model_weights = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_weights)
        
        # Set model to evaluation mode
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to load the tokenizer
def load_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        return tokenizer
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None

# Ensure the model is downloaded and loaded
download_model(MODEL_ID, MODEL_PATH)
model = load_model(MODEL_PATH)
tokenizer = load_tokenizer()

# Streamlit app interface
st.title("Fake News Detection App")
st.write("Enter the text of the news article below to predict if it's real or fake.")

input_text = st.text_area("News Article Content")

if st.button("Predict"):
    if input_text:
        try:
            # Tokenize input text
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

            # Perform prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs
                prediction = torch.argmax(logits, dim=1).item()

            # Map prediction to labels
            result = "Fake News" if prediction == 1 else "Real News"
            st.subheader(f"Prediction: {result}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter some text to predict.")


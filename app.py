import streamlit as st
import torch
from transformers import AutoTokenizer
import os
import gdown
import torch.nn as nn
from transformers import BertModel

# Define the model file and path to download it
MODEL_ID = "1ORYi8TRcLy-sL9Q_y3d0C42Eox5DX_Tu"  # Google Drive file ID
MODEL_PATH = './model/c2_new_model_weights.pt'  # Local path to store the model file

# Download model function
def download_model(model_id, model_path):
    url = f"https://drive.google.com/uc?id={model_id}"
    output = model_path
    gdown.download(url, output, quiet=False)

# Download the model if it is not already present
if not os.path.exists(MODEL_PATH):
    st.write("Model not found locally, downloading...")
    download_model(MODEL_ID, MODEL_PATH)
    st.write("Model downloaded successfully!")

# Define your custom model with the correct layer dimensions
class CustomBERTForSequenceClassification(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        super(CustomBERTForSequenceClassification, self).__init__()

        self.bert = BertModel.from_pretrained(model_name)
        
        # Adjust the custom layers to match the saved model's architecture
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)  # Update to match saved model's fc1 dimensions
        self.fc2 = nn.Linear(512, num_labels)  # Update to match saved model's fc2 dimensions

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Pass the input through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # Extract the [CLS] token output
        cls_token_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through the custom layers
        x = torch.relu(self.fc1(cls_token_output))
        logits = self.fc2(x)
        
        return logits

# Function to load the model with custom layers
def load_model(model_path):
    try:
        model = CustomBERTForSequenceClassification()  # Load custom model with updated layers
        model_weights = torch.load(model_path)  # Load saved weights
        model.load_state_dict(model_weights)  # Load weights into the model
        model.eval()  # Set model to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to load tokenizer
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer

# Initialize the model and tokenizer
model = load_model(MODEL_PATH)
tokenizer = load_tokenizer()

# Streamlit app layout
st.title("Fake News Detection")
st.write("Enter news content below to predict if it is fake or real.")

input_text = st.text_area("Enter news text here:")

if st.button("Predict"):
    if input_text:
        try:
            # Tokenize input text and prepare it for the model
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # Perform prediction
            with torch.no_grad():
                outputs = model(**inputs)  # Forward pass
                logits = outputs
                prediction = torch.argmax(logits, dim=1).item()  # Get predicted class index

            result = "Fake News" if prediction == 1 else "Real News"
            st.subheader(f"Prediction: {result}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter some news text to predict.")

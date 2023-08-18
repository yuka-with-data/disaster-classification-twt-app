""" 
Streamlit App Setup
Natural Disaster Tweet Binary Classification 
User input: Insert Tweet 
Output: Class 0 or 1

 """
# Load required libraries
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import streamlit as st
from preprocess_tweet import preprocess_tweet_2

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the trained model checkpoint 
model_path = './saved_model/'
model = BertForSequenceClassification.from_pretrained(model_path)

# Set the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load model to the decive
model.to(device)

# Define the prediction function
# text input - user input
def predict(text):
    preprocessd_tweet = preprocess_tweet_2(text)
    encoding = tokenizer(preprocessd_tweet, truncation=True, padding=True, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        # No gradient calculation are performed during the prediction.
        outputs = model(input_ids, attention_mask)
        pred_class = torch.argmax(outputs.logits).item()
        # return
        return pred_class

def main():
    """ 
     Streamlit Functionality:
     Set up the user input prompt
     Set up the Classify button to run the predict function
       """
    st.title("🌋Natural Disaster Tweet Classification Checker App")
    st.write("🗒️This demo app is powered by the base BERT model with fine-tuned checkpoint.")
    user_input = st.text_input("Enter a Tweet")
    # Classify button
    if st.button("Classify"):
        # execute predict function
        prediction = predict(user_input)
        # Print outcome
        st.write("Predicted Class: ", prediction)
        st.write("0: Not Natural Disaster Related Tweet")
        st.write("1: Natural Disaster Related Tweet")

# call main 
if __name__ == "__main__":
    main()
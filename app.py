import requests
from bs4 import BeautifulSoup
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from summarizer import Summarizer
import torch

# --- Page Config ---
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="🤖",
    layout="centered"
)

# --- Load Models (Cached for Performance) ---
@st.cache_resource
def load_t5_model():
    """Loads the T5 model for abstractive summarization."""
    model_name = "t5-small" # Change to "t5-base" for better results (but slower)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_bert_model():
    """Loads the BERT model for extractive summarization."""
    return Summarizer()

# --- Summarization Functions ---
def t5_summarize(text, max_len, min_len):
    tokenizer, model = load_t5_model()
    input_text = "summarize: " + text
    
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    summary_ids = model.generate(
        input_ids,
        max_length=max_len,
        min_length=min_len,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def bert_summarize(text, num_sentences):
    model = load_bert_model()
    # ratio=0.2 means it will pick top 20% of sentences, or you can use num_sentences
    result = model(text, num_sentences=num_sentences)
    return result

@st.cache_data
def get_text_from_url(url):
    try:
        # User-Agent header makes us look like a real browser, not a bot
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Check for errors (like 404)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from all <p> tags
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        return f"Error: {e}"

# --- UI Layout ---
st.title("📄 AI-Powered Text Summarizer")
st.markdown("Summarize long documents using **Abstractive (T5)** or **Extractive (BERT)** methods.")

# Sidebar for controls
st.sidebar.header("Settings")
model_type = st.sidebar.radio("Choose Model Type", ["Abstractive (T5)", "Extractive (BERT)"])

# Dynamic Sidebar Controls based on model choice
if model_type == "Abstractive (T5)":
    st.sidebar.info("T5 generates *new* sentences to summarize the meaning.")
    max_len = st.sidebar.slider("Max Summary Length", 50, 300, 150)
    min_len = st.sidebar.slider("Min Summary Length", 10, 100, 40)
else:
    st.sidebar.info("BERT extracts the *most important* sentences from the text.")
    num_sentences = st.sidebar.slider("Number of Sentences to Extract", 1, 10, 3)

# Main Input Area
# --- Main Input Area ---
tab1, tab2 = st.tabs(["✍️ Paste Text", "🔗 Input URL"])

final_text = ""

with tab1:
    text_input = st.text_area("Input Text", height=300, placeholder="Paste your article here...")
    if text_input:
        final_text = text_input

with tab2:
    url_input = st.text_input("Enter Article URL", placeholder="https://example.com/article")
    if url_input:
        with st.spinner("Fetching article content..."):
            fetched_text = get_text_from_url(url_input)
            if fetched_text.startswith("Error"):
                st.error(fetched_text)
            else:
                st.success("Content fetched successfully!")
                with st.expander("View Fetched Content"):
                    st.write(fetched_text[:1000] + "...") # Preview first 1000 chars
                final_text = fetched_text

# Process Button (Updated to use 'final_text')
if st.button("Generate Summary"):
    if not final_text:
        st.warning("Please provide text or a valid URL.")
    else:
        # ... The rest of your existing summarization logic goes here ...
        # Just ensure you pass 'final_text' to your summarize functions instead of 'text_input'
        with st.spinner('AI is processing...'):
             if model_type == "Abstractive (T5)":
                 summary = t5_summarize(final_text, max_len, min_len)
             else:
                 summary = bert_summarize(final_text, num_sentences)
             
             st.subheader("📝 Summary")
             st.success(summary)
# Process Button
if st.button("Generate Summary"):
    if not text_input:
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner('AI is processing...'):
            try:
                if model_type == "Abstractive (T5)":
                    summary = t5_summarize(text_input, max_len, min_len)
                else:
                    summary = bert_summarize(text_input, num_sentences)
                
                # Display Result
                st.subheader("📝 Summary")
                st.success(summary)
                
                # Optional: Statistics
                orig_len = len(text_input.split())
                summ_len = len(summary.split())
                st.caption(f"Original: {orig_len} words | Summary: {summ_len} words | Reduction: {round((1 - summ_len/orig_len)*100)}%")

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")

st.markdown("Built with Python, Streamlit, and Hugging Face Transformers.")


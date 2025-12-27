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
text_input = st.text_area("Input Text", height=300, placeholder="Paste your article or report here...")

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
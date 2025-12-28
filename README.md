# 📄 textsumt5: Abstractive Text Summarization with T5

**textsumt5** is a robust deep learning model designed to generate concise, abstractive summaries from long articles, documents, and text passages. Built on the **T5 (Text-to-Text Transfer Transformer)** architecture, this project leverages the power of transfer learning to understand context and rephrase information rather than just extracting sentences.

## 🚀 Features

* **Abstractive Summarization:** Generates new sentences to capture the core meaning, mimicking human-like summarization.
* **State-of-the-Art Architecture:** Utilizes Google's T5 model, fine-tuned for sequence-to-sequence tasks.
* **Customizable Length:** Users can define `min_length` and `max_length` for the output summary.
* **Easy Integration:** Simple Python API for integrating into web apps or other pipelines.
* **Beam Search Decoding:** Implements beam search to ensure high-quality and coherent output sequences.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Model Architecture:** T5 (Text-to-Text Transfer Transformer)
* **Libraries:**
* `transformers` (Hugging Face)
* `torch` (PyTorch)
* `sentencepiece` (Tokenizer)
* `pandas` (Data handling)



---

## ⚙️ Installation

1. **Clone the Repository**
```bash
git clone https://github.com/[YourUsername]/textsumt5.git
cd textsumt5

```


2. **Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

```


3. **Install Dependencies**
```bash
pip install -r requirements.txt

```



---

## 💻 Usage

You can use the model directly via the inference script. Here is a quick example:

### Quick Start

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 1. Load the pre-trained model and tokenizer
model_name = "t5-small"  # Or your fine-tuned path
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 2. Prepare the text
text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. 
Leading AI textbooks define the field as the study of "intelligent agents": any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
"""

# T5 specific prefix for summarization
preprocess_text = "summarize: " + text

# 3. Tokenize
tokenized_text = tokenizer.encode(preprocess_text, return_tensors="pt")

# 4. Generate Summary
summary_ids = model.generate(
    tokenized_text,
    num_beams=4,
    no_repeat_ngram_size=2,
    min_length=30,
    max_length=100,
    early_stopping=True
)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Original Text:", text)
print("\nSummary:", output)

```

---

## 🧠 Model Architecture

The **T5 model** treats every NLP problem as a "text-to-text" problem.

* **Encoder:** Processes the input text and understands the context.
* **Decoder:** Generates the summary token by token.

Unlike BERT (which is encoder-only), T5 is an Encoder-Decoder model, making it perfect for generation tasks like summarization and translation.

---

## 📊 Dataset & Training

This model was fine-tuned on the **[CNN/DailyMail or XSum]** dataset containing over 300k news articles.

* **Optimizer:** AdamW
* **Learning Rate:** 2e-5
* **Batch Size:** 8
* **Epochs:** 3

---

## 🔮 Future Scope

* **Multilingual Support:** Extending the model to summarize text in languages other than English using mT5.
* **UI Implementation:** Building a Streamlit or Flask web interface for easy user interaction.
* **Long Document Handling:** Implementing sliding window techniques for documents longer than 512 tokens.

---



## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---------------------------------------------------------------------------------

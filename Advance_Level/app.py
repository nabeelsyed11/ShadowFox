import streamlit as st
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, pipeline
import torch
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="LLM Playground", page_icon="🧠")
st.title("🧠 LLM Playground - BERT / DistilBERT / GPT-2")

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(name: str):
    if "gpt2" in name.lower():
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForCausalLM.from_pretrained(name)
    else:
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForMaskedLM.from_pretrained(name)
    return tok, mdl

# Model Selection (added tiny models for Streamlit Cloud testing)
model_choice = st.selectbox(
    "Choose a Language Model",
    ["distilbert-base-uncased", "bert-base-uncased", "gpt2", 
     "sshleifer/tiny-distilbert-base-uncased", "sshleifer/tiny-gpt2"]
)
tokenizer, model = load_model_and_tokenizer(model_choice)

st.subheader("🔹 Try it Out")

if "gpt2" in model_choice:
    input_text = st.text_input("Enter a prompt", "Artificial Intelligence will")
    if st.button("Generate Text"):
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        text = generator(input_text, max_length=40, num_return_sequences=1)[0]['generated_text']
        st.write("**Generated Text:**")
        st.success(text)
else:
    input_text = st.text_input("Enter a masked sentence", "The capital of France is [MASK].")
    if st.button("Fill Mask"):
        fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
        result = fill_mask(input_text)
        st.write("**Predictions:**")
        for r in result:
            st.write(f"- {r['token_str']} (score: {r['score']:.4f})")

st.subheader("📊 Pseudo-Perplexity Evaluation")
sentence = st.text_input("Enter a sentence", "Natural language processing enables AI to understand humans.")

def pseudo_perplexity(sentence, model, tokenizer):
    encodings = tokenizer(sentence, return_tensors="pt")
    input_ids = encodings.input_ids
    nlls = []
    for i in range(1, input_ids.size(1) - 1):
        input_ids_masked = input_ids.clone()
        if tokenizer.mask_token_id:
            input_ids_masked[0, i] = tokenizer.mask_token_id
        else:
            return None
        with torch.no_grad():
            outputs = model(input_ids_masked)
            logits = outputs.logits
        log_prob = torch.log_softmax(logits[0, i], dim=-1)
        nll = -log_prob[input_ids[0, i]]
        nlls.append(nll.item())
    return np.exp(np.mean(nlls))

if st.button("Evaluate Perplexity"):
    ppl = pseudo_perplexity(sentence, model, tokenizer)
    if ppl:
        st.write(f"**Pseudo-Perplexity:** {ppl:.2f}")
    else:
        st.warning("Perplexity not supported for causal LM like GPT-2.")

st.subheader("📈 Model Comparison (Example Scores)")
labels = ["DistilBERT", "BERT", "GPT-2"]
scores = [0.85, 0.89, 0.92]

fig, ax = plt.subplots()
ax.bar(labels, scores, color=['skyblue', 'orange', 'green'])
ax.set_ylabel("Performance Score")
ax.set_title("Comparison of Language Models")
st.pyplot(fig)

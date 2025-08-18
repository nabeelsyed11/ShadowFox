# 🧠 LLM Playground - NLP & ML Project  

This project explores **Language Models (LMs)** in Natural Language Processing (NLP).  
It provides a **Jupyter Notebook** for experimentation and a **Streamlit frontend** for interactive testing.  

## 📌 Features
- **Language Model Selection** → DistilBERT, BERT, GPT-2  
- **Text Classification / Cloze Task** → Fill-in-the-blank using Masked LMs  
- **Text Generation** → Creative sentence generation with GPT-2  
- **Pseudo-Perplexity Evaluation** → Estimate fluency of generated sentences  
- **Visualization** → Compare model performance with bar charts  
- **Research Insights** → Strengths, weaknesses, and future applications  

---

## 📂 Project Structure
LLM_Project/
│── LLM_Project.ipynb # Jupyter notebook with full workflow
│── app.py # Streamlit frontend for interactive testing
│── requirements.txt # Dependencies
│── README.md # Project documentation

---

## ⚙️ Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/yourusername/LLM_Project.git
cd LLM_Project
pip install -r requirements.txt
```
📓 Jupyter Notebook

Run the notebook to explore:
jupyter notebook LLM_Project.ipynb

streamlit run app.py

Features:

Select LM (DistilBERT / BERT / GPT-2)

Try text completion and generation

Evaluate pseudo-perplexity

See visual comparisons

📊 Example Outputs
Masked Language Modeling

Input:
The capital of France is [MASK].
Output Predictions:

Paris (0.98)

Lyon (0.01)

GPT-2 Text Generation

Prompt:
Artificial Intelligence will
Generated:
Artificial Intelligence will revolutionize industries and transform the way humans interact with technology.

❓ Research Questions

How well do LMs understand context?

Can GPT-2 maintain coherence in long text?

How do DistilBERT and BERT compare in accuracy for cloze tasks?

What are the limitations of pseudo-perplexity in evaluating fluency?

✅ Conclusion

This project demonstrates the capabilities and limitations of modern Language Models.

DistilBERT → Lightweight, fast, but less accurate.

BERT → Strong contextual understanding.

GPT-2 → Excellent text generation but weaker at cloze tasks.

Future work: fine-tuning on domain-specific datasets, exploring larger LMs like GPT-3 or LLaMA, and integrating explainability tools.

👨‍💻 Author

Developed by [Syed Nabeel Ahmed] as part of an NLP & ML exploration project.

---

👉 Do you also want me to generate a **downloadable `README.md` file** (so you don’t have to copy-paste)?

---

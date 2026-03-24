!pip install -q transformers==4.36.2 torch gradio scikit-learn keybert sentence-transformers

from transformers import pipeline
from keybert import KeyBERT
import gradio as gr

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering")
kw_model = KeyBERT()

def extract_keywords(text):
    keywords = kw_model.extract_keywords(text, top_n=5)
    return ", ".join([kw[0] for kw in keywords])

def get_key_points(text):
    summary = summarizer(text[:1000], max_length=120, min_length=40)[0]['summary_text']
    points = summary.split(". ")
    return "\n".join([f"• {p}" for p in points if len(p) > 5])

def answer_question(context, question):
    result = qa_pipeline(question=question, context=context[:2000])
    return result['answer']

def analyze(text):
    if len(text) < 50:
        return "⚠️ Enter valid blog"

    keywords = extract_keywords(text)
    key_points = get_key_points(text)

    return f"🔑 Keywords:\n{keywords}\n\n📌 Key Points:\n{key_points}"

with gr.Blocks() as app:
    gr.Markdown("## Smart Blog Analyzer")

    blog = gr.Textbox(lines=10, placeholder="Paste blog here...")

    analyze_btn = gr.Button("Analyze")
    output = gr.Textbox(lines=10)

    gr.Markdown("### Ask Question")
    q = gr.Textbox()
    ans_btn = gr.Button("Answer")
    ans = gr.Textbox()

    analyze_btn.click(analyze, inputs=blog, outputs=output)
    ans_btn.click(answer_question, inputs=[blog, q], outputs=ans)

app.launch(share=True)

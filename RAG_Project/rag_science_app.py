
import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from groq import Groq
import openai
import re
from bert_score import score as bert_score

# Set API Keys
os.environ["GROQ_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]
DOCUMENT_PATH = "/content/drive/MyDrive/documents"
client = Groq(api_key=os.environ["GROQ_API_KEY"])

st.set_page_config(page_title="High School Science RAG", layout="wide")
st.markdown("<h1 style='text-align: center;'>High School Science RAG App</h1>", unsafe_allow_html=True)
st.markdown("---")

query_input = st.text_area("Ask your science question:", height=100)

@st.cache_data
def load_documents(folder_path=DOCUMENT_PATH):
    docs, titles = [], []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))
            for page in reader.pages:
                text = page.extract_text()
                if not text:
                    continue
                paragraphs = [re.sub(r"\s+", " ", p.strip()) for p in text.split("\n\n") if len(p.strip()) > 100]
                docs.extend(paragraphs)
                titles.extend([file] * len(paragraphs))
    return docs, titles

docs, titles = load_documents()

@st.cache_resource
def embed_and_index(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))
    return model, index, embeddings

embedder, index, all_embeddings = embed_and_index(docs)

FEW_SHOTS = [
    {
        "context": "Photosynthesis is the process by which green plants convert sunlight into chemical energy...",
        "question": "What is photosynthesis?",
        "answer": "Photosynthesis is the process by which green plants use sunlight to make food."
    },
    {
        "context": "Newton's first law states that an object will remain at rest or in uniform motion...",
        "question": "State Newton's first law.",
        "answer": "An object at rest stays at rest, and an object in motion stays in motion unless acted upon by a force."
    }
]

def build_prompt(query, retrieved_chunks):
    few_shots = "\n\n".join([
        f"Context:\n{ex['context']}\nQ: {ex['question']}\nA: {ex['answer']}" for ex in FEW_SHOTS
    ])
    context = "\n\n".join(retrieved_chunks[:2])
    prompt = (
        "You are a high school science tutor. Only use the information in the context below "
        "to answer the question accurately and concisely.\n\n"
        f"{few_shots}\n\nContext:\n{context}\nQ: {query}\nA:"
    )
    return prompt

def retrieve_chunks(query, k=3):
    q_embed = embedder.encode([query], normalize_embeddings=True)
    _, indices = index.search(np.array(q_embed), k)
    return [docs[i] for i in indices[0]]

def generate_answer(query, k=3, model="llama3-70b-8192"):
    retrieved = retrieve_chunks(query, k)
    prompt = build_prompt(query, retrieved)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

evaluation_dataset = [
    { "query": "What is the difference between prokaryotic and eukaryotic cells?",
      "ground_truth": "Prokaryotic cells lack a true nucleus and other membrane-bound organelles, whereas eukaryotic cells possess a defined nucleus" },
    { "query": "Where in the plant cell does photosynthesis occur, and what is the role of chlorophyll?",
      "ground_truth": "Photosynthesis occurs in the chloroplasts, where chlorophyll absorbs light energy to power the process." },
    { "query": "State Newton's three laws of motion.",
      "ground_truth": "1st: An object remains at rest or in uniform motion unless acted on. 2nd: F = ma. 3rd: For every action, there is an equal and opposite reaction." },
    { "query": "How are elements arranged in the periodic table?",
      "ground_truth": "Elements in the periodic table are arranged by increasing atomic number." },
    { "query": "Describe the different states of matter.",
      "ground_truth": "Matter exists in four primary states: solid, liquid, gas, and plasma. Solids have a fixed shape and volume due to tightly packed particles. Liquids have a definite volume but take the shape of their container. Gases have neither a fixed shape nor volume, expanding to fill any space. Plasma, found in stars, is an ionized state of matter." },
    { "query": "Explain Ohm's Law.",
      "ground_truth": "Ohm's Law (V = IR) describes the relationship between voltage, current, and resistance." },
    { "query": "What is the main function of the small intestine?",
      "ground_truth": "The small intestine uses villi to absorb nutrients." },
    { "query": "What is the role of decomposers in an ecosystem?",
      "ground_truth": "The role of Decomposers is to break down dead matter, like fungi." },
    { "query": "How does litmus paper indicate pH?",
      "ground_truth": "The color of Litmus paper changes color on pH." },
    { "query": "Which planets are classified as gas giants?",
      "ground_truth": "Jupiter and Saturn are classified as gas giants." }
]

def normalize(text):
    return re.sub(r'[^a-z0-9]', ' ', text.lower()).strip()

def compute_metrics(preds, labels, queries):
    preds_clean = [normalize(p) for p in preds]
    labels_clean = [normalize(l) for l in labels]

    f1s, precisions, recalls = [], [], []

    for pred, label in zip(preds_clean, labels_clean):
        pred_tokens = pred.split()
        label_tokens = label.split()
        common = set(pred_tokens) & set(label_tokens)

        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(label_tokens) if label_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    P, R, F1 = bert_score(preds, labels, lang='en', verbose=False)

    return {
        "Precision": round(np.mean(precisions), 3),
        "Recall": round(np.mean(recalls), 3),
        "F1-Score": round(np.mean(f1s), 3),
        "BERTScore F1": round(float(F1.mean()), 3),
    }

def evaluate_with_groq_judge(query, gen_ans, ref_ans, judge_model="llama3-8b-8192"):
    from groq import Groq
    client_judge = Groq(api_key=os.environ["GROQ_API_KEY"])

    eval_prompt = f'''
You are a strict high school science teacher. Evaluate the student\'s answer to a science question.

### Question:
{query}

### Reference Answer:
{ref_ans}

### Student's Answer:
{gen_ans}

### Task:
Rate the student\'s answer from 1 to 5:
- 5 = Perfectly correct and complete
- 4 = Mostly correct, minor omissions
- 3 = Partially correct, missing key info or has small mistakes
- 2 = Mostly incorrect or vague
- 1 = Completely incorrect or irrelevant

Respond in this format:
Score: <number>
Reason: <your explanation>
'''

    try:
        response = client_judge.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error from Groq judge: {str(e)}"


# Display section
if query_input:
    st.markdown("### RAG Answer")
    with st.spinner("Generating answer..."):
        response = generate_answer(query_input)
    st.text_area("Answer:", response, height=200)

if st.button("Run Evaluation"):
    st.subheader("Evaluation Metrics")
    with st.spinner("Evaluating model..."):
        queries = [item["query"] for item in evaluation_dataset]
        labels = [item["ground_truth"] for item in evaluation_dataset]
        preds = [generate_answer(q) for q in queries]
        scores = compute_metrics(preds, labels, queries)

        judge_scores = []
        for query, pred, label in zip(queries, preds, labels):
            judge_output = evaluate_with_groq_judge(query, pred, label)

            match = re.search(r"Score:\s*(\d)", judge_output)
            if match:
                score = int(match.group(1))
                judge_scores.append(score)
            else:
                judge_scores.append(0)  # fallback if score missing

        if judge_scores:
            llm_judge_avg = round(np.mean(judge_scores), 3)
        else:
            llm_judge_avg = 0.0

        scores["LLM-Judge Score"] = round(llm_judge_avg / 5, 3)
        st.write(scores)


import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from groq import Groq
import re
from bert_score import score as bert_score

DOCUMENT_PATH = "RAG_Project/Documents"
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.set_page_config(page_title="High School Science RAG", layout="wide")
st.markdown("<h1 style='text-align: center;'>High School Science RAG App</h1>", unsafe_allow_html=True)
st.markdown("---")

query_input = st.text_input("Ask your science question:")

@st.cache_data
def load_documents(folder_path=DOCUMENT_PATH):
    docs, titles = [], []
    pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]
    for file in pdf_files:
        reader = PdfReader(os.path.join(folder_path, file))
        for page in reader.pages:
            text = page.extract_text()
            if not text:
                continue
            paragraphs = [
                re.sub(r"\s+", " ", p.strip())
                for p in text.split("\n\n")
                if len(p.strip()) > 100
            ]
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

def generate_answer(query, k=3, model="llama-3.1-8b-instant"):
    retrieved = retrieve_chunks(query, k)
    prompt = build_prompt(query, retrieved)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

evaluation_dataset = [
    { "query": "What is hypermetropia and how does it differ from myopia?",
      "ground_truth": "Hypermetropia, or farsightedness, is an eye defect where a person cannot see nearby objects clearly, though their distant vision is clear. Myopia, or nearsightedness, is the opposite defect, where a person cannot see distant objects clearly." },
    { "query": "What is the heating effect of electric current, also known as Joule's Law of Heating?",
      "ground_truth": "It is the phenomenon where the dissipation of energy in a purely resistive circuit occurs entirely in the form of heat. The heat produced (H) is given by the formula H = I²Rt." },
    { "query": "What is the difference between reactants and products in a chemical reaction?",
      "ground_truth": "Reactants are the substances that exist before a chemical reaction begins and are written on the left side of a chemical equation. Products are the new substances that are formed as a result of the reaction and are written on the right side of the equation." },
    { "query": "Write the overall balanced chemical equation for photosynthesis.",
      "ground_truth": "6CO2 + 6H2 O + Light → C6 H12 O6 + 6O2" },
    { "query": "Distinguish between a concave mirror and a convex mirror based on their reflecting surfaces.",
      "ground_truth": "A concave mirror is a spherical mirror whose reflecting surface is curved inwards, towards the center of the sphere. A convex mirror is a spherical mirror whose reflecting surface is curved outwards, away from the center of the sphere." },
    { "query": "What is a solenoid, and what is the nature of the magnetic field inside it?",
      "ground_truth": "A solenoid is a coil comprising several circular turns of insulated copper wire wrapped tightly in the shape of a cylinder. The magnetic field inside the solenoid is uniform." },
    { "query": "What was the basis for Dobereiner's classification of elements into triads?",
      "ground_truth": "Dobereiner used the physical and chemical characteristics of each element to divide them into triads. In a trio, the elements were organized in ascending order of their atomic masses, and he grouped elements with related qualities." },
    { "query": "What is the typical phenotypic ratio of a dihybrid cross in the F2 generation, according to Mendel?",
      "ground_truth": "The typical phenotypic ratio of a dihybrid cross in the F2 generation, according to Mendel, is 9:3:3:1." },
    { "query": "How does litmus paper indicate pH?",
      "ground_truth": "The color of Litmus paper changes color on pH." },
    { "query": "List two physical properties that are characteristic of metals.",
      "ground_truth": "Two physical properties that are characteristic of metals are:1. Luster: Metals have a shiny appearance.2. Conductivity: They conduct heat and electricity." }
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

def evaluate_with_groq_judge(query, gen_ans, ref_ans, judge_model="llama-3.3-70b-versatile"):
    from groq import Groq
    client_judge = Groq(api_key=st.secrets["GROQ_API_KEY"])

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


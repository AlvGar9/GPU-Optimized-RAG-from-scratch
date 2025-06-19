from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import cupy as cp
from transformers import AutoTokenizer, AutoModel, pipeline
from src.data_processing.distances import custom_topk

app = FastAPI()

class Query(BaseModel):
    query: str
    k: int = 2

# Load models
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")
embed_model = AutoModel.from_pretrained("intfloat/multilingual-e5-large-instruct")
chat = pipeline("text-generation", model="facebook/opt-125m")

docs = ["Cats...", "Dogs...", "Hummingbirds..."]
doc_embs = np.vstack([_ for _ in map(lambda d: get_embedding(d), docs)])

def get_embedding(text: str) -> np.ndarray:
    inp = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = embed_model(**inp).last_hidden_state.mean(1).cpu().numpy()
    return out

@app.post("/rag")
def predict(payload: Query):
    # embed query
    q_emb = get_embedding(payload.query)
    # retrieve
    sims = cp.asarray(doc_embs) @ cp.asarray(q_emb).T
    _, idx = custom_topk(sims, payload.k, largest=True)
    chosen = [docs[i] for i in idx.get()]
    prompt = f"Q: {payload.query}\n" + "\n".join(chosen) + "\nA:"
    return {"answer": chat(prompt, max_length=50)[0]["generated_text"]}

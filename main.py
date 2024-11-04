from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Path, HTTPException
import os
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize

load_dotenv()
uri = os.getenv("MONGODB_URI")
try:
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client.ica_conf
    papers_collection = db['papers']
    embeddings_collection = db['embeddings']
except Exception as e:
    print("Error connecting to MongoDB:", e)
    raise 

# Load the embeddings directly from MongoDB
print("Loading embeddings from MongoDB...")
embedding_docs = list(embeddings_collection.find({}, {"_id": 0, "paper_id": 1, "embedding": 1}))
paper_embeddings = {doc["paper_id"]: doc["embedding"] for doc in embedding_docs}
print("Loading embeddings finished!")

# Convert embeddings to NumPy array for FAISS
embeddings = np.array(list(paper_embeddings.values()), dtype=np.float32)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

with open("pca_model.pkl", "rb") as f:
    pca = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",          
        "https://ica-conf.onrender.com",  
        "https://icaconf.vercel.app",
        "https://ica-conference-app-hongtaohs-projects.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to ICA Conf Data Search"}

@app.get("/search")
async def search_papers(query: str, k: int = 5):
    try:
        # Encode the query and search for similar embeddings
        query_embedding = model.encode(query).reshape(1, -1)
        query_embedding = pca.transform(query_embedding) 
        query_embedding = normalize(query_embedding, norm='l2')
        distances, indices = index.search(query_embedding.reshape(1, -1), k)
        
        # Retrieve the top k paper IDs from MongoDB
        top_k_paper_ids = [list(paper_embeddings.keys())[i] for i in indices[0]]
        
        # Fetch paper details from MongoDB
        papers = list(papers_collection.find({"paper_id": {"$in": top_k_paper_ids}}, {"_id": 0}))
        return papers

    except Exception as e:
        print(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Path, HTTPException
import logging
from typing import List, Optional
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random 
import json 

import numpy as np
import gridfs
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()
uri = os.getenv("MONGODB_URI")
try:
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client.ica_conf
    # papers_collection = db['papers']
except Exception as e:
    print("Error connecting to MongoDB:", e)
    raise 

fs = gridfs.GridFS(db, collection="paper_embeddings_fs")

# Load the embeddings from GridFS
print("Loading embeddings from MongoDB...")
file_id = fs.find_one({"filename": "paper_embeddings.json"})._id 
paper_embeddings_data = fs.get(file_id).read().decode("utf-8")  # Load and decode to string
paper_embeddings = json.loads(paper_embeddings_data)  # Convert JSON string to dictionary
print("Loding embeddings finished!")

# Convert embeddings to NumPy array for FAISS
embeddings = np.array(list(paper_embeddings.values()), dtype=np.float32)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

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

# Define a response model for type-checking and documentation
class Authorship(BaseModel):
    position: Optional[int] = None
    author_name: Optional[str] = None
    author_affiliation: Optional[str] = None

class SessionInfo(BaseModel):
    session: str
    session_type: Optional[str] = None
    chair_name: Optional[str] = None
    chair_affiliation: Optional[str] = None
    division: Optional[str] = None
    years: List[int] = []
    paper_count: Optional[int] = None
    session_id: Optional[str] = None

class Paper(BaseModel):
    paper_id: str
    title: str
    paper_type: str
    abstract: Optional[str] = None
    number_of_authors: int
    year: int
    session: Optional[str] = None
    division: Optional[str] = None
    authorships: Optional[List[Authorship]] = None
    author_names: Optional[List[str]] = None
    session_info: Optional[SessionInfo] = None 

@app.get("/")
async def root():
    return {"message": "Welcome to ICA Conf Data Search"}

@app.get("/search")
async def search_papers(query: str, k: int = 5):
    try:
        # Encode the query and search for similar embeddings
        query_embedding = model.encode(query)
        distances, indices = index.search(query_embedding.reshape(1, -1), k)
        
        # Retrieve the top k paper IDs from MongoDB
        top_k_paper_ids = [list(paper_embeddings.keys())[i] for i in indices[0]]

        return top_k_paper_ids
        
        # # Fetch paper details from MongoDB
        # papers = list(papers_collection.find({"paper_id": {"$in": top_k_paper_ids}}, {"_id": 0}))
        # return papers

    except Exception as e:
        print(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


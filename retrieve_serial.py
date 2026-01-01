import json, time
from pydantic import BaseModel
import numpy as np
import argparse
from tqdm import tqdm
import warnings
from transformers import AutoTokenizer, AutoModel
import torch
from datasets import load_dataset
from typing import List
import os
import uvicorn
import pickle
import src.index
from src.index import index_encoded_data
from openai import OpenAI, AsyncOpenAI, OpenAIError
import asyncio
import openai
import re
from rank_bm25 import BM25Okapi
from utils import bm25_utils
from fastapi import FastAPI
import tiktoken
from loguru import logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoding = tiktoken.get_encoding("cl100k_base")

os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_BASE_URL'] = 'https://api.openai.com/v1'

OPENAI_CLIENT = OpenAI()
OPENAI_CLIENT.api_key = os.getenv('OPENAI_API_KEY')

EMBEDDING_MODELS = {
    "contriever-msmarco":{
        "type": "hf",
        "path": "",
        "projection_size": 768,
    },
    "text-embedding-3-small":{
        "type": "openai",
        "model": "text-embedding-3-small",
        "projection_size": 1536,
    },
    "text-embedding-3-large":{
        "type": "openai",
        "model": "text-embedding-3-large",
        "projection_size": 3072,
    },
    "text-embedding-ada-002":{
        "type": "openai",
        "model": "text-embedding-ada-002",
        "projection_size": 1536,
    }
}

HF_MODELS = {
    k: {
        "model": AutoModel.from_pretrained(EMBEDDING_MODELS[k]["path"]).to(DEVICE),
        "tokenizer": AutoTokenizer.from_pretrained(EMBEDDING_MODELS[k]["path"]),
    } for k in EMBEDDING_MODELS.keys() if EMBEDDING_MODELS[k]["type"] == "hf"
}



def truncate_text_tokens(text, max_tokens=8000):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    trunc_encode = encoding.encode(text)[:max_tokens]
    new_text = encoding.decode(trunc_encode)
    return new_text


def hf_retriever_encode(model_name, texts: List[str], batch_size: int = 32) -> List[np.array]:
    model = HF_MODELS[model_name]["model"]
    tokenizer = HF_MODELS[model_name]["tokenizer"]

    batches = []
    embedding_list = []

    for i in range(0, len(texts), batch_size):
        batches.append(texts[i:i+batch_size])

    for batch in batches:
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        
        embeddings = model(**inputs)
        embedding_list.append(embeddings[0][:, 0, :].detach().cpu().numpy())

    return np.concatenate(embedding_list, axis=0)

def gpt_retriever_encode(model_name, texts):
    embeddings = []
    batch_size = 2000 # https://github.com/openai/openai-python/issues/519
    text_batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    for text_batch in text_batches:
        for _ in range(5):
            try:
                responses = OPENAI_CLIENT.embeddings.create(input=text_batch, model=EMBEDDING_MODELS[model_name]["model"]).data
                break
            except openai.BadRequestError as e:
                print(f"BadRequestError: {e}")
                return None
            except openai.RateLimitError as e:
                print(f"RateLimitError: {e}")
                time.sleep(10)
        
        for response in responses:
            embeddings.append(response.embedding)
    
    return np.array(embeddings)

def get_retrieved_documents(
    query_text_list, 
    documents, 
    model_name=list(EMBEDDING_MODELS.keys())[0], 
    top_k=10, 
    n_subquantizers=0,
    n_bits=8, 
    indexing_batch_size=1000000, 
):
    top_k_results = []
    model_name = model_name.split("/")[-1]
        
    if model_name == "bm25":
        for i in tqdm(range(len(query_text_list))):
            # load context
            context_list = documents
            context_list = [bm25_utils.process_text(context) for context in context_list]
            
            bm25 = BM25Okapi(context_list)
            
            doc_scores = bm25.get_scores(query_text_list[i])
            
            # Get indices of top k elements
            top_k_indices = np.argsort(doc_scores)[::-1][:top_k] if top_k != -1 else np.argsort(doc_scores)[::-1]

            # Get values corresponding to top k indices
            top_k_values = doc_scores[top_k_indices]
            
            top_k_results.append((top_k_indices, top_k_values))
    else:
        # Process context_list here first
        context_list = [truncate_text_tokens(i) if i.strip() else "empty string" for i in documents]
        
        # embed query
        if EMBEDDING_MODELS[model_name]["type"] == "openai":
            encode_fn = gpt_retriever_encode
        else:
            encode_fn = hf_retriever_encode

        projection_size = EMBEDDING_MODELS[model_name]["projection_size"]
        query_embeddings = encode_fn(model_name, query_text_list)


        # create index
        index = src.index.Indexer(projection_size, n_subquantizers, n_bits)

        # index context
        context_embeddings = encode_fn(model_name, context_list)

        assert len(context_list) == len(context_embeddings)
        ids = list(range(len(context_list)))

        index_encoded_data(index, ids, context_embeddings, indexing_batch_size)

        k = len(context_list) if top_k == -1 else top_k

        for i in tqdm(range(len(query_text_list))):
            # search top k documents for each query
            top_k_documents_and_scores = index.search_knn(query_embeddings[i].reshape(1, -1), k)
            
            top_k_results.append(top_k_documents_and_scores[0])

    # store retrieved information
    output_data = []
    for i in range(len(query_text_list)):
        retrieved_documents = [
            { 
                "page": int(document_id) + 1, 
                "score": float(score) 
            }
            for document_id, score in zip(*top_k_results[i])
        ]
        output_data.append(retrieved_documents)
    return output_data[0]

if __name__ == "__main__":

    model_name = "bm25"
    with open("./data/test.json", "r") as f:
        data = json.load(f)

    retrieved_results = []
    for example in tqdm(data):
        question_id = example["question_id"]
        question = example["question"]
        texts_path = example["texts"]
        with open(texts_path, "r") as f:
            texts = json.load(f)
        results = get_retrieved_documents(query_text_list=[question], documents=texts, top_k=-1, model_name=model_name)
        retrieved_results.append({
            "question_id": question_id,
            "question": question,
            "retrieved_results": results
        })

    with open(f"./retrieved_results/{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(retrieved_results, f, indent=4, ensure_ascii=False)



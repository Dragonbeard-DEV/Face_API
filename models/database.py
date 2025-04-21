import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

index_map = {}
name_map = {}
vector_map = {}

def get_index(group):
    if group not in index_map:
        index_map[group] = []
        name_map[group] = []
        vector_map[group] = []
    return index_map[group], name_map[group], vector_map[group]

def add_to_faiss(vector, name, group="default"):
    vec = vector / np.linalg.norm(vector)
    _, names, vectors = get_index(group)
    names.append(name)
    vectors.append(vec)

def search_in_faiss(vector, group="default"):
    vec = vector / np.linalg.norm(vector)
    _, names, vectors = get_index(group)
    if not vectors:
        return "Unknown", 0.0
    sims = cosine_similarity([vec], vectors)[0]
    best_idx = int(np.argmax(sims))
    return names[best_idx], float(sims[best_idx])

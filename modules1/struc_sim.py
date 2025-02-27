import math
from collections import defaultdict


def load_triples(file_path):
    """Construct neighboring dictionaries containing isolated nodes (automatically includes all entities)"""
    adj = defaultdict(set)
    entities = set()  # Record all occurrences of the entity
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            h = parts[0]
            t = parts[2]
            entities.update([h, t])
            adj[h].add(t)
            adj[t].add(h)
    # Ensure that isolated entities exist in the adjacency dictionary (empty neighbors)
    for e in entities:
        _ = adj[e]  # Trigger defaultdict to automatically create empty collection
    return adj


def load_ref_ent_ids(file_path):
    """Load alignments, no longer filtering any entities"""
    S_pairs = []
    with open(file_path, 'r') as f:
        for line in f:
            src, tgt = line.strip().split('\t')
            S_pairs.append((src, tgt))
    return S_pairs


def structure_similarity(kg1_adj, kg2_adj, S_pairs):
    """Improved structural similarity computation dealing with three neighbor cases"""
    if not S_pairs:
        return 0.0

    total_sim = 0.0
    m = len(S_pairs)
    S_src = [pair[0] for pair in S_pairs]
    S_tgt = [pair[1] for pair in S_pairs]

    for (i, j) in S_pairs:
        # Get the set of neighbors (all entities exist in the adjacency dictionary at this point)
        neighbors_i = kg1_adj[i]
        neighbors_j = kg2_adj[j]

        # Scenario 1: Neither party has a neighbor
        if not neighbors_i and not neighbors_j:
            sim = 1.0  # The structure is identical

        # Scenario 2: One of the parties has no neighbors
        elif not neighbors_i or not neighbors_j:
            sim = 0.0  # Complete mismatch in structure

        # Scenario 3: Neighbors on both sides
        else:
            # Constructing projection vectors based on alignment spaces
            v1 = [1 if x in neighbors_i else 0 for x in S_src]
            v2 = [1 if y in neighbors_j else 0 for y in S_tgt]

            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm_v1 = math.sqrt(sum(a ** 2 for a in v1))
            norm_v2 = math.sqrt(sum(b ** 2 for b in v2))

            sim = dot_product / (norm_v1 * norm_v2) if norm_v1 * norm_v2 != 0 else 0.0

        total_sim += sim

    return total_sim / m


if __name__ == '__main__':
    # Note: load_triples will now contain all entities
    kg1_adj = load_triples('triples_1')
    kg2_adj = load_triples('triples_2')
    aligned_pairs = load_ref_ent_ids('ref_ent_ids')

    similarity = structure_similarity(kg1_adj, kg2_adj, aligned_pairs)
    print(f"Structure Similarity: {similarity:.4f}")
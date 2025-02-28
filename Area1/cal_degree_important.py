import os
from collections import defaultdict
from tqdm import tqdm
import networkx as nx
import numpy as np
import shutil  # Import shutil for file copying

def s2_degree_and_s3_important(data_dir, is_activation_m1 = True, ablation_config = None):
    S2_S3_Agent_Profile = '''
    Goal: Filter candidate entity pairs for entities with key nodes and extreme informativeness.
    Constraint: Outputs pairs of key candidate entities that meet the requirements of the goal.
    '''
    S2_S3_PRIVATE_MESSAGE_POOL = {
        'top_k_candidate_entities': os.path.join(data_dir, "message_pool", "retriever_outputs.txt"),
        'metrics_kg1': os.path.join(data_dir, "message_pool", "metrics_kg1.txt"),
        'metrics_kg2': os.path.join(data_dir, "message_pool", "metrics_kg2.txt"),
        'important_entities': os.path.join(data_dir, "message_pool", "important_entities.txt"),
    }

    def load_kg_and_build_graph(file_path):
        edges = []
        degrees = defaultdict(int)
        total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))

        with tqdm(total=total_lines, desc=f"Loading {os.path.basename(file_path)}") as pbar:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        edges.append((parts[0], parts[2]))
                        degrees[parts[0]] += 1
                        degrees[parts[2]] += 1
                    pbar.update(1)

        G = nx.Graph(edges)
        return G, degrees

    def save_metrics(degrees, pagerank, filename):
        with open(filename, 'w') as f:
            f.write("node_id\tdegree\tpagerank\n")
            for node in degrees:
                f.write(f"{node}\t{degrees[node]}\t{pagerank.get(node, 0):.6f}\n")

    def filter_entities(degrees, pagerank, s1=3, s2=10, top_percent=0.1):
        # degree of filtration
        low_degree = {n for n, d in degrees.items() if d <= s1}
        high_degree = {n for n, d in degrees.items() if d >= s2}

        # PageRank filter top 10%
        pr_threshold = np.percentile(list(pagerank.values()), 90)
        high_pr = {n for n, pr in pagerank.items() if pr >= pr_threshold}

        return low_degree, high_degree, high_pr


    if not is_activation_m1:
        # Define source and destination paths
        os.makedirs(os.path.dirname(S2_S3_PRIVATE_MESSAGE_POOL['important_entities']), exist_ok=True)
        shutil.copy2(S2_S3_PRIVATE_MESSAGE_POOL['top_k_candidate_entities'],
                     S2_S3_PRIVATE_MESSAGE_POOL['important_entities'])
        matches = []

    else:
        print("\nProcessing knowledge graphs...")
        G1, degrees1 = load_kg_and_build_graph(os.path.join(data_dir, 'triples_1'))
        G2, degrees2 = load_kg_and_build_graph(os.path.join(data_dir, 'triples_2'))

        print("\nCalculating PageRank...")
        pr1 = nx.pagerank(G1)
        pr2 = nx.pagerank(G2)

        # print("\nCalculating Betweenness Centrality...")
        # pr1 = nx.betweenness_centrality(G1)
        # pr2 = nx.betweenness_centrality(G2)

        # print("\nCalculating Closeness Centrality...")
        # pr1 = nx.closeness_centrality(G1)
        # pr2 = nx.closeness_centrality(G2)

        # print("\nCalculating eigenvector_centrality...")
        # pr1 = nx.eigenvector_centrality(G1, max_iter=1000)
        # pr2 = nx.eigenvector_centrality(G2, max_iter=1000)

        # print("\nCalculating katz_centrality...")
        # pr1 = nx.katz_centrality(G1, alpha=0.1, beta=1.0)
        # pr2 = nx.katz_centrality(G2, alpha=0.1, beta=1.0)

        save_metrics(degrees1, pr1, S2_S3_PRIVATE_MESSAGE_POOL['metrics_kg1'])
        save_metrics(degrees2, pr2, S2_S3_PRIVATE_MESSAGE_POOL['metrics_kg2'])


        print("\nFiltering entities...")
        low_deg1, high_deg1, high_pr1 = filter_entities(degrees1, pr1)
        low_deg2, high_deg2, high_pr2 = filter_entities(degrees2, pr2)

        candidates1 = low_deg1 | high_deg1 | high_pr1
        candidates2 = low_deg2 | high_deg2 | high_pr2

        if ablation_config:
            if ablation_config[0] == 'ablation5' and ablation_config[1] == 'no_S2':
                candidates1 = low_deg1 | high_deg1
                candidates2 = low_deg2 | high_deg2
            if ablation_config[0] == 'ablation5' and ablation_config[1] == 'no_S3':
                candidates1 = high_pr1
                candidates2 = high_pr2


        print("\nMatching entities...")
        matches = []
        match_file = S2_S3_PRIVATE_MESSAGE_POOL['top_k_candidate_entities']
        total_lines = sum(1 for _ in open(match_file, 'r', encoding='utf-8'))

        with tqdm(total=total_lines, desc="Matching") as pbar:
            with open(match_file, 'r', encoding='utf-8') as f:
                for line in f:
                    e1, e2 = line.strip().split()
                    if e1 in candidates1 or e2 in candidates2:
                        matches.append((e1, e2))
                    pbar.update(1)

        os.makedirs(os.path.dirname(S2_S3_PRIVATE_MESSAGE_POOL['important_entities']), exist_ok=True)
        with open(S2_S3_PRIVATE_MESSAGE_POOL['important_entities'], 'w', encoding='utf-8') as f:
            for e1, e2 in matches:
                f.write(f"{e1}\t{e2}\n")

    return matches

if __name__ == "__main__":
    data_dir = "/root/autodl-tmp/AdaCoAgent/AdaCoAgent/data/icews_wiki"
    matched_pairs = s2_degree_and_s3_important(data_dir)
    print(f"Found {len(matched_pairs)} matched entity pairs")


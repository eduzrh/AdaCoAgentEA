import os
import queue
import threading

from tqdm import tqdm
import httpx
from openai import OpenAI
from collections import defaultdict
import json
import random
import sys

sys.path.append('/home/dex/Desktop/entity_sy/Aligning')
from ThreadPoolExecutor import ThreadPoolExecutor


import tokens_cal

def load_descriptions(file_path):
    """Load entity description information"""
    with open(file_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    return {int(k): v[0]['description'] if v else "" for k, v in descriptions.items()}

def get_entity_context_m3(entity_id, entity_names, descriptions):
    """Get descriptive information about the entity (used for from_m3=True)"""
    return f"Entity Name: {entity_names.get(entity_id, 'Unknown')}\nDescription:\n{descriptions.get(entity_id, 'No description available')}"

def get_random_rules(data_dir, n=5):
    """Randomly select n rules from the rules file"""
    LLM1_PRIVATE_MESSAGE_POOL = {
        'important_entities': os.path.join(data_dir, "message_pool", "important_entities.txt"),
        'ucon_similarity_results': os.path.join(data_dir, "message_pool", "ucon_similarity_results.txt"),
        'KG1_compared_description': os.path.join(data_dir, "message_pool", "KG1_compared_description.json"),
        'KG2_compared_description': os.path.join(data_dir, "message_pool", "KG2_compared_description.json"),
        'alignment_rules': os.path.join(data_dir, "message_pool", "alignment_rules.txt"),
        'aligned_entities': os.path.join(data_dir, "message_pool", "aligned_entities.txt"),
    }
    rules = []
    rules_file = LLM1_PRIVATE_MESSAGE_POOL['alignment_rules']
    if os.path.exists(rules_file):
        with open(rules_file, 'r', encoding='utf-8') as f:
            all_rules = f.readlines()
        rules = random.sample(all_rules, min(n, len(all_rules)))
    return rules


def load_entity_names(file_path):
    """Load mapping of entity ids and names"""
    entity_names = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity_names[int(parts[0])] = parts[1]
    return entity_names

def load_triples(file_path):
    """Load ternary data"""
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            triples.append([int(x) for x in parts[:3]])
    return triples

def get_entity_context(entity_id, entity_names, triples, rel_names, n=3):
    """Get the first n relationships of the entity"""
    relations = []
    for h, r, t in triples:
        if h == entity_id:
            rel_str = rel_names.get(r, str(r))
            tail_str = entity_names.get(t, str(t))
            relations.append(f"- Has relation '{rel_str}' with {tail_str}")
        elif t == entity_id:
            rel_str = rel_names.get(r, str(r))
            head_str = entity_names.get(h, str(h))
            relations.append(f"- Is {rel_str} of {head_str}")
        if len(relations) >= n:
            break

    context = f"Entity Name: {entity_names.get(entity_id, 'Unknown')}\n"
    context += "Relationships:\n" + "\n".join(relations[:n])
    return context

def group_candidates(input_file):
    """Group candidate entity pairs in the input file by KG1 entities"""
    groups = defaultdict(list)
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            e1, e2 = map(int, line.strip().split('\t'))
            groups[e1].append(e2)
    return groups

def align_entities(data_dir, from_m3 = False, ablation_config = None, no_optimization_tool = False):


    LLM1_PRIVATE_MESSAGE_POOL = {
        'important_entities': os.path.join(data_dir, "message_pool", "important_entities.txt"),
        'ucon_similarity_results': os.path.join(data_dir, "message_pool", "ucon_similarity_results.txt"),
        'KG1_compared_description': os.path.join(data_dir, "message_pool", "KG1_compared_description.json"),
        'KG2_compared_description': os.path.join(data_dir, "message_pool", "KG2_compared_description.json"),
        'alignment_rules': os.path.join(data_dir, "message_pool", "alignment_rules.txt"),
        'aligned_entities': os.path.join(data_dir, "message_pool", "aligned_entities.txt"),
    }

    LLM1_Agent_Profile = '''
Goal: As a knowledge graph alignment expert, determine if the first entity represents the same object as one of the entities in the candidate entity list.
Constraint: if there is a match, return only the ID of the matching candidate entity number; if none of them matches (is not the same object), return "No"; if none of them matches (is not the same object), return "No"; if none of them matches (is not the same object), return "No".
    '''

    input_file = LLM1_PRIVATE_MESSAGE_POOL['ucon_similarity_results'] if from_m3 else LLM1_PRIVATE_MESSAGE_POOL['important_entities']
    output_file = LLM1_PRIVATE_MESSAGE_POOL['aligned_entities']

    # Setting up the OpenAI client
    client = OpenAI(
        base_url="your_base_url",
        api_key="your_api_key",
        http_client=httpx.Client(
            base_url="https://hk.xty.app/v1",
            follow_redirects=True,
        ),
    )


    ent_names_1 = load_entity_names(os.path.join(data_dir, 'ent_ids_1'))
    ent_names_2 = load_entity_names(os.path.join(data_dir, 'ent_ids_2'))

    if from_m3:
        descriptions_1 = load_descriptions(LLM1_PRIVATE_MESSAGE_POOL['KG1_compared_description'])
        descriptions_2 = load_descriptions(LLM1_PRIVATE_MESSAGE_POOL['KG2_compared_description'])
        rules = get_random_rules(data_dir)
    else:
        rel_names_1 = load_entity_names(os.path.join(data_dir, 'rel_ids_1'))
        rel_names_2 = load_entity_names(os.path.join(data_dir, 'rel_ids_2'))
        triples_1 = load_triples(os.path.join(data_dir, 'triples_1'))
        triples_2 = load_triples(os.path.join(data_dir, 'triples_2'))


    # Grouping of candidate entities
    candidate_groups = group_candidates(input_file)
    aligned_pairs = []

    lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=30)
    # Create a queue to store writes to the file
    result_queue = queue.Queue()

    def openai_task(kg1_entity, kg2_candidates):
        try:
            if from_m3:
                context1 = get_entity_context_m3(kg1_entity, ent_names_1, descriptions_1)
                candidates_contexts = [
                    {
                        'entity_id': kg2_entity,
                        'context': get_entity_context_m3(kg2_entity, ent_names_2, descriptions_2)
                    }
                    for kg2_entity in kg2_candidates
                ]
            else:
                context1 = get_entity_context(kg1_entity, ent_names_1, triples_1, rel_names_1)

                # Get the context of all KG2 candidate entities
                candidates_contexts = []
                for kg2_entity in kg2_candidates:
                    context = get_entity_context(kg2_entity, ent_names_2, triples_2, rel_names_2)
                    candidates_contexts.append({
                        'entity_id': kg2_entity,
                        'context': context
                    })

            # Building the Prompt

            prompt = LLM1_Agent_Profile + f"""
                                Entity 1 (ID: {kg1_entity}):
                                {context1}

                                the candidate entity list:"""

            ablation_config_com = 0
            
            if ablation_config:
                if ablation_config[0] == 'ablation5' and ablation_config[1] == 'Communication':
                    prompt = f"""
                    Entity 1 (ID: {kg1_entity}):
                    {context1}

                    the candidate entity list:"""
                    ablation_config_com = 1

            if no_optimization_tool == True:
                prompt = f"""
                                    Entity 1 (ID: {kg1_entity}):
                                    {context1}

                                    the candidate entity list:"""
                ablation_config_com = 1

            for i, candidate in enumerate(candidates_contexts, 1):
                prompt += f"\n\ncandidate entity{i} (ID: {candidate['entity_id']}):\n{candidate['context']}"

            if ablation_config_com == 1:
                    prompt += """\n Output aligned entity pairs, as much as possible:"""

            if ablation_config_com == 0:
                prompt += """\n\nDo any of these candidate entities represent the same object as entity 1? If so, only the corresponding entity ID is returned; if none of them match (is not the same object), only "No" is returned; if none of them match (is not the same object), only "No" is returned; if none of them match (is not the same object), only "No" is returned:"""

            if from_m3 and rules:
                prompt += "\n\nReference Rules:\n" + "".join(rules)

            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{'role': 'user', 'content': prompt}]
            )

            answer = response.choices[0].message.content.strip()

            tokens_cal.update_add_var(response.usage.total_tokens)  # update tokens

            # The kg1_entity and kg2_candidates are printed here for each execution.
            print(f"Processing entity {kg1_entity} with candidates {kg2_candidates}")

            print(prompt, answer)
            # an analytic response
            if answer.lower() != "no":
                # Trying to extract the entity ID from the answer
                for kg2_id in kg2_candidates:
                    if str(kg2_id) in answer:
                        with lock:
                            result_queue.put((kg1_entity, kg2_id))
                            aligned_pairs.append((kg1_entity, kg2_id))
                        break

        except Exception as e:
            print(f"Error processing entity {kg1_entity}: {str(e)}")

    # Processing each group of candidate entities
    for kg1_entity_c, kg2_candidates_c in tqdm(candidate_groups.items()):
        executor.submit(openai_task,kg1_entity_c, kg2_candidates_c)

    # Save results
    # if os.path.exists(output_file):
    #     with open(output_file, 'a', encoding='utf-8') as f:
    #         for e1, e2 in aligned_pairs:
    #             f.write(f"{e1}\t{e2}\n")
    # else:
    #     with open(output_file, 'w', encoding='utf-8') as f:
    #         for e1, e2 in aligned_pairs:
    #             f.write(f"{e1}\t{e2}\n")

    executor.shutdown(wait=True)

    # Write the results from the queue to a file
    with open(output_file, 'a+', encoding='utf-8') as output_f:
        while not result_queue.empty():
            kg1_entity, kg2_id = result_queue.get()
            output_f.write(f"{kg1_entity}\t{kg2_id}\n")
            output_f.flush()  # Flush the buffer immediately to ensure that it is written to disk

    deduplicate_output_file(output_file)

    return aligned_pairs

def deduplicate_output_file(file_path):
    """De-duplication of the output file"""
    if not os.path.exists(file_path):
        return

    # Reads all rows and de-duplicates them
    unique_pairs = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            e1, e2 = map(int, line.strip().split('\t'))
            unique_pairs.add((e1, e2))

    # Rewrite the result after de-duplication
    with open(file_path, 'w', encoding='utf-8') as f:
        for e1, e2 in sorted(unique_pairs):  # Sorting to maintain stable output
            f.write(f"{e1}\t{e2}\n")

    print(f"Deduplicated file {file_path}: {len(unique_pairs)} unique pairs")

if __name__ == "__main__":
    data_dir = "/home/dex/Desktop/entity_sy/Aligning/data/icews_wiki/"
    aligned_pairs = align_entities(data_dir)
    print(f"Found {len(aligned_pairs)} aligned entity pairs.")
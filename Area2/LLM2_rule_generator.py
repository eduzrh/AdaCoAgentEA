import os
import queue
import threading
from concurrent.futures import as_completed

from tqdm import tqdm
import httpx
from openai import OpenAI
from collections import defaultdict
import random
import sys
sys.path.append('/home/dex/Desktop/entity_sy/AdaCoAgent')
from ThreadPoolExecutor import ThreadPoolExecutor

import tokens_cal

def load_entity_names(file_path):
    """Load entity ID to name mapping"""
    entity_names = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity_names[int(parts[0])] = parts[1]
    return entity_names


def load_triples(file_path):
    """Load triple data"""
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            triples.append([int(x) for x in parts[:3]])
    return triples


def load_true_pairs(file_path):
    """Load true aligned entity pairs"""
    pairs = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            e1, e2 = map(int, line.strip().split('\t'))
            pairs.add((e1, e2))
    return pairs


def get_negative_pairs(true_pairs, ent_names_1, ent_names_2, sample_size=100):
    """Generate negative pairs from entities that are not aligned"""
    kg1_entities = set(e1 for e1, _ in true_pairs)
    kg2_entities = set(e2 for _, e2 in true_pairs)

    negative_pairs = set()
    kg1_list = list(ent_names_1.keys())
    kg2_list = list(ent_names_2.keys())

    while len(negative_pairs) < sample_size:
        e1 = random.choice(kg1_list)
        e2 = random.choice(kg2_list)
        if (e1, e2) not in true_pairs:
            negative_pairs.add((e1, e2))

    return negative_pairs


def get_entity_context(entity_id, entity_names, triples, rel_names):
    """Get entity context with 3 random relations"""
    relations = []
    entity_triples = []

    for h, r, t in triples:
        if h == entity_id:
            entity_triples.append((h, r, t, True))  # True for head position
        elif t == entity_id:
            entity_triples.append((h, r, t, False))  # False for tail position

    if entity_triples:
        sampled_triples = random.sample(entity_triples, min(3, len(entity_triples)))
        for h, r, t, is_head in sampled_triples:
            rel_str = rel_names.get(r, str(r))
            if is_head:
                tail_str = entity_names.get(t, str(t))
                relations.append(f"Has relation '{rel_str}' with {tail_str}")
            else:
                head_str = entity_names.get(h, str(h))
                relations.append(f"Is {rel_str} of {head_str}")

    context = {
        "name": entity_names.get(entity_id, f"Entity_{entity_id}"),
        "relations": relations
    }
    return context


def run_full_process_llm2(data_dir, batch_size=500, ablation_config = None, no_optimization_tool = False):
    """Generate alignment rules using LLM analysis"""
    LLM2_PRIVATE_MESSAGE_POOL = {
        'alignment_rules': os.path.join(data_dir, "message_pool", "alignment_rules.txt"),
        'aligned_entities': os.path.join(data_dir, "message_pool", "aligned_entities.txt"),
        'aligned_entities_history': os.path.join(data_dir, "message_pool", "aligned_entities_history.txt"),
        'ucon_similarity_results': os.path.join(data_dir, "message_pool", "ucon_similarity_results.txt"),
    }

    # Read the last round of archived data
    previous_pairs = set()
    if os.path.exists(LLM2_PRIVATE_MESSAGE_POOL['aligned_entities_history']):
        previous_pairs = load_true_pairs(LLM2_PRIVATE_MESSAGE_POOL['aligned_entities_history'])

    true_pairs = load_true_pairs(LLM2_PRIVATE_MESSAGE_POOL['aligned_entities'])

    # Create an archive of the current data
    with open(LLM2_PRIVATE_MESSAGE_POOL['aligned_entities_history'], 'w', encoding='utf-8') as f:
        for e1, e2 in true_pairs:
            f.write(f"{e1}\t{e2}\n")

    # Filtering out added data
    new_pairs = true_pairs - previous_pairs
    if not new_pairs:
        print("No new entity pairs to process.") #Clear entity pairs that are not sure if they are aligned or not
        if os.path.exists(LLM2_PRIVATE_MESSAGE_POOL['ucon_similarity_results']):
            with open(LLM2_PRIVATE_MESSAGE_POOL['ucon_similarity_results'], 'w') as f:
                f.write('')
        return []

    # Initialize OpenAI client
    client = OpenAI(
        base_url="your_base_url",
        api_key="your_api_key",
        http_client=httpx.Client(
            base_url="https://hk.xty.app/v1",
            follow_redirects=True,
        ),
    )

    # Load data
    ent_names_1 = load_entity_names(os.path.join(data_dir, 'ent_ids_1'))
    ent_names_2 = load_entity_names(os.path.join(data_dir, 'ent_ids_2'))
    rel_names_1 = load_entity_names(os.path.join(data_dir, 'rel_ids_1'))
    rel_names_2 = load_entity_names(os.path.join(data_dir, 'rel_ids_2'))
    triples_1 = load_triples(os.path.join(data_dir, 'triples_1'))
    triples_2 = load_triples(os.path.join(data_dir, 'triples_2'))

    # Load true pairs and generate negative pairs

    negative_pairs = get_negative_pairs(new_pairs, ent_names_1, ent_names_2)

    all_rules = []

    executor = ThreadPoolExecutor(max_workers=10)

    def process_positive_pair(e1, e2):
        context1 = get_entity_context(e1, ent_names_1, triples_1, rel_names_1)
        context2 = get_entity_context(e2, ent_names_2, triples_2, rel_names_2)
        return {'type': 'positive', 'kg1': context1, 'kg2': context2}

    def process_negative_pair(e1, e2):
        context1 = get_entity_context(e1, ent_names_1, triples_1, rel_names_1)
        context2 = get_entity_context(e2, ent_names_2, triples_2, rel_names_2)
        return {'type': 'negative', 'kg1': context1, 'kg2': context2}

    def openai_task(batch_start,batch_index):
        batch_pairs = list(new_pairs)[batch_start:batch_start + batch_size]
        batch_examples = []

        # Displaying a progress bar with tqdm
        with ThreadPoolExecutor(max_workers=100) as executor_results:
            # Submit the task and return the Future object
            futures = [executor_results.submit(process_positive_pair, pair[0], pair[1]) for pair in batch_pairs]

            # Manually updating the progress bar
            with tqdm(total=len(batch_pairs), desc=f"Processing Positive Pairs - Batch {batch_index}") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    batch_examples.append(result)
                    pbar.update(1)  # Updates the progress bar every time a task is completed

        # Process negative pairs with multi-threading
        negative_batch = list(negative_pairs)[batch_start:batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=30) as executor_results:
            # Submit the task and return the Future object
            futures = [executor_results.submit(process_negative_pair, pair[0], pair[1]) for pair in negative_batch]

            # Manually updating the progress bar
            with tqdm(total=len(negative_batch), desc=f"Processing Negative Pairs - Batch {batch_index}") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    batch_examples.append(result)
                    pbar.update(1)  # Updates the progress bar every time a task is completed

        # Randomly select 3 key examples from the batch
        selected_examples = random.sample(batch_examples, min(3, len(batch_examples)))

        LLM2_Agent_Profile = '''
        Goal: As a knowledge graph alignment expert, analyze these entity pairs and generate logical rules that capture the patterns of alignment and non-alignment.
        Constraint: Focus on extracting generalizable patterns.
        '''

        if ablation_config or no_optimization_tool:
            if ablation_config[0] == 'ablation5' and ablation_config[1] == 'Multi-Granularity':
                prompt = """\nBased on this information, some experience insights are generated. one per line："""
                for example in selected_examples:
                    prompt += f"\n{'Aligned' if example['type'] == 'positive' else 'Non-aligned'} pair:\n"
                    prompt += f"KG1 Entity: {example['kg1']['name']}\n"
                    prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg1']['relations']) + "\n"
                    prompt += f"KG2 Entity: {example['kg2']['name']}\n"
                    prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg2']['relations']) + "\n"

            elif ablation_config[0] == 'ablation5' and ablation_config[1] == 'Communication':
                prompt = """\nBased on this information, some experience insights are generated："""
                for example in selected_examples:
                    prompt += f"\n{'Aligned' if example['type'] == 'positive' else 'Non-aligned'} pair:\n"
                    prompt += f"KG1 Entity: {example['kg1']['name']}\n"
                    prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg1']['relations']) + "\n"
                    prompt += f"KG2 Entity: {example['kg2']['name']}\n"
                    prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg2']['relations']) + "\n"

                prompt += """\nBased on these examples:
                                            ∃x,y(name(x, "Barack Obama") ∧ name(y, "Barack Obama") ∧ role(x, "President") ∧ role(y, "President") ∧ time_period(x, "2009-2017") ∧ time_period(y, "2009-2017")) ⟹ x = y

                                            ∃x,y(name(x, "John Smith") ∧ name(y, "John Smith") ∧ organization(x, "Company A") ∧ organization(y, "Company B")) ⟹ x ≠ y

                                          """
            elif no_optimization_tool:
                prompt = """\nSome experience insights are generated. As much as possible："""
                for example in selected_examples:
                    prompt += f"\n{'Aligned' if example['type'] == 'positive' else 'Non-aligned'} pair:\n"
                    prompt += f"KG1 Entity: {example['kg1']['name']}\n"
                    prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg1']['relations']) + "\n"
                    prompt += f"KG2 Entity: {example['kg2']['name']}\n"
                    prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg2']['relations']) + "\n"

            else:
                prompt = LLM2_Agent_Profile + """
                            Examples to analyze:

                            """
                for example in selected_examples:
                    prompt += f"\n{'Aligned' if example['type'] == 'positive' else 'Non-aligned'} pair:\n"
                    prompt += f"KG1 Entity: {example['kg1']['name']}\n"
                    prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg1']['relations']) + "\n"
                    prompt += f"KG2 Entity: {example['kg2']['name']}\n"
                    prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg2']['relations']) + "\n"

                prompt += """\nBased on these examples, generate positive/negative logical rules in the following format:


                            ∃x,y(name(x, "Barack Obama") ∧ name(y, "Barack Obama") ∧ role(x, "President") ∧ role(y, "President") ∧ time_period(x, "2009-2017") ∧ time_period(y, "2009-2017")) ⟹ x = y

                            ∃x,y(name(x, "John Smith") ∧ name(y, "John Smith") ∧ organization(x, "Company A") ∧ organization(y, "Company B")) ⟹ x ≠ y


                            Generate only the rules, one per line:"""

        else:
            prompt = LLM2_Agent_Profile + """
            Examples to analyze:
    
            """
            for example in selected_examples:
                prompt += f"\n{'Aligned' if example['type'] == 'positive' else 'Non-aligned'} pair:\n"
                prompt += f"KG1 Entity: {example['kg1']['name']}\n"
                prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg1']['relations']) + "\n"
                prompt += f"KG2 Entity: {example['kg2']['name']}\n"
                prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg2']['relations']) + "\n"


            prompt += """\nBased on these examples, generate positive/negative logical rules in the following format:
    
    
            ∃x,y(name(x, "Barack Obama") ∧ name(y, "Barack Obama") ∧ role(x, "President") ∧ role(y, "President") ∧ time_period(x, "2009-2017") ∧ time_period(y, "2009-2017")) ⟹ x = y
    
            ∃x,y(name(x, "John Smith") ∧ name(y, "John Smith") ∧ organization(x, "Company A") ∧ organization(y, "Company B")) ⟹ x ≠ y
    
    
            Generate only the rules, one per line:"""


        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",#"gpt-4-0125-preview"
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.7
            )

            #print(prompt)
            rules = response.choices[0].message.content.strip().split('\n')
            print(rules)
            tokens_cal.update_add_var(response.usage.total_tokens)  # 更新tokens
            all_rules.extend([r for r in rules if r.strip()])
        except Exception as e:
            print(f"Error processing batch: {str(e)}")


    # Process positive pairs
    for batch_index, batch_start in tqdm(enumerate(range(0, len(new_pairs), batch_size)), desc="Processing Batches"):
        executor.submit(openai_task,batch_start,batch_index)

    executor.shutdown(wait=True)

    if all_rules:
        output_file = LLM2_PRIVATE_MESSAGE_POOL['alignment_rules']
        with open(output_file, 'a+', encoding='utf-8') as f:
            for rule in all_rules:
                f.write(rule + '\n')

    return all_rules


if __name__ == "__main__":
    data_dir = "/home/dex/Desktop/entity_sy/AdaCoAgent/data/icews_wiki/"
    rules = run_full_process_llm2(data_dir)
    print(f"Generated {len(rules)} alignment rules.")
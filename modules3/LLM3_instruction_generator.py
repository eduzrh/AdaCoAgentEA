import os
import json
import datetime
from collections import defaultdict
from concurrent.futures import as_completed
from threading import Lock
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import httpx
from openai import OpenAI
from tqdm import tqdm
from itertools import groupby
from operator import itemgetter
import sys

sys.path.append('/home/dex/Desktop/entity_sy/AdaCoAgent')
from ThreadPoolExecutor import ThreadPoolExecutor

import tokens_cal

# Define the private message pool dictionary
LLM3_PRIVATE_MESSAGE_POOL = {}

@dataclass
class EntityInfo:
    id: int
    name: str
    relations: List[str]


class EntityRoleClassifier:
    ROLES = {
        "POLITICAL": "Political Analyst",
        "BUSINESS": "Business Analyst",
        "ORGANIZATION": "Organization Expert",
        "LOCATION": "Geographic Specialist",
        "EVENT": "Event Analysis Specialist",
        "PERSON": "Personal Profile Analyst",
        "MUSIC": "Music Scholar",
        "HISTORY": "History Scholar"
    }

    def __init__(self, data_dir: str, ablation_config = None, no_optimization_tool = False):
        self.data_dir = data_dir
        self.role_assignments = defaultdict(list)
        self.client = OpenAI(
            base_url="your base_url",
            api_key="your-key",
            http_client=httpx.Client(
                base_url="your base_url",
                follow_redirects=True,
            ),
        )

        # 添加总体加载进度条
        with tqdm(total=6, desc="Initializing system") as pbar:
            print("Loading mappings and triples...")
            self.ent_names_1 = self._load_mapping('ent_ids_1')
            pbar.update(1)
            self.ent_names_2 = self._load_mapping('ent_ids_2')
            pbar.update(1)
            self.rel_names_1 = self._load_mapping('rel_ids_1')
            pbar.update(1)
            self.rel_names_2 = self._load_mapping('rel_ids_2')
            pbar.update(1)
            self.triples_1 = self._load_triples('triples_1')
            pbar.update(1)
            self.triples_2 = self._load_triples('triples_2')
            pbar.update(1)

        print("Data loading completed!")

        # Initialize message pool paths
        global LLM3_PRIVATE_MESSAGE_POOL

        # Initializing the Role Assignment Catalog
        self.role_dir_base = os.path.join(self.data_dir, "role_assignments")
        os.makedirs(self.role_dir_base, exist_ok=True)
        LLM3_PRIVATE_MESSAGE_POOL.update({
            'ucon_similarity_results': os.path.join(data_dir, "message_pool", "ucon_similarity_results.txt"),
            'role_assignments': 'role_assignments',
        })

        # Adding a Role Catalog Creation Progress Bar
        with tqdm(total=len(self.ROLES), desc="Creating role directories") as pbar:
            for role in self.ROLES:
                role_dir = os.path.join(self.role_dir_base, role.lower())
                os.makedirs(role_dir, exist_ok=True)
                # Clear existing entity_pairs.txt files
                entity_pairs_file = os.path.join(role_dir, 'entity_pairs.txt')
                if os.path.exists(entity_pairs_file):
                    with open(entity_pairs_file, 'w', encoding='utf-8') as f:
                        f.write('')  # Clear the file
                pbar.update(1)

    def _load_mapping(self, filename: str) -> Dict[int, str]:
        mapping = {}
        filepath = os.path.join(self.data_dir, filename)
        file_size = os.path.getsize(filepath)

        # Use tqdm to display the progress of reading a file, using the number of bytes as an indication of progress
        with open(filepath, 'r', encoding='utf-8') as f, \
                tqdm(total=file_size, desc=f"Loading {filename}", unit='B', unit_scale=True) as pbar:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    mapping[int(parts[0])] = parts[1]
                pbar.update(len(line.encode('utf-8')))
        return mapping

    def _load_triples(self, filename: str) -> List[Tuple]:
        triples = []
        filepath = os.path.join(self.data_dir, filename)
        file_size = os.path.getsize(filepath)

        with open(filepath, 'r', encoding='utf-8') as f, \
                tqdm(total=file_size, desc=f"Loading {filename}", unit='B', unit_scale=True) as pbar:
            for line in f:
                parts = line.strip().split('\t')
                triples.append(tuple(int(x) for x in parts[:3]))
                pbar.update(len(line.encode('utf-8')))
        return triples

    def _get_entity_relations(self, entity_id: int, kg_num: int) -> List[str]:
        relations = []
        triples = self.triples_1 if kg_num == 1 else self.triples_2
        names = self.ent_names_1 if kg_num == 1 else self.ent_names_2
        rel_names = self.rel_names_1 if kg_num == 1 else self.rel_names_2

        for h, r, t in triples:
            if h == entity_id:
                rel_str = rel_names.get(r, f"relation_{r}")
                target_str = names.get(t, f"entity_{t}")
                relations.append(f"{rel_str} -> {target_str}")
            elif t == entity_id:
                rel_str = rel_names.get(r, f"relation_{r}")
                source_str = names.get(h, f"entity_{h}")
                relations.append(f"{source_str} -> {rel_str}")
        return relations[:1]

    def _determine_role(self, entity_pairs: List[Tuple[int, int]], pbar: tqdm = None, ablation_config = None, no_optimization_tool = False) -> str:
        prompt = self._construct_role_prompt(entity_pairs, ablation_config, no_optimization_tool = False)

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{
                    'role': 'user',
                    'content': prompt
                }]
            )

            role = response.choices[0].message.content.strip().upper()

            tokens_cal.update_add_var(response.usage.total_tokens)  # update tokens

            print("prompt",prompt)
            print("role",role)
            if pbar:
                pbar.set_postfix({'Last Role': role}, refresh=True)
            return role if role in self.ROLES else "PERSON"
        except Exception as e:
            print(f"Error in role determination: {e}")
            return "PERSON"

    def _construct_role_prompt(self, entity_pairs: List[Tuple[int, int]], ablation_config = None, no_optimization_tool = False) -> str:
        ablation_config_Com = 0
        if ablation_config or no_optimization_tool:
            if ablation_config[0] == 'ablation5' and ablation_config[1] == 'Communication':
                ablation_config_Com = 1
            if no_optimization_tool:
                ablation_config_Com = 1

        LLM3_Agent_Profile = '''
        Goal: Given the following entity pairs and their relationships, determine which expert role would be most appropriate to evaluate their alignment
        Constraint: Choose from:
POLITICAL, BUSINESS, LOCATION, EVENT, PERSON, MUSIC, HISTORY
        '''
        if ablation_config_Com == 0:
            prompt = LLM3_Agent_Profile + """
    
    Entity pairs and their relationships:
    """
        if ablation_config_Com == 1:
            prompt = """    """

        times = 0
        for e1, e2 in entity_pairs:
            e1_name = self.ent_names_1.get(e1, f"Entity_{e1}")
            e2_name = self.ent_names_2.get(e2, f"Entity_{e2}")
            e1_rels = self._get_entity_relations(e1, 1)
            e2_rels = self._get_entity_relations(e2, 2)

            if times == 0:
                prompt += f"\nPair:\n"
                prompt += f"Entity 1: {e1_name}\n"
                prompt += f"Relationship: {e1_rels[0] if e1_rels else 'No relationship'}\n"
                times = times + 1

            prompt += f"Entity 2: {e2_name}\n"
            prompt += f"Relationship: {e2_rels[0] if e2_rels else 'No relationship'}\n"

        if ablation_config_Com == 0:
            prompt += "\nBased on these entities and relationships, which expert role would be most appropriate? Just return the role name (e.g., POLITICAL):"
        if ablation_config_Com == 1:
            prompt += "which expert role would be most appropriate? as much as possible:"
            # print("ablation_config_Comablation_config_Comablation_config_Comablation_config_Comablation_config_Comablation_config_Comablation_config_Comablation_config_Comablation_config_Comablation_config_Comablation_config_Com")
        return prompt

    def _save_batch_results(self, role: str, pairs: List[Tuple[int, int]], mode: str = 'a'):
        role_dir = os.path.join(self.role_dir_base, role.lower())
        output_file = os.path.join(role_dir, 'entity_pairs.txt')

        with open(output_file, mode, encoding='utf-8') as f:
            for e1, e2 in pairs:
                f.write(f"{e1}\t{e2}\n")

    def _create_checkpoint(self, checkpoint_num: int, processed: int, total: int):
        checkpoint_dir = os.path.join(self.data_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_{checkpoint_num}.json')
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            checkpoint_data = {
                'processed_pairs': processed,
                'total_pairs': total,
                'progress_percentage': round((processed / total) * 100, 2),
                'timestamp': str(datetime.datetime.now()),
                'role_distribution': {role: len(pairs) for role, pairs in self.role_assignments.items()}
            }
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

    def process_entities(self, batch_size: int = 10, save_interval: int = 100, ablation_config = None, no_optimization_tool = False):
        input_file = LLM3_PRIVATE_MESSAGE_POOL['ucon_similarity_results']
        entity_pairs = []

        # Read and count the total number of rows
        total_lines = sum(1 for _ in open(input_file, 'r'))

        # Read entity pairs and sort them by the first ID
        print("\nReading and sorting entity pairs...")
        # Reading entity pairs and counting blank rows
        empty_lines = 0
        with tqdm(total=total_lines, desc="Reading entity pairs") as pbar:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # If it's a empty line
                        empty_lines += 1
                        continue  # Skip empty lines
                    try:
                        e1, e2 = map(int, line.strip().split('\t'))
                        entity_pairs.append((e1, e2))
                        pbar.update(1)
                    except ValueError:
                        print(f"Skipping invalid line: {line}")
                        continue

        # Output the number of blank lines
        print(f"Total empty lines: {empty_lines}")
        print(f"Total entity pairs loaded: {len(entity_pairs)}")
        # Show Sorting Progress
        print("Sorting entity pairs...")
        entity_pairs.sort(key=itemgetter(0))

        # Empty existing role assignment files to show progress
        with tqdm(total=len(self.ROLES), desc="Initializing role files") as pbar:
            for role in self.ROLES:
                self._save_batch_results(role, [], mode='w')
                pbar.update(1)

        total_pairs = len(entity_pairs)
        processed = 0
        batch_counter = 0

        # Master Processing Progress Bar
        main_pbar = tqdm(total=total_pairs, desc="Processing entity pairs")
        role_lock = Lock()

        def process_group(group_pairs):
            nonlocal processed, batch_counter

            print(f"group_data: {group_pairs}")

            # Processing each batch
            for i in range(0, len(group_pairs), batch_size):
                batch = group_pairs[i:min(i + batch_size, len(group_pairs))]
                role = self._determine_role(batch, main_pbar, ablation_config = ablation_config, no_optimization_tool = no_optimization_tool)

                try:
                    with role_lock:
                        # Preservation of batch results
                        self._save_batch_results(role, batch)
                        self.role_assignments[role].extend(batch)
                        processed += len(batch)
                        batch_counter += 1
                        main_pbar.update(len(batch))

                    # Update the progress bar description
                    main_pbar.set_postfix({
                        'Processed': f"{processed}/{total_pairs}",
                        'Progress': f"{(processed / total_pairs * 100):.1f}%"
                    })

                    # Creating Checkpoints
                    if batch_counter % save_interval == 0:
                        self._create_checkpoint(batch_counter // save_interval, processed, total_pairs)

                except Exception as e:
                    print(f"Error processing batch {i}: {e}")

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = []

            entity_pairs.sort(key=itemgetter(0))
            #print(f"First 10 sorted pairs: {entity_pairs[:10]}")

            for first_id, group_data in groupby(entity_pairs, key=itemgetter(0)):
                group_data_list = list(group_data)
                #print(f"First ID {first_id}: {group_data_list}")
                if not group_data_list:
                    print(f"Warning: Group for ID {first_id} is empty!")
                    continue
                futures.append(executor.submit(process_group, group_data_list))
                #print(f"Submitted group for ID {first_id}, total submitted: {len(futures)}, group_data: {group_data_list}")

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in processing: {e}")


                completed = sum(future.done() for future in futures)
                print(f"Progress: {completed}/{len(futures)} tasks completed")

        main_pbar.close()

        print(f"Total pairs processed: {processed}/{total_pairs}")
        if processed < total_pairs:
            print("Warning: Not all pairs were processed!")

        # Show final statistics
        print("\nProcessing completed!")
        print(f"Total pairs processed: {processed}")

        # Use a table to display statistics for each role
        print("\nRole distribution:")
        max_role_len = max(len(role) for role in self.ROLES)
        print("-" * (max_role_len + 15))
        print(f"{'Role':<{max_role_len}} | {'Count':>10}")
        print("-" * (max_role_len + 15))
        for role, pairs in self.role_assignments.items():
            print(f"{role:<{max_role_len}} | {len(pairs):>10,}")
        print("-" * (max_role_len + 15))


def run_full_process_llm3(data_dir, ablation_config = None, no_optimization_tool = False):
    print("Starting entity role classification process...")
    with tqdm(total=1, desc="Initializing classifier") as pbar:
        classifier = EntityRoleClassifier(data_dir, ablation_config = ablation_config, no_optimization_tool = no_optimization_tool)
        pbar.update(1)
    classifier.process_entities(ablation_config = ablation_config, no_optimization_tool = no_optimization_tool)
    print(f'Tokens Cost : {tokens_cal.global_tokens}')


if __name__ == "__main__":

    data_dir = "/home/dex/Desktop/entity_sy/AdaCoAgent/data/icews_wiki"
    run_full_process_llm3(data_dir)
import os
import json
import random
from concurrent.futures import as_completed

import httpx
from openai import OpenAI
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import sys

sys.path.append('/home/dex/Desktop/entity_sy/AdaCoAgent')
from ThreadPoolExecutor import ThreadPoolExecutor


import tokens_cal

def load_entity_names(file_path):
    """Load entity names from file"""
    entity_names = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity_names[int(parts[0])] = parts[1]
    return entity_names


def load_rules(rules_file, num_samples=10):
    """Load and sample alignment rules"""
    with open(rules_file, 'r', encoding='utf-8') as f:
        rules = f.readlines()
    return random.sample(rules, min(num_samples, len(rules)))


def load_existing_descriptions(file_path):
    """Load existing entity descriptions from JSON file if exists"""
    descriptions = {}
    if os.path.exists(file_path):
        print(f"Loading existing descriptions from {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                descriptions_data = json.load(f)
                descriptions = {int(k): v for k, v in descriptions_data.items()}
        except json.JSONDecodeError:
            print(f"Error reading JSON from {file_path}, starting with empty descriptions")
    return descriptions


def update_descriptions(descriptions, entity_id, new_description, role):
    """Update entity descriptions in JSON format with role information"""
    new_desc_entry = {
        "description": new_description,
        "role": role
    }

    if entity_id in descriptions:
        if isinstance(descriptions[entity_id], list):
            existing_descriptions = descriptions[entity_id]
        else:
            # Convert legacy format to new format with role
            existing_descriptions = [{
                "description": descriptions[entity_id],
                "role": "Legacy"
            }]

        # Only add if this role's description doesn't exist
        if not any(entry["role"] == role for entry in existing_descriptions):
            existing_descriptions.append(new_desc_entry)
            descriptions[entity_id] = existing_descriptions
    else:
        descriptions[entity_id] = [new_desc_entry]

    return descriptions[entity_id]


def save_descriptions(descriptions, output_file):
    """Save descriptions to JSON file"""
    #print(f"Saving descriptions to {output_file}")

    # Create backup
    if os.path.exists(output_file):
        backup_file = f"{output_file}.backup"
        try:
            with open(output_file, 'r', encoding='utf-8') as f_src:
                with open(backup_file, 'w', encoding='utf-8') as f_dst:
                    f_dst.write(f_src.read())
            #print(f"Created backup at {backup_file}")
        except Exception as e:
            print(f"Warning: Could not create backup: {str(e)}")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            formatted_descriptions = {}
            for ent_id, descs in descriptions.items():
                if not isinstance(descs, list):
                    descs = [{
                        "description": descs,
                        "role": "Legacy"
                    }]
                formatted_descriptions[str(ent_id)] = descs

            json.dump(formatted_descriptions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving descriptions: {str(e)}")
        if os.path.exists(backup_file):
            print("Attempting to restore from backup...")
            try:
                with open(backup_file, 'r', encoding='utf-8') as f_src:
                    with open(output_file, 'w', encoding='utf-8') as f_dst:
                        f_dst.write(f_src.read())
                print("Restored from backup successfully")
            except Exception as restore_error:
                print(f"Error restoring from backup: {str(restore_error)}")


def has_role_description(descriptions, entity_id, role):
    """Check if entity already has description for specific role"""
    if entity_id not in descriptions:
        return False

    existing_descriptions = descriptions[entity_id]
    if not isinstance(existing_descriptions, list):
        return False

    # Check if any existing description is from this role
    return any(desc.get("role") == role for desc in existing_descriptions)


def run_full_process_llm4_5(data_dir, role_assignments_dir, ablation_config = None, no_optimization_tool = False):
    """Main function to enhance entity knowledge through role-playing"""
    LLM_EXPERT_PRIVATE_MESSAGE_POOL = {
        'kg1_descriptions': os.path.join(data_dir, "message_pool", "KG1_compared_description.json"),
        'kg2_descriptions': os.path.join(data_dir, "message_pool", "KG2_compared_description.json"),
        'alignment_rules': os.path.join(data_dir, "message_pool", "alignment_rules.txt"),
    }
    client = OpenAI(
        base_url="your_base_url",
        api_key="your_api_key",
        http_client=httpx.Client(
            base_url="your_base_url",
            follow_redirects=True,
        ),
    )

    print("Loading entity names...")
    ent_names_1 = load_entity_names(os.path.join(data_dir, 'ent_ids_1'))
    ent_names_2 = load_entity_names(os.path.join(data_dir, 'ent_ids_2'))

    kg1_descriptions = load_existing_descriptions(LLM_EXPERT_PRIVATE_MESSAGE_POOL['kg1_descriptions'])
    kg2_descriptions = load_existing_descriptions(LLM_EXPERT_PRIVATE_MESSAGE_POOL['kg2_descriptions'])

    print("Loading alignment rules...")
    rules = load_rules(LLM_EXPERT_PRIVATE_MESSAGE_POOL['alignment_rules'])
    rules_text = "\n".join(rules)

    role_folders = [f for f in os.listdir(role_assignments_dir)
                    if os.path.isdir(os.path.join(role_assignments_dir, f))]
    print(f"Found {len(role_folders)} role folders")

    for role_folder in tqdm(role_folders, desc="Processing role folders"):
        folder_path = os.path.join(role_assignments_dir, role_folder)

        instruction_file = os.path.join(folder_path, 'instruction.txt')
        if not os.path.exists(instruction_file):
            print(f"Skipping {role_folder}: No instruction.txt found")
            continue

        with open(instruction_file, 'r', encoding='utf-8') as f:
            instruction = f.read().strip()

        pairs_file = os.path.join(folder_path, 'entity_pairs.txt')
        if not os.path.exists(pairs_file):
            print(f"Skipping {role_folder}: No entity_pairs.txt found")
            continue

        def process_openai(e1, e2):

            for ent_id, ent_name, descriptions in [(e1, ent_names_1.get(e1, str(e1)), kg1_descriptions),
                                                   (e2, ent_names_2.get(e2, str(e2)), kg2_descriptions)]:

                # Skip if entity already has description for this role
                if has_role_description(descriptions, ent_id, role_folder):
                    continue
                maximum = 50
                Expert_Agent_Profile = f"""Goal: As an expert in {role_folder}, provide a BRIEF and CONCISE description (maximum {maximum} words) for:
            Entity ID: {ent_id}
            Entity Name: {ent_name}

            Constraint: Focus on core characteristics only, considering these patterns:
            {rules_text}

            Constraint: Keep the description SHORT and FOCUSED."""

                if ablation_config or no_optimization_tool:
                    if ablation_config[0] == 'ablation5' and ablation_config[1] == 'Meta_Expert':
                        Expert_Agent_Profile = f"""provide some information for:
            Entity ID: {ent_id}
            Entity Name: {ent_name}

            Constraint: Focus on core characteristics only, considering these patterns:
            {rules_text}

            Constraint: Keep the description SHORT and FOCUSED."""

                    if ablation_config[0] == 'ablation5' and ablation_config[1] == 'Communication':
                        Expert_Agent_Profile = f"""Goal: As an expert in {role_folder}, provide a BRIEF and CONCISE description for:
            Entity ID: {ent_id}
            Entity Name: {ent_name}
            
            {rules_text}

           """
                    if no_optimization_tool:
                        Expert_Agent_Profile = f"""provide some information for:
                 Entity ID: {ent_id}
                 Entity Name: {ent_name}

                 {rules_text}

                as much as possible
                """
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo-1106",  # "gpt-4-0125-preview",
                            messages=[{'role': 'user', 'content': Expert_Agent_Profile}],
                        )

                        description = response.choices[0].message.content.strip()

                        tokens_cal.update_add_var(response.usage.total_tokens)  # update tokens

                        # print("Expert_Agent_Profile", Expert_Agent_Profile)
                        # print("description", description)
                        update_descriptions(descriptions, ent_id, description, role_folder)
                    except Exception as e:
                        print(f"\nError processing entity {ent_id}: {str(e)}")
                        continue
                else:

                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo-1106",#"gpt-4-0125-preview",
                            messages=[{'role': 'user', 'content': Expert_Agent_Profile}],
                            max_tokens=100
                        )

                        description = response.choices[0].message.content.strip()

                        tokens_cal.update_add_var(response.usage.total_tokens)  # update tokens

                        #print("Expert_Agent_Profile", Expert_Agent_Profile)
                        #print("description", description)
                        update_descriptions(descriptions, ent_id, description, role_folder)

                    except Exception as e:
                        print(f"\nError processing entity {ent_id}: {str(e)}")
                        continue

            save_descriptions(kg1_descriptions, LLM_EXPERT_PRIVATE_MESSAGE_POOL['kg1_descriptions'])
            save_descriptions(kg2_descriptions, LLM_EXPERT_PRIVATE_MESSAGE_POOL['kg2_descriptions'])

        with open(pairs_file, 'r', encoding='utf-8') as f:
            entity_pairs = [tuple(map(int, line.strip().split('\t'))) for line in f]

        print(f"\nProcessing {len(entity_pairs)} entity pairs in {role_folder}")
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for e1, e2 in tqdm(entity_pairs, desc=f"Processing pairs in {role_folder}"):
                futures.append(executor.submit(process_openai, e1,e2))
            for future in as_completed(futures):
                future.result()

    print("\nProcess completed!")
    return kg1_descriptions, kg2_descriptions


if __name__ == "__main__":
    data_dir = "/home/dex/Desktop/entity_sy/AdaCoAgent/data/icews_wiki"
    role_assignments_dir = os.path.join(data_dir, "role_assignments")
    print("Starting knowledge enhancement process...")
    kg1_desc, kg2_desc = run_full_process_llm4_5(data_dir, role_assignments_dir)
import os
import time
import sys
import re
import argparse
from modules import candidate_entity_retrieval, labeling_and_model_training, role_assignment_and_knowledge_enhancement, calculate_abs_hits_at_1, run_full_process_llm2, run_full_process_s4
from Area1.LLM_rag import load_ents, retrieve_top_k_entities, setup_retriever, llm_rag_all
from Area1.cal_degree_important import s2_degree_and_s3_important
from Area2.LLM1_label_selector import align_entities
from Area3.LLM3_instruction_generator import run_full_process_llm3
from Area3.LLM4_5_expert_system import run_full_process_llm4_5
import openai
import tokens_cal
from Area1.struc_sim import load_triples, load_ref_ent_ids, structure_similarity
import shutil
from collections import defaultdict

# OpenAI API configuration
os.environ["OPENAI_API_BASE"] = 'your_base'
os.environ["OPENAI_API_KEY"] = "your_base_key"

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# Ablation category mappings
ABLATION_CATEGORIES = {
    'ablation1': ['S1', 'S2', 'S3', 'S4'],
    'ablation2': [
        'LLM1_S1', 'LLM1_S2', 'LLM1_S3', 'LLM1_S4',
        'LLM2_S1', 'LLM2_S2', 'LLM2_S3', 'LLM2_S4',
        'LLM3_S1', 'LLM3_S2', 'LLM3_S3', 'LLM3_S4',
        'DomainExperts_S1', 'DomainExperts_S2', 'DomainExperts_S3', 'DomainExperts_S4'
    ],
    'ablation3': [
        'no_LLM1', 'no_LLM2', 'no_LLM3', 'no_DomainExperts',
        'no_S1', 'no_S2', 'no_S3', 'no_S4'
    ],
    'ablation4': ['Area1', 'Area3'],
    'ablation5': [
        'LLM_Agents', 'Multi-Granularity',
        'Meta_Expert', 'Communication'
    ]
}




def validate_ablation_args(args):
    """Validate ablation arguments and return selected ablation config"""
    ablation_selected = []
    for category in ABLATION_CATEGORIES:
        if getattr(args, category):
            ablation_selected.append((category, getattr(args, category)))

    if len(ablation_selected) > 1:
        print("Error: Only one ablation category can be selected at a time.")
        sys.exit(1)

    if not ablation_selected:
        return None

    ablation_type, ablation_value = ablation_selected[0]

    # Check invalid combinations
    if ablation_type == 'ablation1' and ablation_value in ['S2', 'S3', 'S4']:
        print("Error: Lack of necessary pre-conditions, the framework does not work, the effect is 0.")
        sys.exit(1)

    if ablation_type == 'ablation2' and ablation_value in [
        'LLM1_S2', 'LLM1_S3', 'LLM1_S4', 'LLM2_S2', 'LLM2_S3', 'LLM2_S4', 'LLM3_S2', 'LLM3_S3', 'LLM3_S4', 'DomainExperts_S2', 'DomainExperts_S3', 'DomainExperts_S4'
    ]:
        print("Error: Lack of necessary pre-conditions, the framework does not work, the effect is 0.")
        sys.exit(1)
    if ablation_type == 'ablation3' and ablation_value in [
        'no_S1'
    ]:
        print("Error: Lack of necessary pre-conditions, the framework does not work, the effect is 0.")
        sys.exit(1)
    # Add more validation checks here if needed

    return ablation_selected[0]

# Function to query GPT model
def gpt3_query(prompt, model="gpt-4-0125-preview"):
    """Query LLM model and get response."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']


# Function to run candidate entity retrieval process
def run_candidate_entity_retrieval(data_dir, is_activation_m1=True):
    """Run candidate entity retrieval process."""
    print("Running candidate entity retrieval...")
    llm_rag_all(data_dir)
    candidate_entity_retrieval(data_dir, is_activation_m1 = is_activation_m1)


# Function to run labeling and model training process
def run_labeling_and_model_training(data_dir, from_m3=False, is_activation_m2=True):
    """Run labeling and model training process."""
    if is_activation_m2:
        print("Running labeling and model training...")
        labeling_and_model_training(data_dir, from_m3)


# Function to run role assignment and knowledge enhancement process
def run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3=True, ablation_config = None, no_optimization_tool = False):
    """Run role assignment and knowledge enhancement process."""
    if is_activation_m3:
        print("Running role assignment and knowledge enhancement...")
        role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = is_activation_m3, ablation_config = ablation_config, no_optimization_tool = no_optimization_tool)


def calculate_average_degree(triples_file):
    """Calculate average degree of entities in a knowledge graph"""
    entity_degrees = defaultdict(int)
    valid_lines = 0

    with open(triples_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            # Processing triples according to the original data format
            if len(parts) == 5 or len(parts) == 4 or len(parts) == 3:  # Make sure it's the full ternary format
                head = parts[0]
                tail = parts[2]
                entity_degrees[head] += 1
                entity_degrees[tail] += 1
                valid_lines += 1

    if not entity_degrees:
        return 0.0

    total_degree = sum(entity_degrees.values())
    num_entities = len(entity_degrees)
    return total_degree / num_entities

def calculate_degree_difference(data_dir):
    """Calculate the average degree difference between two knowledge graphs"""
    kg1_file = os.path.join(data_dir, 'triples_1')
    kg2_file = os.path.join(data_dir, 'triples_2')
    avg_degree_kg1 = calculate_average_degree(kg1_file)
    print("avg_degree_kg1:",avg_degree_kg1)
    avg_degree_kg2 = calculate_average_degree(kg2_file)
    print("avg_degree_kg2:",avg_degree_kg2)
    return abs(avg_degree_kg1 - avg_degree_kg2)

def is_activation_area1(data_dir):
    # Setting threshold parameters
    K_d = 50  # Mean Degree Difference Threshold
    theta_str = 0.5  # Structural similarity threshold


    # Determine dataset type based on data_dir
    dataset_name = None
    if 'dbp15k' in data_dir or 'fr_en' in data_dir:
        dataset_name = 'DBP15K'
    elif 'dbp-wiki' in data_dir or 'dbp_wiki' in data_dir or 'dbp_wd_100' in data_dir:
        dataset_name = 'DBP-WIKI'
    elif 'icews-wiki' in data_dir or 'icews_wiki' in data_dir:
        dataset_name = 'ICEWS-WIKI'
    elif 'icews-yago' in data_dir or 'icews_yago' in data_dir:
        dataset_name = 'ICEWS-YAGO'
    elif 'beta' in data_dir or 'BETA' in data_dir:
        dataset_name = 'BETA'
    else:
        raise ValueError(f"Unable to recognize dataset type from path {data_dir}")


    # The following section is derived from statistics in the simple-hhea paper.
    # Define the average degree of difference across the datasets
    degree_difference = {
        'DBP15K': 1.08,
        'DBP-WIKI': 0.423,
        'ICEWS-WIKI': 306.88,
        'ICEWS-YAGO': 151.36,
        'BETA': 12.06
    }

    # Define structural similarity across datasets
    structure_similarity = {
        'DBP15K': 0.634,
        'DBP-WIKI': 0.748,
        'ICEWS-WIKI': 0.154,
        'ICEWS-YAGO': 0.140,
        'BETA': 0.6517
    }

    # Calculate the actual degree difference
    # d = calculate_degree_difference(data_dir)

    # kg1_adj = load_triples(os.path.join(data_dir, 'triples_1'))
    # kg2_adj = load_triples(os.path.join(data_dir, 'triples_2'))
    # aligned_pairs = load_ref_ent_ids(os.path.join(data_dir, 'ref_ent_ids'))
    #
    # similarity = structure_similarity(kg1_adj, kg2_adj, aligned_pairs)
    # S_str = similarity
    # print(f"Structure Similarity: {similarity:.4f}")

    # Get the value of the current dataset
    d = degree_difference[dataset_name]
    S_str = structure_similarity[dataset_name]

    # conditional judgement
    if d > K_d or S_str < theta_str:
        print(f"Dataset {dataset_name} Activation area 1:")
        print(f"average degree of disparity = {d:.3f} {'>' if d > K_d else '<='} {K_d}")
        print(f"structural similarity = {S_str * 100:.1f}% {'<' if S_str < theta_str else '>='} {theta_str * 100}%")
        return True
    else:
        print(f"Dataset {dataset_name} inactive area 1:")
        print(f"average degree of disparity = {d:.3f} {'>' if d > K_d else '<='} {K_d}")
        print(f"structural similarity = {S_str * 100:.1f}% {'<' if S_str < theta_str else '>='} {theta_str * 100}%")
        return False


def is_activation_area2(data_dir, src):
    # Check input sources and corresponding files
    if src == 'M1':
        # Check if important_entities.txt exists and is not empty
        file_path = os.path.join(data_dir, 'message_pool', 'retriever_outputs.txt')
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            print("originates from Area 1 and top-k candidate entities are non-empty, activates Area 2")
            return True
        else:
            print("Derived from Area 1, but top-k candidate entities are empty or do not exist, do not activate Area 2")
            return False

    elif src == 'M3':
        # Check if KG1_compared_description.json exists and is not empty
        file_path = os.path.join(data_dir, 'message_pool', 'KG1_compared_description.json')
        file_path2= os.path.join(data_dir, 'message_pool', 'ucon_similarity_results.txt')
        if os.path.exists(file_path) and os.path.exists(file_path2) and os.path.getsize(file_path) > 0 and os.path.getsize(file_path2) > 0:
            print("Sourced from Area 3 and KG1_compared_description.json is not empty, activate Area 2")
            return True
        else:
            print("Sourced from Area 3, but KG1_compared_description.json is empty or does not exist, do not activate Area 2")
            return False

    elif src == 'M3_false':
        print("Area 3 is not activated and no new knowledge is added")
        return False

    else:
        print(f"Invalid input source: {src}, should be 'S1' or 'M3'")
        return False


def is_activation_area3(data_dir):
    ucon_file = os.path.join(data_dir, 'message_pool', 'ucon_similarity_results.txt')

    # ========== Dynamically Generated Result File Path ==========
    # 1. Extract dataset name from data_dir
    dataset_name = os.path.basename(data_dir)  # get data

    # 2. Build result file name
    result_filename = f"{dataset_name}_result_file_mlp.txt"

    # 3. Get the result directory path
    base_dir = os.path.dirname(os.path.dirname(data_dir))  # Getting to the Catalog
    print(base_dir)
    result_dir = os.path.join(base_dir, "result")  # Splicing the results catalog
    result_file = os.path.join(result_dir, result_filename)  # Path to full results file
    # =======================================

    # 1. Checking the original conditions
    if not (os.path.exists(ucon_file) and os.path.getsize(ucon_file) > 0):
        print("ucon_similarity_results.txt is empty or does not exist, do not activate area 3")
        return False

    # 2. Parsing the results file
    def parse_hits_at_1(file_path):
        hits_list = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if "best results: hits@[1, 5, 10]" in line:
                        # Precise matching of target value components
                        match = re.search(
                            r'hits@\[1,\s*5,\s*10\]\s*=\s*\[([\d.]+),\s*([\d.]+),\s*([\d.]+)\]',
                            line
                        )
                        if match:
                            hits_at_1 = float(match.group(1))
                            hits_list.append(hits_at_1)
        except FileNotFoundError:
            print(f"Result file does not exist: {file_path}")
        except ValueError:
            print(f"Numerical formatting error: {line.strip()}")
        return hits_list

    # 3. Get historical hits@1 records
    hits_history = parse_hits_at_1(result_file)

    # 4. Convergence judgment
    if len(hits_history) >= 2:
        print(hits_history)
        print(hits_history[-1])
        print(hits_history[-2])
        delta = abs(hits_history[-1] - hits_history[-2])/100 #x percent
        if delta < 0.01:
            print(f"Convergence detected (Δhits@1={delta:.4f} < 0.01), do not activate area 3")
            return False
        else:
            print(f"Convergence condition not reached (Δhits@1={delta:.4f} >= 0.01), activate area 3")
            return True
    else:
        print("Results recorded in less than 2 rounds, default activation of area 3")
        return True
#
#
# # Full process execution
# def run_full_process(data_dir):
#     start_time = time.time()
#

#     is_activation_m1, is_activation_m2, is_activation_m3 = True, True, True
#
#     #Step 1: Candidate entity retrieval
#     is_activation_m1 = is_activation_area1(data_dir)
#     run_candidate_entity_retrieval(data_dir, is_activation_m1)
#
#     # Step 2: Labeling and model training
#     is_activation_m2 = is_activation_area2(data_dir, src = 'M1')
#     run_labeling_and_model_training(data_dir, False, is_activation_m2)
#
#     # Step 3: Role assignment and knowledge enhancement
#     is_activation_m3 = is_activation_area3(data_dir)
#     run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3)
#
#
#     # Step 4: Check termination conditions and continue if necessary
#     times = 1
#     max_time = 3
#     while is_activation_m2 or is_activation_m3:
#         is_activation_m2 = is_activation_area2(data_dir, src='M3')
#         run_labeling_and_model_training(data_dir, True, is_activation_m2)
#         times = times + 1
#         is_activation_m3 = is_activation_area3(data_dir)
#         # Truncate iterations to reduce costs
#         if times >= max_time:
#             is_activation_m3 = False
#         run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3)
#         if is_activation_m3 == False:
#             src = 'M3_false'
#             is_activation_m2 = is_activation_area2(data_dir, src)
#
#     # Record total time taken
#     end_time = time.time()
#     total_seconds = end_time - start_time
#
#     hours = int(total_seconds // 3600)
#     minutes = int((total_seconds % 3600) // 60)
#     seconds = int(total_seconds % 60)
#
#     print(f"Final Process completed in: {end_time - start_time:.2f} seconds")
#     print(f'Time Cost : {hours}hour, {minutes:02d}min, {seconds:02d}sec')
#     print(f'Tokens Cost : {tokens_cal.global_tokens}')


def run_full_process(data_dir, ablation_config=None, no_optimization_tool=False):
    start_time = time.time()

    # Initialize ablation settings
    ablation_params = {}
    if ablation_config:
        ablation_type, ablation_value = ablation_config
        ablation_params[ablation_type] = ablation_value

    # Apply ablation settings
    is_activation_m1, is_activation_m2, is_activation_m3 = True, True, True

    # Handle different ablation types
    if ablation_config or no_optimization_tool:
        ablation_type, ablation_value = ablation_config

        # Category 1: LLM Agents + single small model
        if ablation_type == 'ablation1':
            print(f"Running ablation 1: LLM Agents + {ablation_value} Agent")

            # Example: disable other small models
            if ablation_value == 'S1':
                # S1+LLM Agent
                llm_rag_all(data_dir, ablation_config) # S1
                src_path = os.path.join(data_dir, "message_pool", "retriever_outputs.txt")
                dst_path = os.path.join(data_dir, "message_pool", "important_entities.txt")
                shutil.copyfile(src_path, dst_path)

                aligned_pairs = align_entities(data_dir, from_m3 = False) # LLM 1

                print(f"Found {len(aligned_pairs)} aligned entity pairs.")
                is_activation_m3 = False

                print("Lack of necessary preconditions for subsequent runs, termination of the framework run")
                ref_path = os.path.join(data_dir, "ref_pairs")
                pred_path = os.path.join(data_dir, "message_pool", "aligned_entities.txt")
                hits = calculate_abs_hits_at_1(ref_path, pred_path) #Functions with built-in print results
            else:
                # S2+LLM Agent，S3+LLM Agents，S4+LLM Agents
                print("Lack of necessary preconditions for subsequent runs, termination of the framework run")


        # Category 2: Single LLM + small model
        elif ablation_type == 'ablation2':
            print(f"Running ablation 2: {ablation_value}")
            # Extract LLM and small model from value
            llm_part, model_part = ablation_value.split('_')
            if llm_part == 'LLM1' and model_part == 'S1':
                # LLM1 + S1 Agent
                llm_rag_all(data_dir, ablation_config) # S1
                src_path = os.path.join(data_dir, "message_pool", "retriever_outputs.txt")
                dst_path = os.path.join(data_dir, "message_pool", "important_entities.txt")
                shutil.copyfile(src_path, dst_path)

                aligned_pairs = align_entities(data_dir, from_m3=False)  # LLM 1
                print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                ref_path = os.path.join(data_dir, "ref_pairs")
                pred_path = os.path.join(data_dir, "message_pool", "aligned_entities.txt")
                hits = calculate_abs_hits_at_1(ref_path, pred_path) #Functions with built-in print results

            elif llm_part == 'LLM2' and model_part == 'S1':
                # LLM2 + S1 Agent
                llm_rag_all(data_dir, ablation_config) # S1
                src_path = os.path.join(data_dir, "message_pool", "retriever_outputs.txt")
                dst_path = os.path.join(data_dir, "message_pool", "aligned_entities.txt")
                shutil.copyfile(src_path, dst_path)

                try:
                    run_full_process_llm2(data_dir) #LLM2
                except Exception as e:
                    print("LLM2 could not run due to lack of necessary preconditions to support it.")
                    # print(f"Error details: {str(e)}")


                ref_path = os.path.join(data_dir, "ref_pairs")
                pred_path = os.path.join(data_dir, "message_pool", "aligned_entities.txt")
                calculate_abs_hits_at_1(ref_path, pred_path) #Functions with built-in print results

            elif llm_part == 'LLM3' and model_part == 'S1':
                # LLM3 + S1 Agent
                llm_rag_all(data_dir, ablation_config) # S1
                src_path = os.path.join(data_dir, "message_pool", "retriever_outputs.txt")
                dst_path = os.path.join(data_dir, "message_pool", "aligned_entities.txt")
                shutil.copyfile(src_path, dst_path)

                try:
                    run_full_process_llm3(data_dir) #LLM3
                except Exception as e:
                    print("LLM3 could not run due to lack of necessary preconditions to support it.")
                    # print(f"Error details: {str(e)}")

                ref_path = os.path.join(data_dir, "ref_pairs")
                pred_path = os.path.join(data_dir, "message_pool", "aligned_entities.txt")
                calculate_abs_hits_at_1(ref_path, pred_path) #Functions with built-in print results

            elif llm_part == 'DomainExperts' and model_part == 'S1':
                # LLM4 + S1 Agent
                llm_rag_all(data_dir, ablation_config) # S1
                src_path = os.path.join(data_dir, "message_pool", "retriever_outputs.txt")
                dst_path = os.path.join(data_dir, "message_pool", "aligned_entities.txt")
                shutil.copyfile(src_path, dst_path)
                try:
                    role_assignments_dir = os.path.join(data_dir, "role_assignments")  # DomainExperts

                    run_full_process_llm4_5(data_dir, role_assignments_dir)

                    print("Role assignment and knowledge enhancement process complete.")
                except Exception as e:
                    print("DomainExperts could not run due to lack of necessary preconditions to support it.")
                    # print(f"Error details: {str(e)}")

                ref_path = os.path.join(data_dir, "ref_pairs")
                pred_path = os.path.join(data_dir, "message_pool", "aligned_entities.txt")
                calculate_abs_hits_at_1(ref_path, pred_path) #Functions with built-in print results
            else:

                print("Lack of necessary preconditions for subsequent runs, termination of the framework run")


        # Category 3: Remove agent
        elif ablation_type == 'ablation3':
            print(f"Running ablation 3: Remove {ablation_value}")
            # Example: disable specific agent
            if ablation_value == 'no_S4':
                # W/o S4 Agent
                is_activation_m1 = is_activation_area1(data_dir)
                llm_rag_all(data_dir, ablation_config) # S1
                candidate_entity_retrieval(data_dir, is_activation_m1 = is_activation_m1)

                aligned_pairs = align_entities(data_dir, from_m3=False)  # LLM 1
                print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                try:
                    run_full_process_llm2(data_dir) #LLM2
                except Exception as e:
                    print("LLM2 could not run due to lack of necessary preconditions to support it.")
                    # print(f"Error details: {str(e)}")

                # Step 3: Role assignment and knowledge enhancement
                is_activation_m3 = is_activation_area3(data_dir)
                run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = is_activation_m3)

                # Step 4: Check termination conditions and continue if necessary
                times = 1
                max_time = 3
                while is_activation_m2 or is_activation_m3:
                    is_activation_m2 = is_activation_area2(data_dir, src='M3')

                    if is_activation_m2 == True:
                        aligned_pairs = align_entities(data_dir, from_m3=True)  # LLM 1
                        print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                        try:
                            run_full_process_llm2(data_dir)  # LLM2
                        except Exception as e:
                            print("LLM2 could not run due to lack of necessary preconditions to support it.")
                            # print(f"Error details: {str(e)}")

                    times = times + 1
                    is_activation_m3 = is_activation_area3(data_dir)
                    # Truncate iterations to reduce costs
                    if times >= max_time:
                        is_activation_m3 = False
                    run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = is_activation_m3)
                    if is_activation_m3 == False:
                        src = 'M3_false'
                        is_activation_m2 = is_activation_area2(data_dir, src)

                ref_path = os.path.join(data_dir, "ref_pairs")
                pred_path = os.path.join(data_dir, "message_pool", "aligned_entities.txt")
                calculate_abs_hits_at_1(ref_path, pred_path) #Functions with built-in print results

            elif ablation_value == 'no_S2':
                # W/o S2 Agent
                is_activation_m1 = is_activation_area1(data_dir)
                llm_rag_all(data_dir, ablation_config) # S1
                s2_degree_and_s3_important(data_dir, is_activation_m1 = is_activation_m1, ablation_config = ablation_config)

                # Step 2: Labeling and model training
                is_activation_m2 = is_activation_area2(data_dir, src='M1')
                run_labeling_and_model_training(data_dir, False, is_activation_m2)

                # Step 3: Role assignment and knowledge enhancement
                is_activation_m3 = is_activation_area3(data_dir)
                run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = is_activation_m3)

                # Step 4: Check termination conditions and continue if necessary
                times = 1
                max_time = 3
                while is_activation_m2 or is_activation_m3:
                    is_activation_m2 = is_activation_area2(data_dir, src='M3')
                    run_labeling_and_model_training(data_dir, True, is_activation_m2)
                    times = times + 1
                    is_activation_m3 = is_activation_area3(data_dir)
                    # Truncate iterations to reduce costs
                    if times >= max_time:
                        is_activation_m3 = False
                    run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = is_activation_m3)
                    if is_activation_m3 == False:
                        src = 'M3_false'
                        is_activation_m2 = is_activation_area2(data_dir, src)

            elif ablation_value == 'no_S3':
                # W/o S3 Agent
                is_activation_m1 = is_activation_area1(data_dir)
                llm_rag_all(data_dir, ablation_config) # S1
                s2_degree_and_s3_important(data_dir, is_activation_m1 = is_activation_m1, ablation_config = ablation_config)

                # Step 2: Labeling and model training
                is_activation_m2 = is_activation_area2(data_dir, src='M1')
                run_labeling_and_model_training(data_dir, False, is_activation_m2)

                # Step 3: Role assignment and knowledge enhancement
                is_activation_m3 = is_activation_area3(data_dir)
                run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = is_activation_m3)

                # Step 4: Check termination conditions and continue if necessary
                times = 1
                max_time = 3
                while is_activation_m2 or is_activation_m3:
                    is_activation_m2 = is_activation_area2(data_dir, src='M3')
                    run_labeling_and_model_training(data_dir, True, is_activation_m2)
                    times = times + 1
                    is_activation_m3 = is_activation_area3(data_dir)
                    # Truncate iterations to reduce costs
                    if times >= max_time:
                        is_activation_m3 = False
                    run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = is_activation_m3)
                    if is_activation_m3 == False:
                        src = 'M3_false'
                        is_activation_m2 = is_activation_area2(data_dir, src)

            elif ablation_value == 'no_LLM1':
                # W/o LLM1s Agent
                is_activation_m1 = is_activation_area1(data_dir)
                llm_rag_all(data_dir, ablation_config) # S1
                s2_degree_and_s3_important(data_dir, is_activation_m1 = is_activation_m1, ablation_config = ablation_config)

                src_path = os.path.join(data_dir, "message_pool", "important_entities.txt")
                dst_path = os.path.join(data_dir, "message_pool", "aligned_entities.txt")
                shutil.copyfile(src_path, dst_path)


                run_full_process_s4(data_dir, "Simple-HHEA")  # S4

                # Due to the lack of LLM1 and the lack of the necessary antecedent support conditions,
                # the results of the subsequent LLM2, 3, and 4 cannot be processed by LLM1,
                # and thus cannot update aligned_entities.txt,
                # and the subsequent effects of the framework will remain unchanged, and therefore terminated.

            elif ablation_value == 'no_LLM2':
                # W/o LLM2 Agent
                is_activation_m1 = is_activation_area1(data_dir)
                llm_rag_all(data_dir, ablation_config)  # S1
                s2_degree_and_s3_important(data_dir, is_activation_m1=is_activation_m1, ablation_config=ablation_config)

                # 1. LLM1 generates labels
                print("Running LLM1 label selection...")
                aligned_pairs = align_entities(data_dir, from_m3=False)
                print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                run_full_process_s4(data_dir, "Simple-HHEA")  # S4

                # Due to the lack of LLM2 and the lack of the necessary antecedent support conditions,
                # the results of the subsequent LLM 3, and 4 cannot be processed by LLM1,
                # and thus cannot update aligned_entities.txt,
                # and the subsequent effects of the framework will remain unchanged, and therefore terminated.

            elif ablation_value == 'no_LLM3':
                # W/o LLM3 Agent
                is_activation_m1 = is_activation_area1(data_dir)
                llm_rag_all(data_dir, ablation_config) # S1
                s2_degree_and_s3_important(data_dir, is_activation_m1 = is_activation_m1, ablation_config = ablation_config)

                # 1. LLM1 generates labels
                print("Running LLM1 label selection...")
                aligned_pairs = align_entities(data_dir, from_m3 = False)
                print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                run_full_process_s4(data_dir, "Simple-HHEA")  # S4

                # 3. LLM2 generation rules
                run_full_process_llm2(data_dir)
                # Lack of LLM3 and necessary antecedent support conditions stops the framework from running.

            elif ablation_value == 'no_DomainExperts':
                # W/o LLM1s Agent
                is_activation_m1 = is_activation_area1(data_dir)
                llm_rag_all(data_dir, ablation_config) # S1
                s2_degree_and_s3_important(data_dir, is_activation_m1 = is_activation_m1, ablation_config = ablation_config)

                # 1. LLM1 generates labels
                print("Running LLM1 label selection...")
                aligned_pairs = align_entities(data_dir, from_m3 = False)
                print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                run_full_process_s4(data_dir, "Simple-HHEA")  # S4

                # 3. LLM2 generation rules
                run_full_process_llm2(data_dir)
                # Lack of DomainExperts and necessary antecedent support conditions stops the framework from running.

            else:
                print("Lack of necessary antecedent support conditions stops the framework from running")
        # Category 4: Remove area
        elif ablation_type == 'ablation4' and no_optimization_tool:
            if no_optimization_tool:
                # W/o Remove area
                if ablation_value == 'Area1':
                    # W/o Area1
                    #is_activation_m1 = is_activation_area1(data_dir)
                    #llm_rag_all(data_dir, ablation_config)  # S1
                    src_path = os.path.join(data_dir, "message_pool", "retriever_outputs.txt")
                    dst_path = os.path.join(data_dir, "message_pool", "important_entities.txt")
                    shutil.copyfile(src_path, dst_path)

                    is_activation_m2 = is_activation_area2(data_dir, src='M1')
                    aligned_pairs = align_entities(data_dir, from_m3=False, ablation_config=ablation_config,
                                                   no_optimization_tool=no_optimization_tool)  # LLM 1
                    print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                    run_full_process_s4(data_dir, "Simple-HHEA")
                    run_full_process_llm2(data_dir, batch_size=500, ablation_config=ablation_config,
                                          no_optimization_tool=no_optimization_tool)  # LLM2

                    # Step 3: Role assignment and knowledge enhancement
                    is_activation_m3 = is_activation_area3(data_dir)
                    run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3=is_activation_m3,
                                                                  ablation_config=ablation_config,
                                                                  no_optimization_tool=no_optimization_tool)

                    # Step 4: Check termination conditions and continue if necessary
                    times = 1
                    max_time = 3
                    while is_activation_m2 or is_activation_m3:
                        is_activation_m2 = is_activation_area2(data_dir, src='M3')

                        aligned_pairs = align_entities(data_dir, from_m3=False, ablation_config=ablation_config,
                                                       no_optimization_tool=no_optimization_tool)  # LLM 1
                        print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                        run_full_process_s4(data_dir, "Simple-HHEA")
                        run_full_process_llm2(data_dir, batch_size=500, ablation_config=ablation_config,
                                              no_optimization_tool=no_optimization_tool)  # LLM2

                        times = times + 1
                        is_activation_m3 = is_activation_area3(data_dir)
                        # Truncate iterations to reduce costs
                        if times >= max_time:
                            is_activation_m3 = False
                        run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3=is_activation_m3,
                                                                      ablation_config=ablation_config,
                                                                      no_optimization_tool=no_optimization_tool)
                        if is_activation_m3 == False:
                            src = 'M3_false'
                            is_activation_m2 = is_activation_area2(data_dir, src)

                elif ablation_value == 'Area3':
                    is_activation_m1 = is_activation_area1(data_dir)
                    llm_rag_all(data_dir, ablation_config)  # S1
                    candidate_entity_retrieval(data_dir, is_activation_m1=is_activation_m1)  # S2,S3

                    is_activation_m2 = is_activation_area2(data_dir, src='M1')
                    aligned_pairs = align_entities(data_dir, from_m3=False, ablation_config=ablation_config,
                                                   no_optimization_tool=no_optimization_tool)  # LLM 1
                    print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                    run_full_process_s4(data_dir, "Simple-HHEA")
                    run_full_process_llm2(data_dir, batch_size=500, ablation_config=ablation_config,
                                          no_optimization_tool=no_optimization_tool)  # LLM2



                else:

                    is_activation_m1 = is_activation_area1(data_dir)
                    llm_rag_all(data_dir, ablation_config)  # S1
                    candidate_entity_retrieval(data_dir, is_activation_m1=is_activation_m1)  # S2,S3

                    is_activation_m2 = is_activation_area2(data_dir, src='M1')
                    aligned_pairs = align_entities(data_dir, from_m3=False, ablation_config=ablation_config,
                                                   no_optimization_tool=no_optimization_tool)  # LLM 1
                    print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                    run_full_process_s4(data_dir, "Simple-HHEA")
                    run_full_process_llm2(data_dir, batch_size=500, ablation_config=ablation_config,
                                          no_optimization_tool=no_optimization_tool)  # LLM2

                    # Step 3: Role assignment and knowledge enhancement
                    is_activation_m3 = is_activation_area3(data_dir)
                    run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3=is_activation_m3,
                                                                  ablation_config=ablation_config,
                                                                  no_optimization_tool=no_optimization_tool)

                    # Step 4: Check termination conditions and continue if necessary
                    times = 1
                    max_time = 3
                    while is_activation_m2 or is_activation_m3:
                        is_activation_m2 = is_activation_area2(data_dir, src='M3')

                        aligned_pairs = align_entities(data_dir, from_m3=False, ablation_config=ablation_config,
                                                       no_optimization_tool=no_optimization_tool)  # LLM 1
                        print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                        run_full_process_s4(data_dir, "Simple-HHEA")
                        run_full_process_llm2(data_dir, batch_size=500, ablation_config=ablation_config,
                                              no_optimization_tool=no_optimization_tool)  # LLM2

                        times = times + 1
                        is_activation_m3 = is_activation_area3(data_dir)
                        # Truncate iterations to reduce costs
                        if times >= max_time:
                            is_activation_m3 = False
                        run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3=is_activation_m3,
                                                                      ablation_config=ablation_config,
                                                                      no_optimization_tool=no_optimization_tool)
                        if is_activation_m3 == False:
                            src = 'M3_false'
                            is_activation_m2 = is_activation_area2(data_dir, src)
            print(f"Running ablation 4: Remove {ablation_value}")



        # Category 5: Remove component
        elif ablation_type == 'ablation5':
            print(f"Running ablation 5: Remove {ablation_value}")
            if ablation_value == 'LLM_Agents':
                # W/o LLM_Agents
                is_activation_m1 = is_activation_area1(data_dir)
                llm_rag_all(data_dir, ablation_config)  # S1
                candidate_entity_retrieval(data_dir, is_activation_m1 = is_activation_m1)  # S2,S3

                src_path = os.path.join(data_dir, "message_pool", "important_entities.txt")
                dst_path = os.path.join(data_dir, "message_pool", "aligned_entities.txt")
                shutil.copyfile(src_path, dst_path)

                try:
                    run_full_process_s4(data_dir, "Simple-HHEA")
                except Exception as e:
                    print("S4 could not run due to lack of necessary preconditions to support it.")
                    # print(f"Error details: {str(e)}")
            # 'LLM_Agents', 'Multi-Granularity',
            # 'Meta_Expert', 'Communication'
            elif ablation_value == 'Multi-Granularity':
                # W/o Multi-Granularity

                is_activation_m1 = is_activation_area1(data_dir)
                llm_rag_all(data_dir, ablation_config)  # S1
                candidate_entity_retrieval(data_dir, is_activation_m1 = is_activation_m1)  # S2,S3

                is_activation_m2 = is_activation_area2(data_dir, src='M1')
                aligned_pairs = align_entities(data_dir, from_m3=False)  # LLM 1
                print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                run_full_process_s4(data_dir, "Simple-HHEA")
                run_full_process_llm2(data_dir, batch_size=500, ablation_config = ablation_config)  # LLM2

                # Step 3: Role assignment and knowledge enhancement
                is_activation_m3 = is_activation_area3(data_dir)
                run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = is_activation_m3)

                # Step 4: Check termination conditions and continue if necessary
                times = 1
                max_time = 3
                while is_activation_m2 or is_activation_m3:
                    is_activation_m2 = is_activation_area2(data_dir, src='M3')

                    aligned_pairs = align_entities(data_dir, from_m3=True)  # LLM 1
                    print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                    run_full_process_s4(data_dir, "Simple-HHEA")
                    run_full_process_llm2(data_dir, batch_size=500, ablation_config=ablation_config)  # LLM2

                    times = times + 1
                    is_activation_m3 = is_activation_area3(data_dir)
                    # Truncate iterations to reduce costs
                    if times >= max_time:
                        is_activation_m3 = False
                    run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = is_activation_m3)
                    if is_activation_m3 == False:
                        src = 'M3_false'
                        is_activation_m2 = is_activation_area2(data_dir, src)

            elif ablation_value == 'Meta_Expert':
                # W/o Meta_Expert

                is_activation_m1 = is_activation_area1(data_dir)
                llm_rag_all(data_dir, ablation_config)  # S1
                candidate_entity_retrieval(data_dir, is_activation_m1 = is_activation_m1)  # S2,S3

                is_activation_m2 = is_activation_area2(data_dir, src='M1')
                aligned_pairs = align_entities(data_dir, from_m3=False)  # LLM 1
                print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                run_full_process_s4(data_dir, "Simple-HHEA")
                run_full_process_llm2(data_dir, batch_size=500, ablation_config = ablation_config)  # LLM2

                # Step 3: Role assignment and knowledge enhancement
                is_activation_m3 = is_activation_area3(data_dir)
                run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = is_activation_m3, ablation_config = ablation_config)

                # Step 4: Check termination conditions and continue if necessary
                times = 1
                max_time = 3
                while is_activation_m2 or is_activation_m3:
                    is_activation_m2 = is_activation_area2(data_dir, src='M3')

                    aligned_pairs = align_entities(data_dir, from_m3=True)  # LLM 1
                    print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                    run_full_process_s4(data_dir, "Simple-HHEA")
                    run_full_process_llm2(data_dir, batch_size=500, ablation_config=ablation_config)  # LLM2

                    times = times + 1
                    is_activation_m3 = is_activation_area3(data_dir)
                    # Truncate iterations to reduce costs
                    if times >= max_time:
                        is_activation_m3 = False
                    run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = is_activation_m3, ablation_config = ablation_config)
                    if is_activation_m3 == False:
                        src = 'M3_false'
                        is_activation_m2 = is_activation_area2(data_dir, src)

            elif ablation_value == 'Communication':
                # W/o Communication

                is_activation_m1 = is_activation_area1(data_dir)
                llm_rag_all(data_dir, ablation_config)  # S1
                candidate_entity_retrieval(data_dir, is_activation_m1 = is_activation_m1)  # S2,S3

                is_activation_m2 = is_activation_area2(data_dir, src='M1')
                aligned_pairs = align_entities(data_dir, from_m3=False, ablation_config = ablation_config)  # LLM 1
                print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                run_full_process_s4(data_dir, "Simple-HHEA")
                run_full_process_llm2(data_dir, batch_size=500, ablation_config = ablation_config)  # LLM2

                # Step 3: Role assignment and knowledge enhancement
                is_activation_m3 = is_activation_area3(data_dir)
                run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = is_activation_m3, ablation_config = ablation_config)

                # Step 4: Check termination conditions and continue if necessary
                times = 1
                max_time = 3
                while is_activation_m2 or is_activation_m3:
                    is_activation_m2 = is_activation_area2(data_dir, src='M3')

                    aligned_pairs = align_entities(data_dir, from_m3=False, ablation_config = ablation_config)  # LLM 1
                    print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                    run_full_process_s4(data_dir, "Simple-HHEA")
                    run_full_process_llm2(data_dir, batch_size=500, ablation_config=ablation_config)  # LLM2

                    times = times + 1
                    is_activation_m3 = is_activation_area3(data_dir)
                    # Truncate iterations to reduce costs
                    if times >= max_time:
                        is_activation_m3 = False
                    run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = is_activation_m3, ablation_config = ablation_config)
                    if is_activation_m3 == False:
                        src = 'M3_false'
                        is_activation_m2 = is_activation_area2(data_dir, src)

            elif ablation_value == 'Communication':
                # W/o Communication

                is_activation_m1 = is_activation_area1(data_dir)
                llm_rag_all(data_dir, ablation_config)  # S1
                candidate_entity_retrieval(data_dir, is_activation_m1 = is_activation_m1)  # S2,S3

                is_activation_m2 = is_activation_area2(data_dir, src='M1')
                aligned_pairs = align_entities(data_dir, from_m3=False, ablation_config = ablation_config, no_optimization_tool = no_optimization_tool)  # LLM 1
                print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                run_full_process_s4(data_dir, "Simple-HHEA")
                run_full_process_llm2(data_dir, batch_size=500, ablation_config = ablation_config, no_optimization_tool = no_optimization_tool)  # LLM2

                # Step 3: Role assignment and knowledge enhancement
                is_activation_m3 = is_activation_area3(data_dir)
                run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = is_activation_m3, ablation_config = ablation_config, no_optimization_tool = no_optimization_tool)

                # Step 4: Check termination conditions and continue if necessary
                times = 1
                max_time = 3
                while is_activation_m2 or is_activation_m3:
                    is_activation_m2 = is_activation_area2(data_dir, src='M3')

                    aligned_pairs = align_entities(data_dir, from_m3=False, ablation_config = ablation_config, no_optimization_tool = no_optimization_tool)  # LLM 1
                    print(f"Found {len(aligned_pairs)} aligned entity pairs.")

                    run_full_process_s4(data_dir, "Simple-HHEA")
                    run_full_process_llm2(data_dir, batch_size=500, ablation_config = ablation_config, no_optimization_tool = no_optimization_tool)  # LLM2

                    times = times + 1
                    is_activation_m3 = is_activation_area3(data_dir)
                    # Truncate iterations to reduce costs
                    if times >= max_time:
                        is_activation_m3 = False
                    run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = is_activation_m3, ablation_config = ablation_config, no_optimization_tool = no_optimization_tool)
                    if is_activation_m3 == False:
                        src = 'M3_false'
                        is_activation_m2 = is_activation_area2(data_dir, src)






    else:
        # Original process flow with ablation-aware activation
        # Step 1: Candidate entity retrieval
        is_activation_m1 = is_activation_area1(data_dir)
        run_candidate_entity_retrieval(data_dir, is_activation_m1)

        # Step 2: Labeling and model training
        is_activation_m2 = is_activation_area2(data_dir, src='M1')
        run_labeling_and_model_training(data_dir, False, is_activation_m2)

        # Step 3: Role assignment and knowledge enhancement
        is_activation_m3 = is_activation_area3(data_dir)
        run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = is_activation_m3)

        # Step 4: Check termination conditions and continue if necessary
        times = 1
        max_time = 3
        while is_activation_m2 or is_activation_m3:
            is_activation_m2 = is_activation_area2(data_dir, src='M3')
            run_labeling_and_model_training(data_dir, True, is_activation_m2)
            times = times + 1
            is_activation_m3 = is_activation_area3(data_dir)
            # Truncate iterations to reduce costs
            if times >= max_time:
                is_activation_m3 = False
            run_role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = is_activation_m3)
            if is_activation_m3 == False:
                src = 'M3_false'
                is_activation_m2 = is_activation_area2(data_dir, src)

    # Record total time taken
    end_time = time.time()
    total_seconds = end_time - start_time

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    print(f"Final Process completed in: {end_time - start_time:.2f} seconds")
    print(f'Time Cost : {hours}hour, {minutes:02d}min, {seconds:02d}sec')
    print(f'Tokens Cost : {tokens_cal.global_tokens}')


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="icews_wiki")

    # Add ablation arguments
    parser.add_argument("--ablation1", choices=ABLATION_CATEGORIES['ablation1'],
                        help="Ablation 1: LLM Agents + single small model agent")
    parser.add_argument("--ablation2", choices=ABLATION_CATEGORIES['ablation2'],
                        help="Ablation 2: Single LLM + small model agent")
    parser.add_argument("--ablation3", choices=ABLATION_CATEGORIES['ablation3'],
                        help="Ablation 3: Remove specific agent")
    parser.add_argument("--ablation4", choices=ABLATION_CATEGORIES['ablation4'],
                        help="Ablation 4: Remove specific area")
    parser.add_argument("--ablation5", choices=ABLATION_CATEGORIES['ablation5'],
                        help="Ablation 5: Remove key component")

    parser.add_argument("--no_optimization_tool", action="store_true",
                        help="Run without optimization tools")

    args = parser.parse_args()
    data_dir = os.path.join("/home/dex/Desktop/entity_sy/AdaCoAgent/data", args.data)

    # Validate ablation arguments
    ablation_config = validate_ablation_args(args)

    # Run the full process with ablation configuration
    run_full_process(data_dir, ablation_config=ablation_config, no_optimization_tool=args.no_optimization_tool)


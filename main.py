import os
import time
import argparse
from modules import candidate_entity_retrieval, labeling_and_model_training, role_assignment_and_knowledge_enhancement
from LLM_rag import load_ents, retrieve_top_k_entities, retriever  # Import functions from LLM_rag.py
import openai
# OpenAI API configuration
os.environ["OPENAI_API_BASE"] = 'your base'
os.environ["OPENAI_API_KEY"] = "your key"

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to query GPT-3.5 model
def LLM_query(prompt, model="gpt-3.5-turbo"):
    """Query LLM model and get response."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']


# Function to run candidate entity retrieval process
def run_candidate_entity_retrieval(data_dir):
    """Run candidate entity retrieval process."""
    print("Running candidate entity retrieval...")
    candidate_entity_retrieval(data_dir)


# Function to run labeling and model training process
def run_labeling_and_model_training(data_dir):
    """Run labeling and model training process."""
    print("Running labeling and model training...")
    labeling_and_model_training(data_dir)


# Function to run role assignment and knowledge enhancement process
def run_role_assignment_and_knowledge_enhancement(data_dir):
    """Run role assignment and knowledge enhancement process."""
    print("Running role assignment and knowledge enhancement...")
    role_assignment_and_knowledge_enhancement(data_dir)


# Full process execution
def run_full_process(data_dir, language="zh_en"):
    start_time = time.time()

    # Step 1: Candidate entity retrieval
    run_candidate_entity_retrieval(data_dir)

    # Step 2: Labeling and model training
    run_labeling_and_model_training(data_dir)

    # Step 3: Role assignment and knowledge enhancement
    run_role_assignment_and_knowledge_enhancement(data_dir)

    # Step 4: Check termination conditions and continue if necessary
    while not check_termination_conditions():
        run_labeling_and_model_training(data_dir)
        run_role_assignment_and_knowledge_enhancement(data_dir)

    # Record total time taken
    end_time = time.time()
    print(f"Process completed in: {end_time - start_time:.2f} seconds")

# Function to check if termination conditions are met
def check_termination_conditions():
    """Check if termination conditions are met."""
    # For now, it returns False (meaning process will not terminate)
    return False  


# Function to retrieve entities from FAISS
def retrieve_entities_from_faiss(query, retriever, k=5):
    """Retrieve top-k entities from FAISS."""
    try:
        top_k_answers = retrieve_top_k_entities(query, retriever, k)
        for idx, (top_answer_id, top_answer_name) in enumerate(top_k_answers):
            print(f"Top {idx + 1}: {top_answer_id} - {top_answer_name}")
    except Exception as e:
        print(f"Error retrieving entities for query '{query}': {e}")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="icews_wiki")
    args = parser.parse_args()
    data_dir = os.path.join("/home/dex/Desktop/HHTEA/data", args.data)

    # Run the full process
    run_full_process(data_dir)

    # Load entities from FAISS
    ents_1 = load_ents('/root/autodl-tmp/data/icews_wiki/ent_ids_1')
    name2idx_1 = {v: k for k, v in ents_1.items()}

    # Example query to retrieve entities from FAISS
    query = "Example Entity"
    retrieve_entities_from_faiss(query, retriever, k=5)

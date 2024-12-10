# coding:GB312
import os
import time
import argparse
from modules import candidate_entity_retrieval, labeling_and_model_training, role_assignment_and_knowledge_enhancement
from LLM_rag import load_ents, retrieve_top_k_entities, retriever  # 碌录脠毛 LLM_rag.py 脰脨碌脛潞炉脢媒
import openai
# 脜盲脰脙 OpenAI API 脙脺脭驴
# 脡猫脰脙 OpenAI API 禄霉卤戮碌脴脰路潞脥 API 脙脺脭驴
os.environ["OPENAI_API_BASE"] = 'https://hk.xty.app/v1'
os.environ["OPENAI_API_KEY"] = "your-key"

# 脰卤陆脫脢鹿脫脙 OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# 碌梅脫脙 OpenAI GPT-3.5 脡煤鲁脡脦脛卤戮
def gpt3_query(prompt, model="gpt-3.5-turbo"):
    """碌梅脫脙 GPT-3.5 脌麓脡煤鲁脡脦脛卤戮"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']


# 脛拢驴茅1拢潞潞貌脩隆脢碌脤氓录矛脣梅
def run_candidate_entity_retrieval(data_dir):
    """脰麓脨脨脢碌脤氓录矛脣梅潞脥陆脷碌茫露脠脢媒路脰脦枚"""
    print("脮媒脭脷脰麓脨脨潞貌脩隆脢碌脤氓录矛脣梅...")
    candidate_entity_retrieval(data_dir)


# 脛拢驴茅2拢潞脨隆脛拢脨脥脩碌脕路潞脥卤锚脟漏脡煤鲁脡
def run_labeling_and_model_training(data_dir):
    """脰麓脨脨卤锚脟漏脡煤鲁脡潞脥脛拢脨脥脩碌脕路拢篓Simple-HHEA拢漏"""
    print("脮媒脭脷脰麓脨脨卤锚脟漏脡煤鲁脡潞脥脛拢脨脥脩碌脕路...")
    labeling_and_model_training(data_dir)


# 脛拢驴茅3拢潞陆脟脡芦路脰脜盲潞脥脰陋脢露脭枚脟驴
def run_role_assignment_and_knowledge_enhancement(data_dir):
    """脰麓脨脨陆脟脡芦路脰脜盲潞脥脰陋脢露脭枚脟驴"""
    print("脮媒脭脷脰麓脨脨陆脟脡芦路脰脜盲潞脥脰陋脢露脭枚脟驴...")
    role_assignment_and_knowledge_enhancement(data_dir)


# 脰梅脕梅鲁脤拢潞麓脫脢媒戮脻录炉麓娄脌铆碌陆脰陋脢露脥录脝脳露脭脝毛
def run_full_process(data_dir, language="zh_en"):
    start_time = time.time()

    # 碌脷脪禄虏陆拢潞潞貌脩隆脢碌脤氓录矛脣梅
    run_candidate_entity_retrieval(data_dir)

    # 碌脷露镁虏陆拢潞赂霉戮脻脛拢驴茅1碌脛脝脌鹿脌陆谩鹿没录陇禄卯脛拢驴茅2
    run_labeling_and_model_training(data_dir)

    # 碌脷脠媒虏陆拢潞陆脟脡芦路脰脜盲潞脥脰陋脢露脭枚脟驴
    run_role_assignment_and_knowledge_enhancement(data_dir)

    # 鲁脰脨酶陆禄脤忙脰麓脨脨脛拢驴茅2潞脥脛拢驴茅3
    while not check_termination_conditions():
        run_labeling_and_model_training(data_dir)
        run_role_assignment_and_knowledge_enhancement(data_dir)

    # 录脝脣茫脰麓脨脨脢卤录盲
    end_time = time.time()
    print(f"脰麓脨脨脢卤录盲: {end_time - start_time:.2f} 脙毛")


def check_termination_conditions():
    """录矛虏茅脢脟路帽脗煤脳茫脰脮脰鹿脤玫录镁"""
    # 脌媒脠莽拢潞录矛虏茅露脭脝毛戮芦露脠隆垄脨脭脛脺脝脌鹿脌脰赂卤锚碌脠
    # 路碌禄脴 True 禄貌 False
    return False  # 脕脵脢卤脡猫脰脙脦陋 False拢卢脨猫赂霉戮脻脢碌录脢脤玫录镁陆酶脨脨脢碌脧脰


# 脨脗脭枚碌脛鹿娄脛脺拢潞碌梅脫脙 LLM_rag.py 脰脨碌脛潞炉脢媒陆酶脨脨录矛脣梅虏脵脳梅
def retrieve_entities_from_faiss(query, retriever, k=5):
    """录矛脣梅脢碌脤氓"""
    try:
        top_k_answers = retrieve_top_k_entities(query, retriever, k)
        for idx, (top_answer_id, top_answer_name) in enumerate(top_k_answers):
            print(f"Top {idx + 1}: {top_answer_id} - {top_answer_name}")
    except Exception as e:
        print(f"Error retrieving entities for query '{query}': {e}")


if __name__ == "__main__":
    # 脙眉脕卯脨脨虏脦脢媒陆芒脦枚
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="icews_wiki")
    args = parser.parse_args()
    data_dir = os.path.join("/home/dex/Desktop/HHTEA/Simple-HHEA-main/data", args.data)

    # 脭脣脨脨脥锚脮没脕梅鲁脤
    run_full_process(data_dir)

    # 脢戮脌媒拢潞脠莽潞脦脢鹿脫脙 FAISS 录矛脣梅脢碌脤氓
    ents_1 = load_ents('/root/autodl-tmp/UEA-main/LLM/zero_shot_data/icews_wiki/ent_ids_1')
    name2idx_1 = {v: k for k, v in ents_1.items()}

    # 录脵脡猫 retriever 脪脩戮颅录脫脭脴虏垄脳录卤赂潞脙
    query = "Example Entity"
    retrieve_entities_from_faiss(query, retriever, k=5)

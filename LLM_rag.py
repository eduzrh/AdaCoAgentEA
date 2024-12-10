import os
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
import time
from tqdm import tqdm


def load_ents(path):
    """
    ¼ÓÔØÊµÌåÎÄ¼þ
    ²ÎÊý:
        path: ÊµÌåÎÄ¼þÂ·¾¶
    ·µ»Ø:
        data: ÊµÌå×Öµä
    """
    data = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            data[line[0]] = line[1]
    print(f'load {path} {len(data)}')
    return data


def retrieve_top_k_entities(query, retriever, k=10):
    """
    Ê¹ÓÃ FAISS ¼ìË÷¸ø¶¨²éÑ¯µÄ TOP-K ÊµÌå
    ²ÎÊý:
        query: ²éÑ¯ÊµÌåÃû³Æ
        retriever: ÓÃÓÚ¼ìË÷µÄÊµÀý
        k: ·µ»ØµÄºòÑ¡ÊµÌåÊýÁ¿
    ·µ»Ø:
        top_k_answers: TOP-K ×îÏà¹ØµÄÊµÌå
    """
    answers = retriever.invoke(query)
    answers_all = {}
    for doc in answers:
        doc1 = doc.page_content.strip().split('\t')
        answers_all[doc1[0]] = doc1[1].replace(' ', '')

    top_k_answers = sorted(answers_all.items(), key=lambda item: item[1], reverse=True)[:k]
    return top_k_answers


# ÅäÖÃ OpenAI API ÃÜÔ¿
os.environ["OPENAI_API_BASE"] = 'https://hk.xty.app/v1'
os.environ["OPENAI_API_KEY"] = "sk-rVzRDfCUBDmAw6IADb20Db2b51214eE5Bc6eCfEc2246E88a"

# ÎÄ¼þÂ·¾¶ÅäÖÃ
retriever_document_path = "/home/dex/Desktop/HHTEA/Simple-HHEA-main/data/icews_wiki/inrag_ent_ids_2_pre_embeding.txt"
faiss_index = "/home/dex/Desktop/HHTEA/Simple-HHEA-main/data/index/faiss_index_icews_wiki"
retriever_output_file = "/home/dex/Desktop/HHTEA/Simple-HHEA-main/data/icews_wiki/retriever_outputs.txt"

# ¼ÓÔØÊµÌå
ents_1 = load_ents('/home/dex/Desktop/HHTEA/Simple-HHEA-main/data/icews_wiki/ent_ids_1')
name2idx_1 = {v: k for k, v in ents_1.items()}
ents_2 = load_ents('/home/dex/Desktop/HHTEA/Simple-HHEA-main/data/icews_wiki/ent_ids_2')
name2idx_2 = {v: k for k, v in ents_2.items()}

# ¼ÓÔØÎÄµµ
loader = TextLoader(retriever_document_path)
raw_documents = loader.load()

# ³õÊ¼»¯ OpenAI Ç¶ÈëÄ£ÐÍ
embeddings = OpenAIEmbeddings()

# ¼ÓÔØ»ò´´½¨ FAISS ÏòÁ¿´æ´¢
if not os.path.exists(faiss_index):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(faiss_index)

# ¼ÓÔØ FAISS ÏòÁ¿´æ´¢
db = FAISS.load_local(faiss_index, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})

# Ö´ÐÐ²éÑ¯²¢±£´æ TOP-K ½á¹û
outputs_pair = ""
with open(retriever_output_file, 'w') as f, tqdm(total=len(ents_1.items()), desc="Processing queries") as pbar:
    for ent_id, ent_name in ents_1.items():
        query = ent_name
        try:
            top_k_answers = retrieve_top_k_entities(query, retriever, k=5)
            for idx, (top_answer_id, top_answer_name) in enumerate(top_k_answers):
                outputs_pair += f"{ent_id}\t{top_answer_id}\n"
            with open(retriever_output_file, 'w') as file:
                file.writelines(outputs_pair)
        except Exception as e:
            print(f"Error with entity {ent_name}: {str(e)}")
        pbar.update(1)  # Update progress bar

# ¼ÆËãÖ´ÐÐÊ±¼ä
end_time = time.time()
print(f"Retriever Execution time: {end_time - start_time:.2f} seconds")

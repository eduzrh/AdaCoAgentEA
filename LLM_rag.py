import os
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
import time
from tqdm import tqdm


def load_ents(path):
    """
    加载实体文件
    参数:
        path: 实体文件路径
    返回:
        data: 实体字典
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
    使用 FAISS 检索最相关的 TOP-K 实体
    参数:
        query: 查询实体名称
        retriever: 用于检索的对象
        k: 返回的前 k 个实体的数量
    返回:
        top_k_answers: TOP-K 排序后的实体
    """
    answers = retriever.invoke(query)
    answers_all = {}
    for doc in answers:
        doc1 = doc.page_content.strip().split('\t')
        answers_all[doc1[0]] = doc1[1].replace(' ', '')

    top_k_answers = sorted(answers_all.items(), key=lambda item: item[1], reverse=True)[:k]
    return top_k_answers


# 配置 OpenAI API 基础设置
os.environ["OPENAI_API_BASE"] = 'your base'
os.environ["OPENAI_API_KEY"] = "your key"

# 文件路径配置
retriever_document_path = "/home/dex/Desktop/HHTEA/Simple-HHEA-main/data/icews_wiki/inrag_ent_ids_2_subKG.txt"
faiss_index = "/home/dex/Desktop/HHTEA/Simple-HHEA-main/data/index/faiss_index_icews_wiki"
retriever_output_file = "/home/dex/Desktop/HHTEA/Simple-HHEA-main/data/icews_wiki/retriever_outputs.txt"

# 加载实体数据
ents_1 = load_ents('/home/dex/Desktop/HHTEA/Simple-HHEA-main/data/icews_wiki/ent_ids_1')
name2idx_1 = {v: k for k, v in ents_1.items()}
ents_2 = load_ents('/home/dex/Desktop/HHTEA/Simple-HHEA-main/data/icews_wiki/ent_ids_2')
name2idx_2 = {v: k for k, v in ents_2.items()}

# 加载文档
loader = TextLoader(retriever_document_path)
raw_documents = loader.load()

# 初始化 OpenAI 嵌入模型
embeddings = OpenAIEmbeddings()

# 初始化并保存 FAISS 索引
if not os.path.exists(faiss_index):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(faiss_index)

# 加载 FAISS 索引
db = FAISS.load_local(faiss_index, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})

# 处理查询并保存 TOP-K 结果
outputs_pair = ""
start_time = time.time()
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
        pbar.update(1)  # 更新进度条

# 输出执行时间
end_time = time.time()
print(f"Retriever Execution time: {end_time - start_time:.2f} seconds")

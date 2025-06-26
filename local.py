import re
import os
import json
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from sentence_transformers import CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from openai import OpenAI
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ----------------------
# 自动处理 JSON 文件模块
# ----------------------
class AutoJSONProcessor(BaseLoader):
    """
    自动处理 JSON 文件的加载器：
    - 递归提取 JSON 数据中所有字符串内容。
    - 将提取的文本拼接成整体，生成 Document 对象。
    """
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding

    @staticmethod
    def extract_text(data):
        texts = []
        if isinstance(data, str):
            texts.append(data)
        elif isinstance(data, dict):
            for value in data.values():
                texts.extend(AutoJSONProcessor.extract_text(value))
        elif isinstance(data, list):
            for item in data:
                texts.extend(AutoJSONProcessor.extract_text(item))
        return texts

    def load(self):
        with open(self.file_path, "r", encoding=self.encoding) as f:
            data = json.load(f)
        texts = self.extract_text(data)
        content = " ".join(texts).strip()
        documents = []
        if content:
            documents.append(Document(page_content=content, metadata={"source": self.file_path}))
        return documents


# 文件递归
class RecursiveLoader:
    def __init__(self, directory: str, file_extension: str = "*.json"):
        self.directory = directory
        self.file_extension = file_extension

    def load_files(self):
        # 使用 DirectoryLoader 来递归加载文件
        loader = DirectoryLoader(self.directory, glob=f"**/{self.file_extension}", loader_cls=AutoJSONProcessor)
        return loader.load()


# ----------------------
# 1. 智能分块预处理
# ----------------------
class SmartDocumentProcessor:
    def __init__(self):
        # 初始化嵌入模型，使用 HuggingFace 的 BAAI/bge-small-zh-v1.5 模型（专为 RAG 设计）
        model_local_path = os.path.abspath("../../models/bge-small-zh-v1.5")
        self.embed_model = HuggingFaceEmbeddings(
            model_name=model_local_path,
            model_kwargs={"local_files_only": True, "device": "cuda"},
            encode_kwargs={"batch_size": 16}
        )

    def _detect_content_type(self, text):
        """动态内容类型检测"""
        if re.search(r'def |import |print\(|代码示例', text):
            return "code"
        elif re.search(r'\|.+\|', text) and '%' in text:
            return "table"
        return "normal"

    def process_documents(self):
        # 加载文档：支持 PDF、TXT 和 JSON 文件
        loaders = [
            DirectoryLoader("../../knowledge_base", glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader("../../knowledge_base", glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}),
            DirectoryLoader("../../knowledge_base", glob="**/*.json", loader_cls=AutoJSONProcessor, loader_kwargs={"encoding": "utf-8"})
        ]
        documents = []
        for loader in loaders:
            documents.extend(loader.load())

        # 使用语义分块器对文档进行初步分块
        chunker = SemanticChunker(
            embeddings=self.embed_model,
            breakpoint_threshold_amount=82,
            add_start_index=True
        )
        base_chunks = chunker.split_documents(documents)

        # 二次动态分块，根据内容类型选择不同的分块参数
        final_chunks = []
        for chunk in base_chunks:
            content_type = self._detect_content_type(chunk.page_content)
            if content_type == "code":
                splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=64)
            elif content_type == "table":
                splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=96)
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
            final_chunks.extend(splitter.split_documents([chunk]))
        # 为每个块添加元数据
        for i, chunk in enumerate(final_chunks):
            chunk.metadata.update({
                "chunk_id": f"chunk_{i}",
                "content_type": self._detect_content_type(chunk.page_content)
            })
        return final_chunks


# ----------------------
# 2. 混合检索系统
# ----------------------
class HybridRetriever:
    def __init__(self, chunks):
        # 构建向量数据库，使用 Chroma 存储文档块，嵌入模型为 BAAI/bge-large-zh-v1.5
        self.vector_db = Chroma.from_documents(
            chunks,
            embedding=HuggingFaceEmbeddings(
                model_name="../../models/bge-large-zh-v1.5",
                model_kwargs={"local_files_only": True, "device": "cuda"}
            ),
            persist_directory="../../vector_db"
        )

        self.bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.vector_db.as_retriever(search_kwargs={"k": 5}),
                self.bm25_retriever
            ],
            weights=[0.6, 0.4]
        )
        self.reranker = CrossEncoder(
            "../../models/bge-reranker-large",
            device="cuda" if torch.backends.cuda.is_built() else "cpu"
        )

    def retrieve(self, query, top_k=3):
        docs = self.ensemble_retriever.invoke(query)  # 使用新方法 'invoke'
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:top_k]]


# ----------------------
# 3. RAG系统集成
# ----------------------
class EnhancedRAG:
    def __init__(self):
        # 文档预处理
        processor = SmartDocumentProcessor()
        chunks = processor.process_documents()
        self.retriever = HybridRetriever(chunks)

        # 初始化DeepSeek API客户端
        self.client = OpenAI(api_key="sk-996a6e8eeec94ce4be8d44acb3f40189",
                             base_url="https://api.deepseek.com")

        with open("../../banned_keywords.TXT", encoding='UTF-8') as f:
            self.banned_keywords = f.read().split()  # 违规关键词列表

    def is_geological_question(self, question):
        """判断是否为地质相关问题"""
        geo_keywords = ["岩体", "围岩", "隧道", "地质", "节理", "地下水", "开挖"]
        return any(keyword in question for keyword in geo_keywords)

    def generate_prompt(self, question, contexts):
        # 保留原有prompt生成逻辑
        context_str = "\n\n".join(
            [f"[来源：{doc.metadata['source']}，类型：{doc.metadata['content_type']}]\n{doc.page_content}"
             for doc in contexts])

        return f"""
当前上下文:
{context_str}

输入与输出示例如下：
示例一：开挖揭示围岩为细粒二长花岗岩，深灰色、褐灰色，弱风化为主，节理裂隙较发育，主要发育两组节理，①号主控节理N40°W/75°SW，②N50°W/70°NE，贯通性较好，结合差，岩体较破碎～破碎，围岩整体稳定性一般，掉块风险较高，掌子面左侧发育节理密集带，有蚀变现象，岩质软，手捏易碎，稳定性差，掉块严重，且坍塌风险高，掌子面湿润，渗水，左侧岩体呈线流状出水，拱顶淋雨状出水，超前探孔股状出水。
答案：
    工程地质信息:
        地层岩性:细粒二长花岗岩
        地质构造:节理裂隙较发育，主要发育两组节理，①号主控节理N40°W/75°SW，②N50°W/70°NE，贯通性较好，结合差，岩体较破碎～破碎，围岩整体稳定性一般，掉块风险较高，掌子面左侧发育节理密集带
        岩溶:有蚀变现象，岩质软，手捏易碎，稳定性差，掉块严重，且坍塌风险高
        特殊地层:无相关信息
        人为坑洞:无相关信息
        地应力:无相关信息
        塌方:无相关信息
        其他:无相关信息
    水文地质:
        地下水信息:掌子面湿润，渗水，左侧岩体呈线流状出水，拱顶淋雨状出水，超前探孔股状出水
        水质分析:无相关信息
        其他:无相关信息
    综合分析:
        岩石坚硬程度:坚硬岩
        理由:原岩为细粒二长花岗岩，属于坚硬岩，受到弱风化，所以为坚硬岩
        岩体完整程度:较破碎
        理由:主要结构面类型节理裂隙较发育，发育两组节理，贯通性较好，结合差，围岩整体稳定性一般，掉块风险较高
        地下水状态:Ⅱ
        理由:掌子面湿润，渗水，左侧岩体呈线流状出水，拱顶淋雨状出水，超前探孔股状出水
示例二：掌子面揭示围岩主要为二长花岗岩，灰黑色，局部夹深灰色、褐黄色，弱风化为主，局部夹强风化，节理裂隙较发育，延伸较好，局部节理面可见铁质侵染，主要节理产状：N70°E/70°NW、N30°W/55°SW，掌子面发育多条石英脉，宽5～10cm，岩体整体较完整，局部较破碎，有掉块剥落现象。掌子面整体湿润，局部呈线状和股状出水，单位渗水量30L/(min·10m) 。
答案：
    工程地质信息:
        地层岩性:二长花岗岩，灰黑色，局部夹深灰色、褐黄色
        地质构造:节理裂隙较发育，延伸较好，局部节理面可见铁质侵染，主要节理产状：N70°E/70°NW、N30°W/55°SW，掌子面发育多条石英脉，宽5～10cm，岩体整体较完整，局部较破碎，有掉块剥落现象
        岩溶:无相关信息
        特殊地层:无相关信息
        人为坑洞:无相关信息
        地应力:无相关信息
        塌方:无相关信息
        其他:无相关信息
    水文地质:
        地下水信息:掌子面整体湿润，局部呈线状和股状出水，单位渗水量30L/(min·10m)
        水质分析:无相关信息
        其他:无相关信息
    综合分析:
        岩石坚硬程度:坚硬岩
        理由:原岩为二长花岗岩，属于A类岩，硬度高，受到弱风化为主，局部强风化，所以整体为坚硬岩
        岩体完整程度:较完整
        理由:节理较发育，主要发育两组节理，故可能为较完整~较破碎；根据局部节理面可见铁质侵染，可以认为结构面结合程度一般；最后结合信息岩体整体较完整，局部较破碎可以认为岩体整体较完整
        地下水状态:Ⅱ
        理由:掌子面整体湿润，局部呈线状和股状出水，单位渗水量30L/(min·10m)
示例三：粗角砾土，褐黄色，角砾含量约65％，呈棱角状、次棱角状，局部夹少量碎石土,中密状，潮湿，砾石成分为花岗岩质。粒径 20～120mm，尖棱状，充填粉质黏土。掌子面无水，整体湿润，有掉块现象，围岩稳定性较差.
答案：
    土体信息：粗角砾土，褐黄色，角砾含量约65％，呈棱角状、次棱角状，局部夹少量碎石土,中密状，潮湿，砾石成分为花岗岩质。粒径 20～120mm，尖棱状，充填粉质黏土
    地下水信息:掌子面无水，整体湿润
    地质构造：有掉块现象，围岩稳定性较差
    综合分析：
        地下水状态:Ⅰ
        理由:掌子面无水，整体湿润

现在分析得到的信息:{question}
首先提取出相关模板信息，然后根据有关岩石坚硬程度、岩体完整程度、地下水状态的描述信息并给它们进行定性,定性时必须要给出理由且要有分析提取信息的过程。
注意：
1.如果用户提供信息和隧道，地质无关，请独立思考并回复，不需要考虑示例内容。
2.请务必按示例中的模板样例进行输出。       
3.如果上下文信息不足，请明确指出缺失的信息，并进行独立思考。最后用中文给出结构化json文件形式答案。
"""

    def contains_banned_content(self, text):
        """检测违规内容"""
        return any(keyword in text for keyword in self.banned_keywords)

    def ask(self, question):
        # 检测违规内容
        if self.contains_banned_content(question):
            return "⚠️ 该内容涉及违规话题，无法回答"

        # 非地质问题直接回答
        if not self.is_geological_question(question):
            messages = [
                {"role": "system", "content": "你是一个友好的助手"},
                {"role": "user", "content": question}
            ]
        else:
            # 地质问题处理流程
            contexts = self.retriever.retrieve(question)
            prompt = self.generate_prompt(question, contexts)
            messages = [
                {"role": "system", "content": "你是一个专业地质领域助手"},
                {"role": "user", "content": prompt},
            ]

        # 调用API
        response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            max_tokens=8000,
            stream=False
        )

        # 获取回答
        answer = response.choices[0].message.content
        return answer


# 修改后的主程序
if __name__ == "__main__":
    rag = EnhancedRAG()
    print("输入问题开始对话（输入'quit'退出）")

    while True:
        try:
            user_input = input("\n>>> 用户: ").strip()
            if user_input.lower() == 'quit':
                print("\n对话结束")
                break

            if not user_input:
                print("问题不能为空")
                continue

            print("\n🤖 正在思考...", end="", flush=True)
            answer = rag.ask(user_input)
            print("\r" + " " * 20 + "\r", end="")
            print("\n💡 系统回答:")
            print("-" * 50)
            print(answer)
            print("-" * 50)

        except Exception as e:
            print(f"\n⚠️ 错误: {str(e)}")


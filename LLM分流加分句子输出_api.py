from openai import OpenAI
import json
import re
import os
# 其余依赖（如 sentence_transformers, langchain 等）和原来保持一致
import torch
from unsloth import FastLanguageModel
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def _call_deepseek_api(prompt: str ,api_key:str) -> str:
    """
    使用 DeepSeek API 来完成对话补全，返回模型生成的文本。
    这里的 prompt 可以是系统+用户的组合，也可以只包含用户部分，
    根据实际需要自行调整。
    """
    # 实例化客户端
    # api_key 和 base_url 替换为你在 DeepSeek 上的配置
    client = OpenAI(
        api_key= api_key,
        base_url="https://api.deepseek.com"
    )

    # 调用 chat.completions 接口
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一名专业地质领域AI"},
                {"role": "user", "content": prompt},
            ],
            stream=False  # 如果需要流式推送，可改为 True
        )
        # 返回生成结果
        return response.choices[0].message.content
    except Exception as e:
        return f"DeepSeek API 调用异常: {str(e)}"

class AutoJSONProcessor:
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


class RecursiveLoader:
    def __init__(self, directory: str, file_extension: str = "*.json"):
        self.directory = directory
        self.file_extension = file_extension

    def load_files(self):
        loader = DirectoryLoader(self.directory, glob=f"**/{self.file_extension}", loader_cls=AutoJSONProcessor)
        return loader.load()


class SmartDocumentProcessor:
    def __init__(self):
        # 这里保留 Embedding 模型（使用本地 bge-small-zh-v1.5），供 SemanticChunker 用
        model_local_path = os.path.abspath("../../models/bge-small-zh-v1.5")
        self.embed_model = HuggingFaceEmbeddings(
            model_name=model_local_path,
            model_kwargs={"local_files_only": True, "device": "cuda"},
            encode_kwargs={"batch_size": 16}
        )

    def _detect_content_type(self, text):
        if re.search(r'def |import |print\(|代码示例', text):
            return "code"
        elif re.search(r'\|.+\|', text) and '%' in text:
            return "table"
        return "normal"

    def process_documents(self):
        loaders = [
            DirectoryLoader("../../knowledge_base", glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader("../../knowledge_base", glob="**/*.txt", loader_cls=TextLoader,
                            loader_kwargs={"encoding": "utf-8"}),
            DirectoryLoader("../../knowledge_base", glob="**/*.json", loader_cls=AutoJSONProcessor,
                            loader_kwargs={"encoding": "utf-8"})
        ]
        documents = []
        for loader in loaders:
            documents.extend(loader.load())

        # 先做语义级别的分块
        chunker = SemanticChunker(
            embeddings=self.embed_model,
            breakpoint_threshold_amount=82,
            add_start_index=True
        )
        base_chunks = chunker.split_documents(documents)

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

        for i, chunk in enumerate(final_chunks):
            chunk.metadata.update({
                "chunk_id": f"chunk_{i}",
                "content_type": self._detect_content_type(chunk.page_content)
            })
        return final_chunks


class HybridRetriever:
    def __init__(self, chunks):
        # 构建向量数据库
        self.vector_db = Chroma.from_documents(
            chunks,
            embedding=HuggingFaceEmbeddings(
                model_name="../../models/bge-large-zh-v1.5",
                model_kwargs={"local_files_only": True, "device": "cuda"}
            ),
            persist_directory="../../vector_db"
        )

        # BM25
        self.bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
        # Ensemble: 向量检索 + BM25
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.vector_db.as_retriever(search_kwargs={"k": 5}),
                self.bm25_retriever
            ],
            weights=[0.6, 0.4]
        )

        # CrossEncoder 交互式重排
        self.reranker = CrossEncoder("../../models/bge-reranker-large", device="cuda")

    def retrieve(self, query, top_k=5):
        docs = self.ensemble_retriever.invoke(query)
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:top_k]]

class EnhancedRAG:
    def __init__(self):
        """
        参数:
          api_key: DeepSeek API key
        """
        # 第一步：构建文档检索器
        processor = SmartDocumentProcessor()
        chunks = processor.process_documents()
        self.retriever = HybridRetriever(chunks)

        # 如果有违禁词文件，就加载进来；没有就空置
        try:
            with open("../../banned_keywords.TXT", "r", encoding="utf-8") as f:
                self.banned_keywords = f.read().split()
        except FileNotFoundError:
            self.banned_keywords = []

        # 保存 DeepSeek API key，供后续调用
        self.api_key = "sk-996a6e8eeec94ce4be8d44acb3f40189"

    def is_geological_question(self, user_input):
        """判断是否为地质相关问题"""
        geo_keywords = ["岩体", "围岩", "隧道", "地质", "节理", "地下水", "开挖", "掌子面"]
        return any(keyword in user_input for keyword in geo_keywords)

    def contains_banned_content(self, text: str) -> bool:
        return any(keyword in text for keyword in self.banned_keywords)

    def parse_input_to_categories(self, user_text: str, type: str) -> dict:

        # 构造提示词，要求模型分类+融合每类句子为一段话
        prompt = f"""
    请对如下信息按分类方法进行理解分类：{user_text}
    要求选出你认为和{type}最相关的句子，并合并为一句话。

    分类包括以下三类：
    1. 岩石坚硬程度）：判断标准：岩石坚硬程度由岩性、风化作用、水作用三者共同决定。
    2. 岩体完整程度判断标准：结构面发育程度、结合程度、结构面类型、岩体结构类型。
    3. 地下水判断标准：句子中涉及渗水、出水、潮湿、湿润、滴水等描述。

    注意事项：
    - 每个句子可以属于多个类别；
    - 每一类请整合并润色所有相关句子，形成一段完整表达；
    """
        # 调用模型
        try:
            answer = _call_deepseek_api(prompt, self.api_key)
            return answer
        except Exception as e:
            return f"生成回答时发生错误: {str(e)}"

    def _call_model_for_category(self, category_text: str, category_name: str, extra_instructions: str) -> str:
        """
        通用的函数：给定用户文本(某一类) + 检索到的上下文 + 特定格式提示，
        用 DeepSeek API 获取结果。
        """
        if not category_text.strip():
            return "（无相关信息）"

        # 1) 用用户文本拼出检索 query
        category_query = f"{category_text}\n与{category_name}相关"

        # 2) 检索
        retrieved_docs = self.retriever.retrieve(category_query, top_k=3)
        context_str = "\n\n".join(
            [f"[来源：{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in retrieved_docs]
        )

        # 3) 构造 Prompt
        prompt = f"""
你是一名专业地质领域AI。以下是用户拆分出的与【{category_name}】相关信息：
{category_text}

这是检索到的参考资料：
{context_str}

现在请你对【{category_name}】作出一个专业判断，并说明理由。
请务必输出以下JSON格式，形如：
{{
  "（判定对象）": "这里写最后结论",
  "理由": "这里写依据相关信息和参考资料的思考过程"
}}
以下内容为注意事项：
{extra_instructions}
        """

        # 4) 调用 DeepSeek API 完成生成
        answer = _call_deepseek_api(prompt, self.api_key)
        return answer

    def analyze_and_stitch(self, user_text: str) -> list:
        rock_text = self.parse_input_to_categories(user_text, '岩石坚硬程度')
        integrity_text = self.parse_input_to_categories(user_text, "岩体完整程度")
        water_text = self.parse_input_to_categories(user_text, "地下水")

        # 分别给出三种场景下的不同提示要求
        rock_ans = self._call_model_for_category(
            rock_text,
            "岩石坚硬程度",
            """
            1.岩石坚硬程度由岩性、风化作用和水作用三者共同决定；2.理由要充分结合资料；3.只输出结构化结果。
            输出结果在如下中选择根据结果选择对应的数字，只选一项不要范围：
            坚硬岩：1；较坚硬岩：2；较软岩：3；软岩：4；极软岩：5；
            """
        )
        integrity_ans = self._call_model_for_category(
            integrity_text,
            "岩体完整程度",
            """
            1.岩体完整度主要看结构面发育程度；2.理由要充分结合资料；3.只输出结构化结果。
            输出结果在如下中选择根据结果选择对应的数字，只选一项不要范围：
            完整：1；较完整：2；较破碎：3；破碎：4；极破碎：5；
            """
        )
        water_ans = self._call_model_for_category(
            water_text,
            "地下水",
            "1.地下水状态的判定结果要为Ⅰ,Ⅱ等等级并将罗马数字换成阿拉伯数字输出；2.理由要充分结合资料；3.只输出结构化结果。"
        )

        # 简易提取函数：从文本中找出 JSON 块
        def extract_json_list(text: str) -> list:
            json_blocks = re.findall(r'({.*?})', text, re.DOTALL)
            json_list = []
            for block in json_blocks:
                try:
                    json_obj = json.loads(block)
                    json_list.append(json_obj)
                except json.JSONDecodeError:
                    pass
            return json_list

        # 最后统一拼到一起，再尝试解析所有 JSON
        final_result = f"{rock_ans}\n{integrity_ans}\n{water_ans}"
        return extract_json_list(final_result)

    def ask(self, user_input: str) -> str:
        # 违禁词检测
        if self.contains_banned_content(user_input):
            return "⚠️ 该内容涉及违规话题，无法回答"
        if not self.is_geological_question(user_input):
            prompt = f"""
        你是一个通用智能助手。请以简洁、清晰、友好的方式回答下面的问题：
        {user_input}
        """
            answer = _call_deepseek_api(prompt, self.api_key)
            return answer
        else:
            # 调用三分类 + 检索 + DeepSeek API
            result = self.analyze_and_stitch(user_input)
            return json.dumps(result, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    rag = EnhancedRAG()

    print("输入问题开始对话（输入'quit'退出）")
    while True:
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

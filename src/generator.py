import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from src.retriever import retrieve_docs
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()
import os


# ✅ With input_variables
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="You are a helpful travel assistant.\nUse the following travel guide context to answer the question.\nIf the answer is not found, say you don't know.\nContext:\n{context}\nQuestion:\n{question}\nAnswer:"
)


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    # ✅ Correct variable name matching your .env
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    task="text-generation",
    max_new_tokens=256,
)

chat_model = ChatHuggingFace(llm=llm)
qa_chain = prompt_template | chat_model

def generate_answer(query: str) -> str:
    docs = retrieve_docs(query)
    context = "\n".join(docs)
    result = qa_chain.invoke({"context": context, "question": query})
    return result.content
from fastapi import UploadFile
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import COLLECTION_NAME
from src.embeddings import get_embeddings
from src.vectorstores import get_qdrant_client
from qdrant_client.models import PointStruct
import uuid

async def ingest_pdf(file: UploadFile):
    print(f"Processing PDF file: {file.filename}")
    content = await file.read()
    docs = []
    pdf = fitz.open(stream=content, filetype="pdf")
    page_count = len(pdf)
    try:
        for page_num in range(page_count):
            page = pdf[page_num]
            text = page.get_text()
            docs.append(Document(
                page_content=text,
                metadata={"page": page_num, "source": file.filename}
            ))
    finally:
        pdf.close()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = get_embeddings(texts)
    client = get_qdrant_client()
    payloads = [{"text": chunk.page_content, **chunk.metadata} for chunk in chunks]
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload=payloads[i]
        )
        for i in range(len(chunks))
    ]
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"Upload complete! Added {len(chunks)} chunks from {page_count} pages")
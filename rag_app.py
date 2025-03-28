import pandas as pd
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

embeddings=OllamaEmbeddings(model='llama3.1')
vstore = InMemoryVectorStore(embeddings)

pdf_loader = PyPDFLoader("./aiswre/data/IEC_62304-2006.pdf")
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
docs_from_pdf = pdf_loader.load_and_split(text_splitter=splitter)

print(f"Documents from PDF: {len(docs_from_pdf)}.")
#print(docs_from_pdf[0])
inserted_ids_from_pdf = vstore.add_documents(docs_from_pdf)
print(f"Inserted {len(inserted_ids_from_pdf)} documents.")
#print(inserted_ids_from_pdf[0])
df = pd.DataFrame({
    'inserted_id': [i for i in inserted_ids_from_pdf],
    'chunk':[d.page_content for d in docs_from_pdf]
})
df.head(5).to_excel("./aiswre/data/IEC_62304-2006_sample_chunks.pdf")

retriever = vstore.as_retriever(search_kwargs={"k": 3})

excerpt="""
IEC 62304:2006 defines in section 4.3 the software safety classes, based only on the consequence of a hazardous situation on the patient:

Class A: No injury or damage to health is possible
Class B: Non-SERIOUS INJURY is possible
Class C: Death or SERIOUS INJURY is possible
Another way of viewing this definition is to disregard the probability of risks linked to a software failure, and to focus only on the severity.
"""

retrieved_docs = vstore.similarity_search(excerpt)

print(retreived_docs)

quality_template = """
You are a software quality engineer currently reviewing medical device documentation and need to provide feedback on how well the documentation aligns with compliance standards. Use the provided context as to support your answers and do not make anything up.

CONTEXT:
{context}

QUESTION: {question}

YOUR ANSWER:"""

quality_prompt = ChatPromptTemplate.from_template(quality_template)

llm = ChatOllama(model='llama3.1')

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | quality_prompt
    | llm
    | StrOutputParser()
)

q="""
What is the recommended approach to assigning a Software Safety Classification?  
"""
result = chain.invoke(q)
print(result)
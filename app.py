from rich import print
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI()

def rag(chunks, collection_name) :
    vectorstore= Chroma.from_documents(
        documents=documents ,
        collection_name=collection_name,
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"))
    retriever = vectorstore.as_retriever()
    prompt_template = """Answer the question based only on the following context:
    {context}
    Question: {question} 
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context" : retriever, "question" : RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    result = chain.invoke("What is the use of Text Splitting?")
    print(result)

# 1. Character Text Splitting
print('### Character Text Splitting')

text = "Text splitting in Langchain is a criticial features athat allows you to split text into characters. This is useful for many applications such as text generation, text classification, and text summarization."

# Manual Splitting

chunks = []
chunk_size = 35 #Characers

for i in range(0, len(text), chunk_size):
    chunk = text[i:i+chunk_size]
    chunks.append(chunk)    

documents = [Document(page_content=chunk, metadata={"source": "local"}) for chunk in chunks]
print(documents)

# Automatic Text Splitting

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=35, chunk_overlap=0, separator='', strip_whitespace=False) 
documents = text_splitter.create_documents([text])
print(documents)

# 2 .Recursive Character Text Splitting
print('### Recursive Character Text Splitting')

from langchain.text_splitter import RecursiveCharacterTextSplitter

with open('content.txt', 'r', encoding='utf-8') as file:
    text = file.read()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=0)#, separator='', strip_whitespace=False)
documents = text_splitter.create_documents([text])
print(documents)

# 3. Document Text Splitting
print('### Document Text Splitting')

# Document Specific Splitting - Markdown

from langchain.text_splitter import MarkdownTextSplitter

splitter = MarkdownTextSplitter(chunk_size=450, chunk_overlap=0)
markdown_text = """
# Fun in California

## Driving

Try to avoid driving in California. The traffic is terrible.

### Food

Make sure to eat a burrito in San Francisco. It's the best.

## Hiking

Go to Yosemite for some amazing hikes.
"""

print(splitter.create_documents([markdown_text]))

# Document Specific Splitting - Python
from langchain.text_splitter import PythonCodeTextSplitter
python_text = """
Class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
p = Person('John', 36)
for i in range(10):
    print(i)
"""

python_splitter = PythonCodeTextSplitter(chunk_size=450, chunk_overlap=0)
print(python_splitter.create_documents([python_text]))

# Document Specific Splitting - SQL

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

sql_text = """
select * from users join
orders on users.id = orders.user_id
where users.name = 'John'
group by users.id
"""

sql_splitter = RecursiveCharacterTextSplitter.from_language(Language.SOL, chunk_size=450, chunk_overlap=0)

print(sql_splitter.create_documents([sql_text]))

# 4. Semantic Text Splitting

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

text_splitter = SemanticChunker(
    OpenAIEmbeddings(model="text-embedding-3-large"), breakpoint_threshold_type="percentile"
)

documents = text_splitter.create_documents([text])
print(documents)

# 5. Agentic Chunking

print("### Agentic Chunking")
print("### Proposition-based Chunking")

# https://arxiv.org/pdf/2312.06648.pdf

from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core. runnables import RunnableLambda
from langchain. chains import create_extraction_chain
from typing import Optional, List
from langchain. chains import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel
from langchain import hub


prompt_template = hub.pull("wfh/proposal-indexing")
llm = ChatOpenAI()

runnable = prompt_template | llm 

class Sentences(BaseModel):
    sentences: List[str]

extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm)

def get_propositions(text):
    runnable_output = extraction_chain.invoke({"input" : text})
    propositions = extraction_chain.invoke(runnable_output)["text"][0].sentences
    return propositions
    
paragraph = text.split("\n\n")

text_propositions = []

for i, para in enumerate(paragraph[:5]):
    propositions = get_propositions(para)
    text_propositions.extend(propositions)
    print(f"Done with {i}")

print(f"You haven {len(text_propositions)} propositions")
print(text_propositions[:10])

print("#### Group Chunk")

from agentic_chunker import AgenticChunker
ac = AgenticChunker()
ac.add_propositions(text_propositions)
print(ac.pretty_print_chunks())
chunks = ac.get_chunks(get_type="list_of_strings")
print(chunks)

documents = [Document(page_content=chunk, metadata={"source": "local"}) for chunk in chunks]
rag(documents, "agentic_chunks")
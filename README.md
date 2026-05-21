
# Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

## AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

## PROBLEM STATEMENT:
The challenge is to develop an agent that can efficiently retrieve and synthesize information from a large corpus of documents, ensuring that it answers queries with precision and relevance, leveraging LlamaIndex for effective retrieval and summarization.

## DESIGN STEPS:
Gather a set of research articles or documents relevant to the topic.
Preprocess the data by converting the articles into a suitable format (e.g., plain text or structured format).
Tokenize the content and remove any irrelevant or noisy information.

### STEP 1:
Use LlamaIndex (formerly known as GPT Index) to create an index for the documents.
LlamaIndex will help build an optimized index for efficient retrieval, making it easy to query multiple documents at once.
Incorporate features like semantic search to improve relevance and accuracy of retrieval.

### STEP 2:
Develop the query interface where users can input questions related to the research articles.
Integrate the query interface with the LlamaIndex-powered retrieval system.
Process the retrieved documents to extract relevant information and synthesize a concise response, potentially using additional techniques like summarization.

### STEP 3:
Test the system with a range of diverse queries to evaluate its performance in terms of accuracy, relevance, and conciseness of responses.
Collect feedback and refine the system based on test results.

### PROGRAM:
```
import os
import requests
import nest_asyncio
from pathlib import Path
nest_asyncio.apply()
# OPENAI API KEY
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# OPENREVIEW PDF URLS
urls = [
    "https://openreview.net/pdf?id=6xnZ048nAM",
    "https://openreview.net/pdf?id=IQGxkTCJmt",
    "https://openreview.net/pdf?id=Sa6ivpFokl",
]
# CREATE PAPERS DIRECTORY
pdf_dir = Path("papers")
pdf_dir.mkdir(exist_ok=True)
# DOWNLOAD PDFs
papers = []
for i, url in enumerate(urls, start=1):
    pdf_path = pdf_dir / f"{i}pdf.pdf"
    print(f"\nDownloading {url}")
    response = requests.get(url)
    if response.status_code == 200:
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        print(f"Saved: {pdf_path}")
        papers.append(str(pdf_path))
    else:
        print(f"Failed to download: {url}")
# VERIFY FILES
print("\nVERIFYING FILES:\n")
for paper in papers:
    path = Path(paper)
    print(
        f"{path} -> Exists: {path.exists()}"
    )
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleDirectoryReader
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import (
    FunctionCallingAgentWorker,
    AgentRunner
)
def get_doc_tools(file_path, name):
    print(f"\nLoading document: {file_path}")
    # Load document
    documents = SimpleDirectoryReader(
        input_files=[file_path]
    ).load_data()
    # Split into chunks
    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=100
    )
    nodes = splitter.get_nodes_from_documents(
        documents
    )
    # Create vector index
    vector_index = VectorStoreIndex(nodes)
    # Create summary index
    summary_index = SummaryIndex(nodes)
    # Query engines
    vector_query_engine = vector_index.as_query_engine()
    summary_query_engine = summary_index.as_query_engine()
    # Vector tool
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        name=f"vector_tool_{name}",
        description=f"Useful for detailed questions about {name}"
    )
    # Summary tool
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        name=f"summary_tool_{name}",
        description=f"Useful for summarizing {name}"
    )

    return vector_tool, summary_tool
paper_to_tools_dict = {}
for paper in papers:
    print(f"\nCreating tools for: {paper}")
    vector_tool, summary_tool = get_doc_tools(
        paper,
        Path(paper).stem
    )
    paper_to_tools_dict[paper] = [
        vector_tool,
        summary_tool
    ]
all_tools = [
    tool
    for paper in papers
    for tool in paper_to_tools_dict[paper]
]

print(f"\nTotal tools created: {len(all_tools)}")
llm = OpenAI(
    model="gpt-3.5-turbo-0125"
)
agent_worker = FunctionCallingAgentWorker.from_tools(
    all_tools,
    llm=llm,
    verbose=True
)

agent = AgentRunner(agent_worker)
print("\nAgent created successfully!")
response = agent.query(
    "Give me a summary of both 1pdf and 2pdf"
)
print("\n================ SUMMARY RESPONSE ================\n")
print(response)
response = agent.query(
    "Tell me about the evaluation dataset used in the papers"
)

print("\n================ EVALUATION RESPONSE ================\n")
print(response)
response = agent.query(
    "Compare the methodologies used in all papers"
)
print("\n================ COMPARISON RESPONSE ================\n")
print(response)

```
### OUTPUT:
#### 3 paper model
<img width="1920" height="923" alt="image" src="https://github.com/user-attachments/assets/ee5895d6-3e13-41bf-ae4a-8678b722e999" />

#### 5 paper model
<img width="1920" height="809" alt="image" src="https://github.com/user-attachments/assets/c3a5c06e-90d5-4c94-9357-7f2e98d949d4" />



### RESULT:
Thus the completed successfully  Multidocument Retrieval Agent Using LlamaIndex

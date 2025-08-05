import os
from neo4j import GraphDatabase
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from neo4j.exceptions import ServiceUnavailable, AuthError

# Replace with your actual Neo4j connection details
uri = ""
user = "neo4j"
password = ""

# Set the OpenAI API key and endpoint
os.environ["AZURE_OPENAI_API_KEY"]  = ''
os.environ["AZURE_OPENAI_ENDPOINT"]  = ""
api_version = '2024-02-01'

# Initialize the Azure OpenAI embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="embedding_func",
    api_key=os.environ["AZURE_OPENAI_API_KEY"] ,
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=api_version,
)

def create_node(tx, node):
    query = (
        "MERGE (n:{type} {{id: $id}}) "
        "SET n += $properties "
        "RETURN n"
    ).format(type=node['type'])
    tx.run(query, id=node['id'], properties=node.get('properties', {}))

def create_relationship(tx, relationship):
    query = (
        "MATCH (a {{id: $source_id}}), (b {{id: $target_id}}) "
        "MERGE (a)-[r:{type}]->(b) "
        "RETURN r"
    ).format(type=relationship['type'])
    tx.run(query, source_id=relationship['source'], target_id=relationship['target'])

llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo", api_key='sk-proj-d5f9qTtNCRUIEG4vo8t7T3BlbkFJZvPY4zDZURNw2YxM66UT')
llm_transformer = LLMGraphTransformer(llm=llm)

text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris.
"""
documents = [Document(page_content=text)]

llm_transformer_filtered = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Country", "Organization"],
    allowed_relationships=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"],
)
graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(documents)

parsed_nodes = [{"id": node.id, "type": node.type} for node in graph_documents_filtered[0].nodes]
parsed_relationships = [{"source": rel.source.id, "target": rel.target.id, "type": rel.type} for rel in graph_documents_filtered[0].relationships]

# Print the parsed nodes and relationships to verify
print("Parsed Nodes:")
print(parsed_nodes)
print("Parsed Relationships:")
print(parsed_relationships)

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        for node in parsed_nodes:
            session.execute_write(create_node, node)
        
        for relationship in parsed_relationships:
            session.execute_write(create_relationship, relationship)

except ServiceUnavailable as e:
    print(f"Service unavailable: {e}")
except AuthError as e:
    print(f"Authentication error: {e}")
finally:
    if driver:
        driver.close()

vector_index = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url=uri,
    username=user,
    password=password,
    index_name='tasks',
    node_label="Task",
    text_node_properties=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"],
    embedding_node_property="embedding"
)

# Query the vector index
query_text = "Who is Pierre Curie?"
query_embedding = embeddings.embed_query(query_text)

# Perform similarity search
results = vector_index.similarity_search(query_text, k=1)
print(results)

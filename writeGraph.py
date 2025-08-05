import os
from langchain_community.callbacks import get_openai_callback
from langchain_openai import AzureChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.llms import OpenAI
from neo4j import GraphDatabase
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.vectorstores import Neo4jVector
from neo4j.exceptions import ServiceUnavailable, AuthError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

#from langchain.chat_models import ChatOpenAI

import PyPDF2
#Local

# Replace with your actual Neo4j connection details
uri = "" #Neo4j uri
user = "neo4j" #neo4j user
password = "" # neo4j password

os.environ["NEO4J_URI"] = uri
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = password
#graph = Neo4jGraph() #this may need authentitcating


# Set the OpenAI API key and endpoint
os.environ["AZURE_OPENAI_API_KEY"] = 'ccef62baeacd480a8efd577c9ee5b50c'
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://muse-mvp.openai.azure.com/"
api_version = '2024-02-01'
llm = AzureChatOpenAI(
    azure_deployment="gpt40-mini-dynnamicQs",
    api_version="2024-06-01",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# Initialize the Azure OpenAI embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="embedding_func",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=api_version,
)
graph = Neo4jGraph(url= os.environ["NEO4J_URI"], username = os.environ["NEO4J_USERNAME"], password= os.environ["NEO4J_PASSWORD"] )
# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

def pdf_to_text(pdf_path): #PDF to text
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        # Iterate through each page
        for page in reader.pages:
            text += page.extract_text()
    return text

#Gets all the nodes and relationships for the program in an organized way
def get_dynamic_nodes(story_id):
    queryStart =f''' // Step 1: Retrieve all nodes with specific labels and 'dynamic' property set to true
    MATCH (n)
    WHERE any(label in labels(n) WHERE label IN ['Character_{story_id}', 'Item_{story_id}', 'Group_{story_id}', 'Location_{story_id}'])
    AND n.dynamic = true
    WITH collect({{
        id: id(n), 
        labels: labels(n), 
        properties: properties(n)
    }}) AS nodes

    // Step 2: Return the nodes as JSON
    RETURN {{nodes: nodes}} AS graph
    '''
    response = graph.query(queryStart)
    #print("This method is done")
    return response


'''This method is used to take a pdf of the kneldge base and make it into text'''
'''This can be local'''
def initGraphIntellegence(path):
    pdf_path = path
    text = pdf_to_text(pdf_path)
    #print(text)

    deleteQuery = f'''
    MATCH (c:ChunkInstruction) 
    DETACH DELETE c'''
    graph.query(deleteQuery)

    # Split the text into chunks
    chunks = text_splitter.split_text(text)

    # Create unique constraint for chunks
    graph.query("""
    CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
        FOR (c:ChunkInstruction) REQUIRE c.chunkId IS UNIQUE
    """)

    # Create chunks and their nodes
    node_count = 0
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"  # Generate a unique ID for each chunk
        print(f"Creating `:ChunkInstruction` node for chunk ID {chunk_id}")
        graph.query("""
            CREATE (c:ChunkInstruction {chunkId: $chunkId, text: $chunkText})
        """, params={
            'chunkId': chunk_id,
            'chunkText': chunk
        })
        node_count += 1
    print(f"Created {node_count} nodes")

    # Create vector index
    graph.query("""
         CREATE VECTOR INDEX `instruction_chunks` IF NOT EXISTS
          FOR (c:ChunkInstruction) ON (c.textEmbedding) 
          OPTIONS { indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'    
         }}
    """)

    # Show indexes
    graph.query("SHOW INDEXES")

    # Encode text embeddings for chunks where textEmbedding is NULL
    graph.query("""
    MATCH (chunk:ChunkInstruction) WHERE chunk.textEmbedding IS NULL
    WITH chunk, genai.vector.encode(
      chunk.text, 
      "AzureOpenAI", 
      {
        token: $openAiApiKey, 
        resource: $resourceEx,
        deployment: $openAiEndpoint
      }) AS vector
    CALL db.create.setNodeVectorProperty(chunk, "textEmbedding", vector)
    """, 
    params={"resourceEx":'muse-mvp' ,"openAiApiKey": os.environ["AZURE_OPENAI_API_KEY"], "openAiEndpoint": "embedding_func"} )

    # Refresh schema and print
    graph.refresh_schema()
    print(graph.schema)


#gives u all relationships and Nodes in a JSOn that starts with array of Nodes in the Nodes section and relationships exist in their own part of the JSON
#Note the Node type is an array that in the current state only holds  single value at index 0 Below is an Example 
'''
[{'graph':{
    "nodes": [
        {
        "id": 232,
        "labels": ["Character_5"],
        "properties": {
            "id": "Kael",
            "height": "tall",
            "hair_color": "sandy blonde",
            "age": "24",
            "role": "Protagonist",
            "gender": "Male"
        }
        },
        {
        "id": 238,
        "labels": ["Location_5"],
        "properties": {
            "id": "Rivermoor"
        }
        },
        {
        "id": 243,
        "labels": ["Item_5"],
        "properties": {
            "id": "Heart Of Winter"
        }
        }
    ],
    "relationships": [
        {
        "startNode": "Kael",
        "type": "SEEK",
        "endNode": "Heart Of Winter",
        "properties": {
            "reason_for_relationship_because": "Kael sought the Heart of Winter to gain the Frost Giants' support."
        }
        },
        {
        "startNode": "Kael",
        "type": "TRAVELLED_TO",
        "endNode": "Rivermoor",
        "properties": {
            "reason_for_relationship_because": "Kael traveled to Rivermoor to find allies."
        }
        }
    ]
    }
}]
'''
def get_graph_entities(story_id):
    cypher_query = f"""
    // Step 1: Retrieve all nodes with the specified labels
    MATCH (n)
    WHERE any(label in labels(n) WHERE label IN ['Character_{story_id}', 'Item_{story_id}', 'Group_{story_id}', 'Location_{story_id}'])
    WITH collect({{
        id: id(n), 
        labels: labels(n), 
        properties: properties(n)
    }}) AS nodes

    // Step 2: Retrieve all relationships between these nodes, including their startNode and endNode 'id' properties
    MATCH (n)-[r]->(m)
    WHERE any(label in labels(n) WHERE label IN ['Character_{story_id}', 'Item_{story_id}', 'Group_{story_id}', 'Location_{story_id}'])
    AND any(label in labels(m) WHERE label IN ['Character_{story_id}', 'Item_{story_id}', 'Group_{story_id}', 'Location_{story_id}'])
    WITH nodes, collect({{
        type: type(r), 
        startNode: startNode(r).id, 
        endNode: endNode(r).id, 
        properties: properties(r)
    }}) AS relationships

    // Step 3: Return the combined nodes and relationships as JSON
    RETURN {{nodes: nodes, relationships: relationships}} AS graph

    """

    response = graph.query(cypher_query)
    #print(response)
    return (response)




    
'''
Text is the draft, numSet is the project number , persist is if the graph has already been written to before, 
keepDynammicNodes is a work in progress to maintain nodes made only in the dynnamic questioning
'''
def graphCreator(text, numSet:int, persist = True, keepDynammicNodes=True):

    story_id = numSet

    query_persist = f"""
       MATCH (n)
        WHERE n:Character_{story_id} OR n:Item_{story_id} OR n:Location_{story_id} OR n:Group_{story_id}
        RETURN n
    """
    result_1 = graph.query(query_persist)
    if result_1:
        persist = True

    else:
        persist = False
        #print("the nodes dont exist")

    
    query =f"""
    MATCH (n)
    WHERE n:Character_{story_id} OR n:Item_{story_id} OR n:Location_{story_id} OR n:Group_{story_id}
    OPTIONAL MATCH (n)-[r]->(m)
    WHERE m:Character_{story_id} OR m:Item_{story_id} OR m:Location_{story_id} OR m:Group_{story_id}
    RETURN DISTINCT properties(n) AS character1_properties, properties(m) AS character2_properties, type(r) AS RelationshipType,r.reason_for_relationship_because AS Reason
    """

    
    response = graph.query(query)
    #print(response)
    relationships_array = []

    #WORK IN PROGRESS - Finished  Aug 7th
    if keepDynammicNodes:
        dynamicQ_response = get_dynamic_nodes(story_id)[0].get('graph')
        #print(dynamicQ_response)
        
        #return
        

    #print(dynamicQ_response)
    #return

    for record in response:
        character1_properties = record['character1_properties']
        character2_properties = record.get('character2_properties')
        relationship_type = record.get('RelationshipType')
        reason = record.get('Reason', '')

        # Check if dynammic is false or does not exist for character1_properties
        if not character1_properties.get('dynammic', False):
            character1_id = character1_properties.get('id')
            character2_id = character2_properties.get('id') if character2_properties else None

            # Add to relationships_array
            if relationship_type:
                if reason:
                    relationship_str = f"{character1_id} -> {relationship_type} because of in_story_reason -> {character2_id}"
                else:
                    relationship_str = f"{character1_id} -> {relationship_type} -> {character2_id}"
                relationships_array.append(relationship_str)

    stringRelationships = '\n'.join(relationships_array)
    #print(stringRelationships)

    #delete graph project to re assert new draft
    
    nodesPossible = ["Character","Item","Group","Location", "Chunk"]
    for vals in nodesPossible:
        deleteQuery = f'''
        MATCH (c:{vals}_{story_id}) 
        DETACH DELETE c'''
        graph.query(deleteQuery)


    # Initialize ChatOpenAI and LLMGraphTransformer

    story_id = numSet
    # Define the document 

    documents = [Document(page_content=text)]
    
    
    system_prompt = f'''
    You are an AI system that is good at finding relationships between nodes. The nodes can be ["Character_{story_id}", "Location_{story_id}", "Group_{story_id}", "Item_{story_id}"] and they have their own relationships respectivlty
    Node Descriptions
        - Character_{story_id} refers to characters in the story. Titles should not be included so 'Clair of Water' is just the name Clair. Characters can be ambigous meaning they dont need an exact Name terms like 'Snake' or 'Queen' are still valid Characters.
        - Location_{story_id} stands for any locations in the story, these locations must be recorded. Examples can be both general and specific 'Ancient City Ruins' and 'Eldorado'. ANy place a character visits or arrives at also fits this node type.
        - Group_{story_id} refers to multiple characters or some form of an organization. This means multiple Character_{story_id} would have a relationship APART_OF the respective Group_{story_id} node.
        - Item_{story_id} refers to an object of some sort in the story like jewlery. This can be any object but must not be a Location, always check if a potetial item can rather be a Location or Character. Items can be general or specific, for example, 'Armor', 'Talisman of the crucible'.
   
    Note that relationships should be general words like LIKE, HATE, ATTACK, ANGER, ADMIRE, LOVE, try to keep the relationships general.
    Important Relationships between Nodes can be explicit and implicit in a story Example : Explicit - "A hates B", Implicit - "A fought side by side with B creating a level of Trust" Make sure to catch as many realtionships as possible. Its better to Catch as many relationships as possible.


    The reason for the Nodes relationships must be documented in the relationship properties under the key "Reason_for_relationship_Because" .
    Make sure to may attention where characters travelled make a relationship between characters and where they travelled, a Relationship called Travelled_To would suffice, under the key "Reason_for_relationship_Because" if possible try to document the reason for travel.
    Any explicitly stated Character, Location, Group, Item Regardless of importance MUST be recorded. One good way to record them is to look for proper nouns as they will always fit in the Character, Location, Group, and item. 
    But some nodes will be general nouns like "Lions" or "Turtles"

    If multiple Characters have a relationship with on node make sure all the nodes get a relationship to the singular target node dont leave any out!.
    All NAMES and proper nouns should fall into Character_{story_id}", "Location_{story_id}", "Group_{story_id}", "Item_{story_id},
    Important, Identify and make as many relationships as possible
    '''
    system_message = SystemMessage(content=system_prompt)
    #parser = JsonOutputParser(pydantic_object=UnstructuredRelation)

    human_prompt = ""
    if persist:
        #print(stringRelationships)

        human_prompt = PromptTemplate(
            template="""
        Examples:
        Chan likes greg because Greg had helped him pick up a bag. Result: Node(id='Chan', type='Character_{story_id}') Node(id='Greg', type='Character_{story_id}')
        Relationship(source=Node(id='Chan', type='Character_{story_id}'), target=Node(id='Greg', type='Character_{story_id}'), type='LIKE', properties='reason_for_relationship_because': 'Greg had helped Chan pick up a bag')
        
        Chan likes to travel to Paris because he like the weather. Result: Node(id='Chan', type='Character_{story_id}') Node(id='Paris', type='Location_{story_id}')
        Relationship(source=Node(id='Chan', type='Character_{story_id}'), target=Node(id='Paris', type='Location_{story_id}'), type='Travel_to', properties='reason_for_relationship_because': 'Chan loved the weather in Paris')
        
        Also make multiple relationships for multiple characters if needed
        Chan and Drake likes to travel to Paris because Chan liked the weather and Drake like the Effiel tower. Result: Node(id='Chan', type='Character_{story_id}'), Node(id='Drake', type='Character_{story_id}'), Node(id='Paris', type='Location_{story_id}')
        Relationship(source=Node(id='Chan', type='Character_{story_id}'), target=Node(id='Paris', type='Location_{story_id}'), type='Travel_to', properties='reason_for_relationship_because': 'Chan loved the weather in Paris')
        Relationship(source=Node(id='Drake', type='Character_{story_id}'), target=Node(id='Paris', type='Location_{story_id}'), type='Travel_to', properties='reason_for_relationship_because': 'Drake liked the Effiel tower')


        Reason_for_relationship_Because Should Include Names and avoid pronouns like he and she make it specific
        For the following text, extract entities and relations as in the provided example. Focuse on recording Reason_for_relationship_Because

        Also if a Node is a character try to track these Aspects  ["hair_color","height","age","gender", "role"], track these when the appear in the story, but leave Blank if they do not exist
        For character 'gender' Can be only 4 options 1- 'Male' 2- "Female" 3- "Other" (Meaning not a male or Female) 4- "" (Blank if not known) Only choices
        For character 'role' Can be only 4 options, you only choose one that best fits the character
        - "Antagonist" (A Villain, Works against the Protagonist or Hero, this character is usually evil and there is atleast one in each story)
        - "Protagonist" (The main character in the story, can only be one character in the story)
        - "Supporting Character" (A character that supports the Protagonist in their story)
        - "Minor Role" (Character that is none of the other traits)

        These are some of the parts of the story, some of these will be outdated so you will need to change them or delete them. Add new relationships as seen fit. 
        in_story_reason indicates an existing relationship reason.
        Hints : {stringRelationships} 
        These are outdated and may not have all the information use this information as a hint only, make your own decisions. VERY IMPORTAN If these Hints are true for the story then make sure to add them aswell, DONT ADD THE ones that arent True.
        There are many more Node and relationships to find and verify all the ones given above as well to see if they still make sense
        If multiple Characters have a relationship with on node make sure all the nodes get a relationship to the singular target node dont leave any out!. 
        

        \nText: {input}""",
            input_variables=["input"],
            input_types={"input": str},
            partial_variables={"story_id": story_id, "stringRelationships": stringRelationships}
        )
        #print("Hint used")

    else:
        human_prompt = PromptTemplate(
        template="""
        Examples:
        Chan likes greg because Greg had helped him pick up a bag. Result: Node(id='Chan', type='Character_{story_id}') Node(id='Greg', type='Character_{story_id}')
        Relationship(source=Node(id='Chan', type='Character_{story_id}'), target=Node(id='Greg', type='Character_{story_id}'), type='LIKE', properties='reason_for_relationship_because': 'Greg had helped Chan pick up a bag')
        
        Chan likes to travel to Paris because he like the weather. Result: Node(id='Chan', type='Character_{story_id}') Node(id='Paris', type='Location_{story_id}')
        Relationship(source=Node(id='Chan', type='Character_{story_id}'), target=Node(id='Paris', type='Location_{story_id}'), type='Travel_to', properties='reason_for_relationship_because': 'Chan loved the weather in Paris')
        
        Also make multiple relationships for multiple characters if needed
        Chan and Drake likes to travel to Paris because Chan liked the weather and Drake like the Effiel tower. Result: Node(id='Chan', type='Character_{story_id}'), Node(id='Drake', type='Character_{story_id}'), Node(id='Paris', type='Location_{story_id}')
        Relationship(source=Node(id='Chan', type='Character_{story_id}'), target=Node(id='Paris', type='Location_{story_id}'), type='Travel_to', properties='reason_for_relationship_because': 'Chan loved the weather in Paris')
        Relationship(source=Node(id='Drake', type='Character_{story_id}'), target=Node(id='Paris', type='Location_{story_id}'), type='Travel_to', properties='reason_for_relationship_because': 'Drake liked the Effiel tower')


        Reason_for_relationship_Because Should Include Names and avoid pronouns like he and she make it specific
        For the following text, extract entities and relations as in the provided example. Focuse on recording Reason_for_relationship_Because

        Also if a Node is a character try to track these Aspects  ["hair_color","height","age","gender", "role"], track these when the appear in the story, but leave Blank if they do not exist
        For character 'gender' Can be only 4 options 1- 'Male' 2- "Female" 3- "Other" (Meaning not a male or Female) 4- "" (Blank if not known) Only choices
        For character 'role' Can be only 4 options, you only choose one that best fits the character
        - "Antagonist" (A Villain, Works against the Protagonist or Hero, this character is usually evil and there is atleast one in each story)
        - "Protagonist" (The main character in the story, can only be one character in the story)
        - "Supporting Character" (A character that supports the Protagonist in their story)
        - "Minor Role" (Character that is none of the other traits)
        If multiple Characters have a relationship with on node make sure all the nodes get a relationship to the singular target node don't leave any out!. 

        \nText: {input}""",
            input_variables=["input"],
            input_types={"input": str},
            partial_variables={"story_id": story_id}
        )

    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
    # if a Node is a character 'background' tracks the some of their major achievements and moments in the story, IMPORTANT KEEP THIS short a summary of the character in the story. For example "Character X hopes for the nation to find peace, X works with his allies to save the world, X is from the town Caelid" 
    #          This exmaple focusses correctly on goals, acheivements and aspirations. Make sure to do the same for if a Node is a character 'background' 
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message, human_message_prompt]
    )

    # Filter and convert the document to graph documents
    llm_transformer_filtered = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=[f"Character_{story_id}", f"Location_{story_id}", f"Group_{story_id}", f"Item_{story_id}"],
        strict_mode = False,
        #prompt = "Consider relationships Ally, Teacher , Enemy, and Family"
        prompt=chat_prompt,
        node_properties = ["hair_color","height","age","gender", "role"],
        relationship_properties = ["Reason_for_relationship_Because"]
    )
    
    with get_openai_callback() as cb:
        graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(documents)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")


    # Parse nodes and relationships
    parsed_nodes = [{"id": node.id, "type": node.type, "properties": node.properties} for node in graph_documents_filtered[0].nodes]
    parsed_relationships = [{"source": rel.source.id, "target": rel.target.id, "type": rel.type} for rel in graph_documents_filtered[0].relationships]


    print(f"Nodes:{graph_documents_filtered[0].nodes}")
    print(f"Relationships:{graph_documents_filtered[0].relationships}")
    graph.add_graph_documents(
        graph_documents_filtered, 
        baseEntityLabel=False, 
        include_source=False
    )

    #WORK IN PROGRESS - Finished Aug 7th
    if keepDynammicNodes:
        
        for item in dynamicQ_response['nodes']:
            #type_info = item.get('Type', {})
            print(item)
            node_id = item['id']
            labels = item['labels']
            properties = item['properties']
            
            # Handle properties correctly, quoting only string values
            formatted_properties = ', '.join(
                [f"{key}: '{value}'" if isinstance(value, str) else f"{key}: {value}"
                for key, value in properties.items() ]
            )
                #dynamicQ_array_cypher.append(f'''CREATE (c:{node_type[0]} {type_info}''')

            queryCheck = f'''MATCH (c:{labels[0]} {{id: $node_info_id}})
                            RETURN c
                            '''
            responseCheck  = graph.query(queryCheck, params={'node_info_id':properties['id']})
            print(responseCheck)
            if not responseCheck:
                cypher_query_draft = f'''
                WITH {{
                    id: {node_id},
                    labels: {labels},
                    properties: {{{formatted_properties}}}
                }} AS data
                CREATE (n)
                SET n += data.properties
                SET n.id = data.properties.id
                WITH n, data.labels AS labels
                CALL apoc.create.addLabels(n, labels) YIELD node
                RETURN node'''

                graph.query(cypher_query_draft)
            

    vector_search_setter(text, numSet)

    response_ender = get_graph_entities(story_id)
    print(response_ender)
    return response_ender


'''THis method runs everytime the story is runnning splitting the story into segments for Vector RAG'''
def vector_search_setter(text, numSet: int):
    # Initialize text splitter
    story_id = numSet

    # Split the text into chunks
    chunks = text_splitter.split_text(text)

    # Create unique constraint for chunks
    graph.query(f"""
    CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
        FOR (c:Chunk_{story_id}) REQUIRE c.chunkId IS UNIQUE
    """)

    # Create chunks and their nodes
    node_count = 0
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"  # Generate a unique ID for each chunk
        print(f"Creating `:Chunk` node for chunk ID {chunk_id}")
        create_chunk_query = f"""
        CREATE (c:Chunk_{story_id} {{chunkId: $chunkId, text: $chunkText}})
        """
        graph.query(create_chunk_query, params={
            'chunkId': chunk_id,
            'chunkText': chunk
        })
        node_count += 1
    #print(f"Created {node_count} nodes")

    # Create vector index
    create_index_query = f"""
    CREATE VECTOR INDEX `text_chunks_{story_id}` IF NOT EXISTS
    FOR (c:Chunk_{story_id}) ON (c.textEmbedding) 
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """
    graph.query(f"""DROP INDEX `text_chunks_{story_id}` IF EXISTS""")
    graph.query(create_index_query)
    # Show indexes
    graph.query("SHOW INDEXES")

    # Encode text embeddings for chunks where textEmbedding is NULL
    match_query = f"""
    MATCH (chunk:Chunk_{story_id}) WHERE chunk.textEmbedding IS NULL
    WITH chunk, genai.vector.encode(
      chunk.text, 
      "AzureOpenAI", 
      {{
        token: $openAiApiKey, 
        resource: $resourceEx,
        deployment: $openAiEndpoint
      }}) AS vector
    CALL db.create.setNodeVectorProperty(chunk, "textEmbedding", vector)
    """
    graph.query(match_query, params={
        "resourceEx": 'muse-mvp',
        "openAiApiKey": os.environ["AZURE_OPENAI_API_KEY"],
        "openAiEndpoint": "embedding_func",
    })
    # Refresh schema and print
    graph.refresh_schema()
    #print(graph.schema)

'''may need to add Graph as parameter, this methods clearrs all Nodes that have dynamic as true in a given story'''
def clear_graph_dynamic_nodes(story_id ):

    
    possiblities = ['Character', 'Location', "Item", 'Group']

    for items in possiblities:
        graph.query(f'''MATCH (n:{items}_{story_id})
            WHERE n.dynamic = TRUE
            DETACH DELETE n''')
    return

# Main function to execute the program
def main():
    text = """
In the mystical land of Elaria, a diverse group of heroes came together She chronicled not only their adventures but also the histories and legends of Elaria, preserving the culture and lore of the land. Her songs, filled with the spirit of their journey, inspired countless individuals to strive for greatness."""

    test2 = """
In the mystical land of Eldardo, a realm of breathtaking beauty and untold secrets, and their ultimate victory against the Shadow Queen became the stuff of legend, a story passed down through the generations as a reminder of the power of unity, love, and sacrifice in the face of overwhelming darkness.
"""

    try:
        
            start_time = time.time()  # Timer starts
            canit = graphCreator(test2, 7)          # Call the method
            end_time = time.time()
            #canit = graphCreator(test2,2)
            elapsed_time = end_time - start_time
            print(f"Method took {elapsed_time:.2f} seconds to run.")


    except Exception as e:
        print(f"Error: {e}")

    

# Execute the main function
if __name__ == "__main__":
    main()
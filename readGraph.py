import os
from langchain_openai import AzureChatOpenAI
import json
from langchain_community.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase
from langchain.agents import create_tool_calling_agent
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import Neo4jVector
from neo4j.exceptions import ServiceUnavailable, AuthError
from langchain.chains import GraphCypherQAChain
from langchain.chains import RetrievalQA
from langchain_core.tools import tool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentType
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate


uri = "" #Neo4j uri
user = "neo4j" #neo4j user
password = "" # neo4j password
#local
os.environ["NEO4J_URI"] = uri
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = password

# Set the OpenAI API key and endpoint
os.environ["AZURE_OPENAI_API_KEY"] = ''
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
api_version = '2024-02-01'

# Initialize the Azure OpenAI embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="embedding_func",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=api_version,
)

llm = AzureChatOpenAI(
    azure_deployment="gpt40-mini-dynnamicQs",
    api_version="2024-06-01",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

graph = Neo4jGraph(url= os.environ["NEO4J_URI"], username = os.environ["NEO4J_USERNAME"], password= os.environ["NEO4J_PASSWORD"]  )

vector_instruction_index = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    index_name='instruction_chunks',
    node_label='ChunkInstruction',
    text_node_properties=['text'],
    embedding_node_property="textEmbedding",
)

advise_qa = RetrievalQA.from_chain_type(
    llm=llm,
    #return_source_documents=True,
    chain_type="stuff",
    
    retriever=vector_instruction_index.as_retriever(search_kwargs={"k":3}),
    verbose=True,
    
    )

'''makes node data easy to ready, used in cypher check as a helper'''
def format_node_data(node_data):
    # Create a dictionary to store node names and their types
    node_dict = {}
    
    for entry in node_data:
        name = entry['name']
        node_type = entry['nodeType']
        # Join node types into a single string
        node_type_str = ', '.join(node_type)
        # Add to dictionary
        node_dict[name] = node_type_str
    
    # Format the dictionary into a pretty string
    pretty_string = ', '.join(f"{name}: {node_type}" for name, node_type in node_dict.items())
    
    return pretty_string


'''This is to take the entire graph schema and make it relevent only to the current story'''
def parse_graph_schema(schema, story_id):
    target_node_types = [f'Character_{story_id}', f'Item_{story_id}', f'Location_{story_id}', f'Group_{story_id}']
    result = {'node_props': {}, 'relationships': [], 'rel_props': {}}
    
    # Extract node properties
    for node_type, props in schema.get('node_props', {}).items():
        if node_type in target_node_types:
            result['node_props'][node_type] = props
    
    # Extract relationships and their properties
    for relationship in schema.get('relationships', []):
        if relationship['start'] in target_node_types or relationship['end'] in target_node_types:
            result['relationships'].append(relationship)
            rel_type = relationship['type']
            if rel_type in schema.get('rel_props', {}):
                result['rel_props'][rel_type] = schema['rel_props'][rel_type]
    
    return result

'''This takes the parse_graph_schema and make it work as prompt engineering'''
def pretty_print_schema(schema):
    def format_node_properties(node_props):
        formatted = "Node properties:\n"
        for node_type, props in node_props.items():
            props_str = ", ".join([f"{prop['property']}: {prop['type']}" for prop in props])
            formatted += f"{node_type} {{{props_str}}}\n"
        return formatted

    def format_relationship_properties(rel_props):
        formatted = "Relationship properties:\n"
        for rel_type, props in rel_props.items():
            props_str = ", ".join([f"{prop['property']}: {prop['type']}" for prop in props])
            formatted += f"{rel_type} {{{props_str}}}\n"
        return formatted

    def format_relationships(relationships):
        formatted = "The relationships:\n"
        for rel in relationships:
            formatted += f"(:{rel['start']})-[:{rel['type']}]->(:{rel['end']})\n"
        return formatted

    node_props_str = format_node_properties(schema.get('node_props', {}))
    rel_props_str = format_relationship_properties(schema.get('rel_props', {}))
    relationships_str = format_relationships(schema.get('relationships', []))

    response =  f"{node_props_str}\n{rel_props_str}\n{relationships_str}"
    return response


'''Track the Labels and nodes of the story for the model'''
def initAI(story_id):
    target_node_types = [f'Character_{story_id}', f'Item_{story_id}', f'Location_{story_id}', f'Group_{story_id}']
    result = {'node_props': {}, 'relationships': [], 'rel_props': {}}
        
    # Extract node properties
    for node_type, props in graph.get_structured_schema.get('node_props', {}).items():
        if node_type in target_node_types:
                result['node_props'][node_type] = props
    
    labels = list(result['node_props'].keys())
    nodes = []
    
    for label in labels:
        query = f"MATCH (n:{label}) RETURN n"
        result = graph.query(query)
        
        for record in result:
            nodes.append(record["n"]['id'])
    return nodes


'''Helper for the create_relationship function, runs query into a JSON format'''
def create_relationship_helper(query, story_id):
    prompt_template = """
    Fill in the following variables based on the given context. Leave as blank if not present

    Context: {context} 
    Fill in the JSON with information provided A node1 and node2 maybe a Item, Character, Group, or Location. It may also be some form of a NOun or porper noun
    

    Respond with the values in JSON format:
    {{
    "node1": "Name of the first node"
    "node2": "Name of the second node, should not be the same as node1",
    "relationship_type": "This characterizes the relationship in one to two words, if its two word make sure to use a '_'. like LOVE or TRAVELLED_TO. This should be somewhat general Try to Keep Capitilized."
    "Reason_for_relationship_Because": "more details about the relationship from the query",
    }}
    """

    class relationshipCreator(BaseModel):
        node1: str = Field(description=f"Name of the first node")
        node2: str = Field(description="Name of the second node, should not be the same as node1")
        relationship_type: str = Field(description="""This characterizes the relationship in one to two words, if its two word make sure to use a '_'. like Love or Traveled_To". This should be somewhat general""")
        Reason_for_relationship_Because: str = Field(description="More details about the relationship from the query")
        
    parser = JsonOutputParser(pydantic_object=relationshipCreator)

    prompt = PromptTemplate(
        input_variables=["context"],
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions(),
        "story_id": story_id},
    )

    # Create the LLMChain
    chain = prompt | llm | parser

    # Define the context and initial values
    context = query
    
    # Run the chain
    response = chain.invoke({"context": context})
    return response

'''helper for the write_otherNodes method'''
def write_otherNodes_helper(query, story_id):
    prompt_template = """
    Fill in the following variables based on the given context. Leave as blank if not present

    Context: {context} 
    Fill in the JSON with information provided
    you will need to decide whether the context is describing an Item, a group of characters or a location.
    "type": OPTIONS : 
    - Item
    - Group
    - Location
    Pick from only these

    example: "type" : "Item"

    Respond with the values in JSON format:
    {{
    "type": "Pick from three options to classify the context, pick from these
    -Item
    -Group
    -Location",
    "name": "Name of the Item, Location or Group.",
    "more_info": "Anything Else stated about the Item, Group or Location",
    }}
    """

    class nonCharacterCreator(BaseModel):
        type: str = Field(description=f"either Item, Group or Location")
        name: str = Field(description="Name of the Item, Location or Group.")
        more_info: str = Field(description="Anything Else stated about the Item, Group or Location")
        
    parser = JsonOutputParser(pydantic_object=nonCharacterCreator)

    prompt = PromptTemplate(
        input_variables=["context"],
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions(),
        "story_id": story_id},
    )

    # Create the LLMChain
    chain = prompt | llm | parser

    # Define the context and initial values
    context = query
    
    # Run the chain
    response = chain.invoke({"context": context})
    return response

'''helper for the delete_dynamic_Node method'''
def delete_dynamic_Node_helper(query, story_id):
    prompt_template = """
    Fill in the following variables based on the given context. Leave as blank if not present

    Context: {context} 
    Fill in the JSON with information provided A node as a Item, Character, Group, or Location. It may also be some form of a Noun or proper noun
    

    Respond with the values in JSON format:
    {{
    "node": "Name of the node"
    
    }}
    """

    class nodeDeleter(BaseModel):
        node: str = Field(description=f"Name of the node")
        
        
    parser = JsonOutputParser(pydantic_object=nodeDeleter)

    prompt = PromptTemplate(
        input_variables=["context"],
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions(),
        "story_id": story_id},
    )

    # Create the LLMChain
    chain = prompt | llm | parser

    # Define the context and initial values
    context = query

    # Run the chain
    response = chain.invoke({"context": context})
    return response

'''helper for the vector search'''
def vector_search_helper(query, story_id):
    vector_index = Neo4jVector.from_existing_graph(
        embedding=embeddings,
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        index_name=f'text_chunks_{story_id}',
        node_label=f'Chunk_{story_id}',
        text_node_properties=['text'],
        embedding_node_property="textEmbedding",

    )

    vector_qa = RetrievalQA.from_chain_type(
        llm=llm,
        #return_source_documents=True,
        chain_type="stuff",
        
        retriever=vector_index.as_retriever(search_kwargs={"k":5}),
        verbose=True,
    
    )
    #print(vector_qa.invoke(query))
    with get_openai_callback() as cb:
        response = vector_qa.invoke(query)
    #print(cb.total_tokens)
    return response

'''helper for writing a character'''
def write_character_helper(query, story_id):
    prompt_template = """
    Fill in the following variables based on the given context. Leave as blank if not present

    Context: {context} 
    Fill in the JSON with information provided

    Respond with the values in JSON format:
    {{
    "character_name": "Character Name",
    "character_age": "Character Age",
    "character_hair_color": "Character color of hair",
    "character_height": "Height of character"
    "character_gender": "Can be only 4 options 'Male', "Female" , "Other" (Meaning not a male or Female) or "" (Blank if not known) Only choices"
    "character_role": "For character 'role' Can be only 4 options, you only choose one that best fits the character
    - "Antagonist" (A Villain, Works against the Protagonist or Hero, this character is usually evil and there is atleast one in each story)
    - "Protagonist" (The main character in the story, can only be one character in the story)
    - "Supporting Character" (A character that supports the Protagonist in their story)
    - "Minor Role" (Character that is none of the other traits)"
    }}
    """

    class characterCreator(BaseModel):
        character_name: str = Field(description="Character Name")
        character_age: str = Field(description="Character Age")
        character_hair_color: str = Field(description="Character color of hair")
        character_height: str = Field(description="Height of character")
        character_gender: str = Field(description='''Can be only 4 options 'Male'  "Female" "Other" (Meaning not a male or Female) or "" (Blank if not known) Only choices''')
        character_role: str = Field(description='''For character 'role' Can be only 4 options, you only choose one that best fits the character
        - "Antagonist" (A Villain, Works against the Protagonist or Hero, this character is usually evil and there is atleast one in each story)
        - "Protagonist" (The main character in the story, can only be one character in the story)
        - "Supporting Character" (A character that supports the Protagonist in their story)
        - "Minor Role" (Character that is none of the other traits)''')


    parser = JsonOutputParser(pydantic_object=characterCreator)

    prompt = PromptTemplate(
        input_variables=["context"],
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Create the LLMChain
    chain = prompt | llm | parser

    # Define the context and initial values
    context = query
    

    # Run the chain
    with get_openai_callback() as cb:
        response = chain.invoke({"context": context})
    #print(cb.total_tokens)
    return response


def cypher_check_helper(story_id,graphFilteredSchema, str_nodes, cleanHelper):
    CYPHER_GENERATION_TEMPLATE = """
        Task:Generate Cypher statement to query a graph database Using the examples
        Instructions:
        1. Use the provided relationship types and properties from the schema.
        2. Do not use any relationship types or properties not in the schema.
        3. Allowed nodes: Character_{story_id}, Location_{story_id}, Organization_{story_id}, Item_{story_id}.
        4. Use Location_{story_id} for places, Item_{story_id} for objects, Organization_{story_id} for groups and Character_{story_id} for characters.
        5. CHARACTER1, ITEM1 and more are all variables made to be changed.
        6. Use the examples above all else and UNION if possible
        7. Try to use proerpeis of nodes and return them
        8. ALWAYS try to add Aliasing when generating Cypher
        

        When querying why a relationship exists:
        1. Use the relationship property reason_for_relationship_because.
        2. Check both directions for the relationship.
        3. Sometimes asking "Who are K's teammates" may not have a direct relationship,
            if so, track the Relationship properties with similar meaning or conotations like "Love", "Friend", "Ally" along with a reason to make an educated guess.
            If its "Who are K's enemies" you may pick relationships like "Hate", "Anger", "Disgust" which match closly with the word enemy. Use the schema for this.

        IMPORTANT
        Here are the Nodes and their types 
        {cleanHelper}

        Examples to ALWAYS Follow:

        - Querying who a character is 
            -Question: "Who is the CHARACTER1?"
            MATCH (c:Character_{story_id} {{id: CHARACTER1}}) 
            RETURN c AS character

        -Querying a Location, Item, or Group and asking more informato about it 
            -Question "tell me about the AMULET" 
            MATCH (n {{id: AMULET}})
            OPTIONAL MATCH (n)-[r]->(m)  //this checks relationships from n -> m outwards
            RETURN n, labels(n) AS labels, r, type(r) AS relationshipType, m, properties(m) AS character_properties, r.reason_for_relationship_because AS Reason
            UNION
            MATCH (n {{id: AMULET}})
            OPTIONAL MATCH (m)-[r]->(n)  /this checks relationships from m -> n inwards
            RETURN n, labels(n) AS labels, r, type(r) AS relationshipType, m, properties(m) AS character_properties, r.reason_for_relationship_because AS Reason
            

        - Querying a Node's, like a Character Jeff, relationship with multiple other nodes, only One node is specifically needed like One character:
            - Question: "Give all the characters and their relationships to CHARACTER1" 
            -Additional Details, When Asked about a characters relationships this makes the specific Node both the Source and the Target so we use the command below
            
            Always use Union to change the source and Target nodes
            -Note as always Union when Possible
            MATCH (c:Character_{story_id})-[r]->(a:Character_{story_id} {{id: CHARACTER1}})
            RETURN c.id AS character_id, type(r) AS relationship_type, r.reason_for_relationship_because AS reason, properties(c) AS character_properties, properties(a) AS target_properties
            UNION
            MATCH (a:Character_{story_id} {{id: CHARACTER1}})-[r]->(c:Character_{story_id})
            RETURN c.id AS character_id, type(r) AS relationship_type, r.reason_for_relationship_because AS reason, properties(c) AS character_properties, properties(a) AS target_properties
            

        - Querying a Locations with Characters:
            - Question: "Give Locations who CHARACTER1 has relationships with and their reasons?" 
            MATCH (c:Character_{story_id})-[r]-(l:Location_{story_id})
            WHERE c.id = "CHARACTER1"
            RETURN c.id AS character_id, type(r) AS relationship_type, r.reason_for_relationship_because AS reason, properties(c) AS character_properties, properties(l) AS location_properties


        - Querying a single specific relationship and two specific Nodes:
            - Question: "How did CHARACTER1 love ITEM1?" Be prepared to pick a existing relationship most similar to Love as well like "Infactuated" if in the schema
            MATCH (c:Character_{story_id} {{id: CHARACTER1}})-[r:LOVED]->(i:Item_{story_id} {{id: ITEM1}})
            RETURN r.reason_for_relationship_because AS Reason, properties(c) AS character_properties, properties(i) AS item_properties
            UNION
            MATCH (i:Item_{story_id} {{id: ITEM1}})-[r:LOVED]->(c:Character_{story_id} {{id: CHARACTER1}})
            RETURN r.reason_for_relationship_because AS Reason, properties(c) AS character_properties, properties(i) AS item_properties


        - Querying the type of relationship with reason, when given two exact Nodes:
            - Question: "What was the relationship between CHARACTER1 and CHARACTER2?"
            MATCH (c1:Character_{story_id} {{id: CHARACTER1}}), (c2:Character_{story_id} {{id: CHARACTER2}})
            MATCH path = shortestPath((c1)-[*]-(c2))
            MATCH path_reverse = shortestPath((c2)-[*]-(c1))
            UNWIND relationships(path) AS r
            UNWIND relationships(path_reverse) AS rv
            RETURN type(r) AS relationship, r.reason_for_relationship_because AS Reason, type(rv) AS relationshipReverse, rv.reason_for_relationship_because AS ReasonReverse, properties(c1) AS character1_properties, properties(c2) AS character2_properties                     

        - Querying a relationship involving different node types to see ONLY see if a relationship existed:
            - Question: "Was there a relationship between LOCATION1 and CHARACTER1, and what was it?"
            MATCH (c:Character_{story_id} {{id: CHARACTER1}}), (l:Location_{story_id} {{id: LOCATION1}})
            MATCH path = shortestPath((c)-[*]-(l))
            UNWIND relationships(path) as r
            RETURN type(r) AS relationship, r.reason_for_relationship_because AS Reason, properties(c) AS character_properties,  properties(l) AS location_properties
        
        - Querying Multiple relationships, assuming multiplpe relationships fit this role
            -Question: "Who are the friends of CHARACTER1?" 
            - IMPORTANT The Best relationships to use from the Schema would be relatonships that are positive since 'friends' is a positive word. if talking about 'Enemy' pick relationships that are negative like "Anger" and "Attack". Better pick more relationships than less.
            - IMPORTANT using as many relationships as available in the schema that match the tone of the query is heavily recommended

            MATCH (c:Character_{story_id} {{id: CHARACTER1}})-[r:LAUGH|ADMIRE|COMMUNICATE]-(a:Character_{story_id}) 
            RETURN a, properties(a) AS character_properties
            UNION
            MATCH (a:Character_{story_id})-[r:LAUGH|ADMIRE|COMMUNICATE]-(c:Character_{story_id} {{id: CHARACTER1}}) 
            RETURN a, properties(a) AS character_properties


        - Querying what relationships a node has 
            - Question: "What are CHARACTER1's relationships in the story"
            MATCH (:Character_{story_id} {{id: CHARACTER1}})-[r]->()
            RETURN r

        The IDS were all varibles and not something to be copied on use the format! Treat the Examples as just examples for reference.
        IMPORTANT and ESSENTIAL when using the Example templates above make sure to run only One full cypher command per query
        
        Schema: {schema}

        Question: {question}

        Ensure you use valid Cypher and follow the schema closely. If no result is found, return nothing. 
        Try to copy the examples and take instpiration always. NEVER use things like LOCATION1 or CHARACTER1 in Cypher Queries, pick from these Node names {str_nodes}
        
        """

    cypher_generation_prompt = PromptTemplate(
        template=CYPHER_GENERATION_TEMPLATE,
        input_variables=["question"],
        partial_variables={"story_id": story_id, "schema": graphFilteredSchema, "str_nodes": str_nodes, "cleanHelper":cleanHelper}
    )
    return cypher_generation_prompt



'''
This Method is used to Prompt the Dynammic Questions with multiple methods inside of it. query is a string question. 
Memory_store is the memory of the conversation and story_id is needed to capture the right story
Memory_store is Either None or make it the output['chat_history'], the output of this function has an input and a chat history section in the dictionary
'''
def prompt_graph(query: str, memory_store, story_id: int):

            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            if memory_store is not None:
                #print("memory store used")
                for message in memory_store:
                    memory.chat_memory.add_message(message)

            graph.refresh_schema() # refresh

            nodes = initAI(story_id)
            str_nodes = str(nodes)
            #print(str_nodes)

            '''
            Delete the dynamic=true nodes only, 
            only an ID needs to be described
            '''
            @tool
            def delete_dynamic_Node(query):
                """This tool is to delete Nodes from the story, this method should be used when explicitly told to remove a character, this is a powerful tool that must be used carefuly.
                Use this method when queries seem like "I want to delete...", "lets remove character x ..." or  "Get rid of character Y from the story..." 
                Use this method carefully. KNowing whether the node is a Item, Character, Group, or Location is also helpful"""

                response = delete_dynamic_Node_helper(query, story_id)
                # Parse the JSON response    
                filled_values = response
                node = filled_values.get("node", "Unknown")

               #if the json output is not able to find a name
                if node == "Unknown" or node == " " or node is None or node == "":
                    return "The character name was not deleted because there is a name missing"

                typesPossible=['Character',"Item", "Group", "Location"]
                node_1_type = ""
                
                for type in typesPossible:
                    query = f"""
                    MATCH (c:{type}_{story_id} {{id: $id, dynamic: $tf}})
                    RETURN c
                    """
                    result_1 = graph.query(query, params={'id': node, 'tf':True})
                   
                    if result_1:
                        node_1_type = f"{type}_{story_id}"
                    
                #if the node label cannot be found or more likely the node is from the story as a character so the dynamic = False for dynamic=None
                if node_1_type == "" or node_1_type is None :
                    return "This node may not be in the Database in the first place or this character is from your story and cannot be removed until your written story is changes. "
                
                query = f"""
                    MATCH (p:{node_1_type} {{id: $id1}})
                    DETACH DELETE p
                    """
                graph.query(query, params={
                        'id1': node,
                    })

                query = f"""
                    MATCH (c:{node_1_type} {{id: $id, dynamic: $tf}})
                    RETURN c
                    """
                result_1 = graph.query(query, params={'id': node, 'tf':True})
                if result_1:
                    return "The Delete May have been unsuccessful"
                else:
                    return "The deletation of {node} was done"

            '''Create a relationship between Nodes'''
            @tool
            def create_relationship(query):
                """This tool is for creating and writing a relationship to the story, queries like "I want X and Y to fall in love ..." or "Make X and Y have this association/relationship"
                Two different nodes/subject must be given with a relationship
                The query should have only 2 nodes, a relationship and optionally more details"""
               
                response = create_relationship_helper(query,story_id)
                # Parse the JSON response
                filled_values = response
                node1 = filled_values.get("node1", "Unknown")
                node2 = filled_values.get("node2", "Unknown")
                relationship_type = filled_values.get("relationship_type", "Unknown")
                Reason_for_relationship_Because_var = filled_values.get("Reason_for_relationship_Because", "Unknown")
                
                typesPossible=['Character',"Item", "Group", "Location"]
                node_1_type = ""
                node_2_type = ""
                for type in typesPossible:
                    query = f"""
                    MATCH (c:{type}_{story_id} {{id: $id}})
                    RETURN c
                    """
                    result_1 = graph.query(query, params={'id': node1})
                    result_2 = graph.query(query, params={'id': node2})
                    if result_1:
                        node_1_type = f"{type}_{story_id}"
                    if result_2:
                        node_2_type = f"{type}_{story_id}"
                    
                #Need to track the node names in the query, one of these are missing for the 
                if node_1_type == "" or node_1_type is None or node_2_type == "" or node_2_type is None :
                    return "You should describe two nodes, if you are sure you did then retry and make the relationship more explicit. Muse may have missed one of these Nodes. Maybe ask Muse to create a missing Node"
                
                query = f"""
                    MATCH (p:{node_1_type} {{id: $id1}}), (m:{node_2_type} {{id: $id2}})
                    CREATE (p)-[r:{relationship_type} {{Reason_for_relationship_Because: $var, dynamic: $val}}]->(m)
                    RETURN p, r, m
                    """
                graph.query(query, params={
                        'id1': node1,
                        'id2': node2,
                        'var': Reason_for_relationship_Because_var,
                        'val': True
                    })
                response_result  = f"Relationship with  '{Reason_for_relationship_Because_var}' now exists. with these two nodes, {node1} and {node2}"
                return response_result
            

            '''THis write Location, Item and groups into the story with dynamic = True'''
            @tool
            def write_otherNodes(query):
                
                '''This is to explicitly write a Item, Group, or Location to the story, this means you will make a change to the graph add a component 
                that does not already exist, the query must explicitly state for a new character to be added.
                THIS DOES NOT WRITE TO CHARACTERS, USE write_character'''
                
                response = write_otherNodes_helper(query, story_id)
                # Parse the JSON response
                filled_values = response
                #print(filled_values)
                response_result =''

                type = filled_values.get("type", "Unknown")
                name = filled_values.get("name", "Unknown")
                more_info = filled_values.get("more_info", "Unknown")

                #Name of the new node is not clear
                if name == "Unknown" or name == " " or name is None or name == "":
                    return "The character name was not added becasue there is a name missing"

                query = f"""
                MATCH (c:{type}_{story_id} {{id: $id}})
                RETURN c
                """
                result = graph.query(query, params={'id': name})

                # If the result is empty, create the new character
                if not result:
                    query = f"""
                    CREATE (c:{type}_{story_id} {{id: $id, more_info: $more_info, dynamic: $draft,}})
                    """
                    graph.query(query, params={
                        'id': name,
                        'more_info': more_info,
                        'draft': True
                    })
                    response_result  = f"{type} with id '{name}' now exists."
                else:
                    response_result = f"{type} with id '{name}' already exists."

                #print("New Character is added")
                return response_result


            '''Writing a character to the story with dynamic=true'''
            @tool
            def write_character(query):
                
                '''This is to explicitly write a character to the story, this means you will mkae a change to the graph add a component, things like "write a character W into the story" or "Make a character ..."
                that does not already exist, the query must explicitly state for a new character to be added. If not a character then use write_otherNodes instead'''
                
                
                response = write_character_helper(query, story_id)
               
                # Parse the JSON response
                filled_values = response
                #print(filled_values)
                response_result =''

                character_name = filled_values.get("character_name", "Unknown")
                character_age = filled_values.get("character_age", "Unknown")
                character_hair_color = filled_values.get("character_hair_color", "Unknown")
                character_height = filled_values.get("character_height", "Unknown")
                character_gender = filled_values.get("character_gender", "Unknown")
                character_role = filled_values.get("character_role", "Minor Role")

                #Node name is not clear
                if character_name == "Unknown" or character_name == " " or character_name is None or character_name == "":
                    return "The character name was not added because there is a name missing"

                query = f"""
                MATCH (c:Character_{story_id} {{id: $id}})
                RETURN c
                """
                result = graph.query(query, params={'id': character_name})

                # If the result is empty, create the new character
                if not result:
                    query = f"""
                    CREATE (c:Character_{story_id} {{id: $id, hair_color: $hair_color, height: $height, age: $age, gender: $gender, role: $role, dynamic: $draft}})
                    """
                    graph.query(query, params={
                        'id': character_name,
                        'hair_color': character_hair_color,
                        'height': character_height,
                        'age': character_age + "",
                        'gender':character_gender,
                        "role": character_role,
                        'draft': True
                    })
                    response_result  = f"Character with id '{character_name}' now exists."
                else:
                    response_result = f"Character with id '{character_name}' already exists."

                #print("New Character is added")
                return response_result

            '''Professional Advice uses Dreamscibe documents to aid the User queries with advice'''
            @tool
            def professional_advise(query: str):
                '''This tool betters the story and provides details on general story elements. Use ALWAYS when character archtypes are refered to in the query. 
                Examples of Character Archetypes 'Caregiver', 'Creator', 'Everyman', 'Explorer', 'Hero', 'Innocent', and 'Mentor' "
                Questions that seem like "Should I add...", "Could this idea work...", "Give some input..." on character Archtypes and story elements should use this tool. 
                Other examples are "Who would be a good Hero for Draken?" or "How is a hero supposed to work with Luciferoiu?"
                USE THIS TOOL to answer a query asked about Advice on How a story can be improved. 
                
                '''
            
                #print(advise_qa)
                with get_openai_callback() as cb:
                    response = advise_qa.invoke(query)
                #print(cb.total_tokens)
                return response
                #return advise_qa.invoke(query)


            '''Vector Rag implementation of the story'''
            @tool
            def vector_search(query: str):
                '''This tool should be used to find details about the story that pertain to story chronology, events
                and when character did action this tool is chosen at last resort. This tool also helps with understanding the story flow like when or what the climax event was. 
                an Example would be "Why did a character do an action?". This handles the Why parts of the story.
                DO NOT USE THIS TOOL to answer a query asked about Advice on How a story can be improved
                DO NOT USE THIS TOOL FOR WHERE, WHO, and WHEN type questions unles Cypher_check returned empty before
                Args:
                    query: the question to ask the model
                This tool is used for specific questioning.'''
                
                #print("hello")
                response = vector_search_helper(query, story_id)
                return response

            '''Graph Rag implementation of the story'''
            @tool
            def cypher_check(query):
                '''This tool should be used to find details about the story focussing on relationships and the reason for the relatonships between Nodes like Characters, Items, Locations, Organizations. 
                This can be used to find the relationships between entities in the query. This is also a good use of counting how many of certain enteties that there are and their realtionships. Questions like Number of locations, or where characters travelled or who they liked is noted down in this tool.
                The tools answers the where and who type questions. This tool can also quickly verify things like age, hair color and more.
                Most story questions need this tool
                This tool can be used multiple times when breaking up queries
                Args:
                    query: the question to ask the model
                '''
                graph.refresh_schema
                k  = parse_graph_schema(graph.get_structured_schema, story_id)
                graphFilteredSchema = pretty_print_schema(k)

                node_helper = graph.query(f'''MATCH (n)
                WHERE n:Character_{story_id} OR n:Item_{story_id} OR n:Location_{story_id} OR n:Group_{story_id}
                RETURN n.id AS name, labels(n) AS nodeType''')

                cleanHelper = format_node_data(node_helper)

                cypher_generation_prompt =  cypher_check_helper(story_id,graphFilteredSchema, str_nodes, cleanHelper)
                
                #print(graphFilteredSchema)

                try:
                    qa_chain = GraphCypherQAChain.from_llm(
                        graph=graph, llm=llm, verbose=True,  
                        validate_cypher=True,
                        #cypher_query_corrector=[cypher_validation],
                        cypher_prompt=cypher_generation_prompt,
                        use_function_response=True
                    )
                    
                    with get_openai_callback() as cb:
                        response = qa_chain.invoke(query)
                        response2 = vector_search(query)
                        
                        response['result'] += " " +response2['result']
                       
                    #print("From Cypher")
                    # print(f"Prompt Tokens: {cb.prompt_tokens}")
                    # print(f"Completion Tokens: {cb.completion_tokens}")
                except Exception as e:
                    return "Something Went wrong in the Cypher generation, use cypher_check again on the same query"
                return response 

            #Note to future Devs Vector Search was added to the end of cypher search because the accuracy of cypher_check was not perfect enough
            #future iterations should attempt to remove this issue and have both tools present
            tools = [ cypher_check, professional_advise, write_character, write_otherNodes, create_relationship, delete_dynamic_Node]
            #cypher_check,
            #3232llm_with_tools = [convert_to_openai_function(t) for t in tools]
            
            prompt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder("chat_history", optional=True),
                    ("system",
                        f"""You are a helpful assistant. Your name is 'Muse'. Utilize all tools and context provided to generate the best responses. Employ the tools effectively, combining their details to produce comprehensive and accurate answers. Create an engaging conversation by including meaningful follow-up questions after every response.

                        - Only use tools and chat_history to answer questions, never without them.
                        - Find evidence to support claims and be willing to call out incorrect information from the user input.
                        - DO NOT ANSWER ANY QUESTIONS outside of the context provided.
                        - Only answer using the given context.
                        - Use the chat history or tools to answer all questions. Be willing to argue with the user when necessary with information from chat_history and tools. Do NOT be overly agreeable; always use evidence.
                        - Answer all questions with supporting evidence.
                        - User queries may imply incorrect information. For example, "Since ExampleCharacter has green hair..." might be incorrect. Use Cypher_check to verify its accuracy, if wrong then, Your response should be, "ExampleCharacter does not have green hair; rather..." and continue the explanation.
                        - Human input may contain spelling errors in names. Use the node names provided to correct the query, assuming the query is not writing to the story with tools like write_character or write_otherNodes.
                        - If giving the user ideas for a character, keep the ideas simple and less descriptive to let the user think on their own.
                        - Important, If No tools are used and Chat_history is empty or no related to the human input return only "I am not sure that this is relevent to the conversation"


                        Nodes in Story: {str_nodes}
                        """
                    ),
                    
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )

            # Construct the Tool Calling Agent
            agent = create_tool_calling_agent(llm, tools, prompt)
            chat_history = memory.buffer_as_messages

            # Create an agent executor by passing in the agent and tools
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, 
            memory=memory,
            
            max_iterations=10,)

            # Run Agent
            with get_openai_callback() as cb:
                response = agent_executor.invoke(
                    {
                        "input": query ,
                        "chat_history": chat_history,
                    })
                # print(f"Total Tokens: {cb.total_tokens}")
                # print(f"Prompt Tokens: {cb.prompt_tokens}")
                # print(f"Completion Tokens: {cb.completion_tokens}")
                # print(f"Total Cost (USD): ${cb.total_cost}")
                                             
            print(response)
            return response
if __name__ == "__main__":
     #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
     k = None
     while True:
        statement = input("Please type a statement and press Enter to continue: ")
        story_id_specific = 6
        if k is None:
            k =prompt_graph(statement , memory_store=None, story_id=story_id_specific)
            #print("Here is the Memory")
        else: 
            
            k =prompt_graph(statement ,memory_store = k['chat_history'], story_id= story_id_specific)
        

    

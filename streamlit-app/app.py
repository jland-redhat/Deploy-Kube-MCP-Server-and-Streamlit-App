import os
import traceback
from httpx import URL
from llama_stack_client.lib.agents.tool_parser import ToolParser
from llama_stack_client.types.agents.turn_create_params import ToolConfig
import streamlit as st
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger
import uuid
from pathlib import Path

# Load environment variables
load_dotenv()

# Constants
MODEL_ID = os.getenv("INFERENCE_MODEL_ID", "llama32-3b")
MODEL_PROMPT = """You are a helpful OpenShift assistant. You have access to OpenShift cluster tools and resources.
Help users with their OpenShift deployments, troubleshooting, and best practices.
When discussing OpenShift concepts, provide clear and accurate information."""

# Get the absolute path to the assets directory
BASE_DIR = Path(__file__).parent.absolute()
LOGO_PATH = BASE_DIR / "assets" / "redhat-car.png"

# Create a header with logo and title
header_col1, header_col2 = st.columns([1, 4])

with header_col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=100)  # Smaller logo for the header
    else:
        st.warning(f"Logo image not found at: {LOGO_PATH}")

with header_col2:
    st.markdown("""
    <h1 style='font-family: "Comic Sans MS", cursive, sans-serif; 
                color: #FF6B35; 
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
                margin-top: 10px;'>
        Open-Shifter!!! 🚀
    </h1>
    """, unsafe_allow_html=True)
    st.caption("Your AI-powered OpenShift assistant")

# Apply Fun Orange Theme
st.markdown("""
    <style>
    :root {
        --primary-color: #FF6B35;  /* Orange */
        --background-color: #FFF8F0;  /* Light orange background */
        --secondary-background-color: #FFFFFF;
        --text-color: #2D2D2D;  /* Darker text for better contrast */
        --accent-color: #FF9F1C;  /* Brighter orange for accents */
    }
    
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
        background-image: linear-gradient(135deg, #FFF8F0 0%, #FFE8D6 100%);
    }
    
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #E65C2E;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stTextInput>div>div>input {
        border: 2px solid var(--primary-color);
        border-radius: 10px;
        padding: 8px 12px;
    }
    
    .stSelectbox>div>div>div {
        color: var(--text-color);
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--primary-color);
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    
    .stSidebar {
        background-color: white;
        border-right: 1px solid #FFD9C0;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }
    
    .stAlert {
        border-left: 4px solid var(--primary-color);
        border-radius: 8px;
    }
    
    /* Add some fun hover effects */
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Style the chat messages */
    .stChatMessage {
        border-radius: 15px;
        padding: 12px;
        margin: 8px 0;
    }
    
    /* Add a subtle pattern to the background */
    body::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        opacity: 0.03;
        z-index: -1;
        background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23FF6B35' fill-opacity='0.3' fill-rule='evenodd'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/svg%3E");
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Llama Stack client
def get_llama_client() -> LlamaStackClient:
    base_url = os.getenv("REMOTE_BASE_URL")
    if not base_url:
        st.error("REMOTE_BASE_URL environment variable is not set")
        st.stop()
    
    # Get Tavily API key if available
    tavily_search_api_key = os.getenv("TAVILY_SEARCH_API_KEY")
    provider_data = {"tavily_search_api_key": tavily_search_api_key} if tavily_search_api_key else None
    
    # Initialize the client
    try:
        client = LlamaStackClient(
            base_url=base_url,
            provider_data=provider_data
        )
        st.toast("Connected to Open-Shifter service", icon="✅")
        return client
    except Exception as e:
        st.error(f"Failed to initialize Open-Shifter client: {str(e)}")
        st.stop()

# Initialize MCP servers and RAG
def initialize_llama_stack(client: LlamaStackClient):
    # Set up sampling parameters
    temperature = float(os.getenv("TEMPERATURE", 0.0))
    if temperature > 0.0:
        top_p = float(os.getenv("TOP_P", 0.95))
        strategy = {"type": "top_p", "temperature": temperature, "top_p": top_p}
    else:
        strategy = {"type": "greedy"}
    
    max_tokens = int(os.getenv("MAX_TOKENS", 512))
    
    # Store sampling parameters in session state
    st.session_state.sampling_params = {
        "strategy": strategy,
        "max_tokens": max_tokens,
    }
    
    # Initialize MCP servers from environment variables
    st.session_state.mcp_servers = {}
    


def add_mcp_server(server_name: str, server_url: str):
    """Add an MCP server to the list of registered servers"""
    try:
        registered_tools = st.session_state.llama_client.tools.list()
        registered_toolgroups = [tool.toolgroup_id for tool in registered_tools]
        
        if f"mcp::{server_name}" not in registered_toolgroups:
            st.session_state.llama_client.toolgroups.register(
                toolgroup_id=f"mcp::{server_name}",
                provider_id="model-context-protocol",
                mcp_endpoint=dict(uri=server_url)
            )
            if server_name not in st.session_state.registered_mcp_servers:
                st.session_state.registered_mcp_servers.append(server_name)
            return True, f"Successfully registered MCP server: {server_name}"
        else:
            if server_name not in st.session_state.registered_mcp_servers:
                st.session_state.registered_mcp_servers.append(server_name)
            return True, f"MCP server '{server_name}' already registered"
    except Exception as e:
        return False, f"Failed to register MCP server: {str(e)}"

def add_mcp_server_form():
    """Display form to add a new MCP server"""
    with st.form("add_mcp_server"):
        st.subheader("Add MCP Server")
        server_name = st.text_input("Server Name", 
                                 help="A unique name to identify this MCP server")
        server_url = st.text_input("Server URL", 
                                 help="Base URL of the MCP server (e.g., http://mcp-server:8080)")
        
        if st.form_submit_button("Add Server"):
            if not server_name or not server_url:
                st.error("Please provide both server name and URL")
                return
                
            success, message = add_mcp_server(server_name, server_url)
            if success:
                st.toast(message, icon="✅")
            else:
                st.error(message)
            st.session_state.show_add_mcp = False
            st.rerun()

# Initialize session state
if 'llama_client' not in st.session_state:
    st.session_state.llama_client = get_llama_client()
    initialize_llama_stack(st.session_state.llama_client)

if 'agents' not in st.session_state:
    st.session_state.agents = {}
if 'active_agent' not in st.session_state:
    st.session_state.active_agent = None
if 'mcp_servers' not in st.session_state:
    st.session_state.mcp_servers = {}
if 'registered_mcp_servers' not in st.session_state:
    st.session_state.registered_mcp_servers = []

# Register default OpenShift MCP server if not already registered
default_server_name = "openshift"
default_server_url = os.getenv("REMOTE_OCP_MCP_URL", "http://ocp-mcp-server:8000/sse")
success, message = add_mcp_server(default_server_name, default_server_url)

def create_agent(name=None):
    """Create a new agent with a unique ID and optional name
    
    Args:
        name (str, optional): Name for the agent. If None, will use a default name.
    """
    agent_id = f"agent_{len(st.session_state.agents) + 1}"
    if name is None:
        name = f"Agent {len(st.session_state.agents) + 1}"
    
    # Set up agent configuration
    temperature = float(os.getenv("TEMPERATURE", 0.0))
    if temperature > 0.0:
        top_p = float(os.getenv("TOP_P", 0.95))
        strategy = {"type": "top_p", "temperature": temperature, "top_p": top_p}
    else:
        strategy = {"type": "greedy"}
    
    max_tokens = int(os.getenv("MAX_TOKENS", 512))
    
    st.session_state.agents[agent_id] = {
        'id': agent_id,
        'name': name,
        'messages': [],
        'steps': [],
        'config': {
            'model': MODEL_ID,
            'sampling_params': {
                'strategy': strategy,
                'max_tokens': max_tokens
            }
        }
    }
    st.session_state.active_agent = agent_id

def send_message(agent_id: str, message: str):
    """Send a message to the agent and process the response"""
    if not message.strip() or 'llama_client' not in st.session_state:
        return
    
    agent = st.session_state.agents[agent_id]
    
    # Add user message to chat
    agent['messages'].append({
        'role': 'user',
        'content': message
    })

    agent['steps'].append({
        'type': 'user_input',
        'color': 'white',
        'content': message
    })
    
    
    try:
        # Create a new agent if it doesn't exist
        if 'agent' not in agent:
            st.error("Agent not found")
            return
        
        if 'session_id' not in agent:
            # Create a new session for the agent
            session_name = f"{agent['name']}_session"
            agent['session_id'] = agent['agent'].create_session(session_name=session_name)
        
        # Send message to the agent using create_turn
        response = agent['agent'].create_turn(
            messages=[{"role": "user", "content": message}],
            session_id=agent['session_id']
        )

        # Display steps in the sidebar
        message_placeholder = st.empty()
        full_response = ""

        for log in AgentEventLogger().log(response):
            log.print()

            if log.role == 'tool_execution':
                agent['steps'].append({
                    'type': 'tool_execution',
                    'color': log.color if hasattr(log, 'color') and log.color is not None else 'white',
                    'content': log.content
                })
            if log.role == None:
                full_response += log.content
                message_placeholder.markdown(full_response + "▌")


        agent['messages'].append({
            'role': 'assistant',
        'content': full_response
        })
        
    except Exception as e:
        st.error(f"Error processing message: {str(e)}")
        traceback.print_exc()
        # TODO: Add this back currently there is a weird issue with the client where ToolParser is no implemented
        # agent['messages'].append({
        #     'role': 'assistant',
        #     'content': f"Sorry, I encountered an error: {str(e)}"
        # })
        # So for now we just print as much of the message as we got
        agent['messages'].append({
            'role': 'assistant',
            'content': full_response
        })

# Sidebar for MCP server management
with st.sidebar:
    st.title("MCP Servers")
    
    # Show registered servers
    if st.session_state.registered_mcp_servers:
        st.subheader("Registered Servers")
        for server in st.session_state.registered_mcp_servers:
            st.markdown(f"- {server}")
    else:
        st.info("No MCP servers registered yet")
    
    # Add Server button
    if st.button("➕ Add MCP Server", key="add_mcp_button"):
        st.session_state.show_add_mcp = True

    # Show add server form if button was clicked
    if st.session_state.get('show_add_mcp', False):
        add_mcp_server_form()
        
        # Add a cancel button
        if st.button("Cancel"):
            st.session_state.show_add_mcp = False
            st.rerun()

# Main app
st.title("")

# Agent management
with st.expander("➕ Create New Agent", expanded=False):
    with st.form("create_agent_form"):
        agent_name = st.text_input("Agent Name", key="new_agent_name", value="🔍 Inspector")
        
        # Show checkboxes for available MCP servers
        st.write("Attach MCP Servers:")
        selected_servers = []
        for server in st.session_state.get('registered_mcp_servers', []):
            if st.checkbox(server, key=f"mcp_server_{server}"):
                selected_servers.append(server)
        
        if st.form_submit_button("Create Agent"):
            if agent_name:
                # Create agent with selected MCP servers
                agent_tools = [f"mcp::{server}" for server in selected_servers]
                agent_tools.append("builtin::websearch")
                
                # Create the agent with the selected tools
                agent = Agent(
                    client=st.session_state.llama_client,
                    instructions=MODEL_PROMPT,
                    model=MODEL_ID,
                    tools=agent_tools,
                    tool_parser=ToolParser(),
                    tool_config=ToolConfig(
                        tool_choice="auto"
                    ),
                    sampling_params=st.session_state.sampling_params,
                )
                
                # Store the agent
                agent_id = str(uuid.uuid4())
                st.session_state.agents[agent_id] = {
                    'name': agent_name,
                    'agent': agent,
                    'messages': [],
                    'steps': [],
                    'config': {
                        'model': MODEL_ID,
                        'tools': agent_tools
                    }
                }
                st.session_state.active_agent = agent_id
                st.success(f"Agent '{agent_name}' created successfully!")
                st.rerun()
            else:
                st.error("Please provide an agent name")

# Agent selection dropdown
if st.session_state.agents:
    agent_names = {id: agent['name'] for id, agent in st.session_state.agents.items()}
    selected_agent = st.selectbox(
        "Select Agent",
        options=list(agent_names.keys()),
        format_func=lambda x: agent_names[x],
        index=next((i for i, id in enumerate(agent_names) if id == st.session_state.active_agent), 0)
    )
    st.session_state.active_agent = selected_agent

# Chat interface
if st.session_state.active_agent:
    agent = st.session_state.agents[st.session_state.active_agent]
    
    
    # Display chat messages
    st.subheader(f"Chat with {agent['name']}")
    
    # Chat container
    chat_container = st.container()
    
    # Steps container (right side)
    steps_container = st.sidebar.container()
    
    with chat_container:
        for msg in agent['messages']:
            with st.chat_message(msg['role']):
                st.write(msg['content'])

    
    with steps_container:
        st.subheader("Processing Steps")
        if agent['steps']:
            for i, step in enumerate(agent['steps'], 1):
                step_type = step.get('type', 'Step')
                step_color = step.get('color', '#f0f0f0')
                is_user_input = 'user_input' in step_type.lower() or 'user' in step_type.lower()
                
                # Create columns for the step indicator and content
                col1, col2 = st.columns([0.05, 0.95])
                
                # Step indicator with color
                with col1:
                    st.markdown(
                        f'<div style="height: 30px; display: flex; align-items: center; justify-content: center; color: {step_color}; font-weight: bold;">•</div>',
                        unsafe_allow_html=True
                    )
                
                # Step content
                with col2:

                    with st.expander(f"Step {i}: {step_type}", expanded=is_user_input):
                        content = step.get('content', 'No content')
                        if is_user_input:
                            st.info(content)
                        elif isinstance(content, dict):
                            st.json(content)
                        else:
                            st.text(content)
        else:
            st.info("No steps available. Send a message to see the processing steps.")
    
    # Input for new message
    # default_question = "Who did the panthers draft in 2025?"
    prompt = st.chat_input("Enter your question here")
    
    # Check if we should send the default question (first load)
    if 'default_question_sent' not in st.session_state and 'default_question' in locals():
        st.session_state.default_question_sent = True
        send_message(st.session_state.active_agent, default_question)
        st.rerun()
    # Handle user input
    elif prompt:
        send_message(st.session_state.active_agent, prompt)
        st.rerun()


else:
    st.info("Create a new agent using the '+' button to get started.")

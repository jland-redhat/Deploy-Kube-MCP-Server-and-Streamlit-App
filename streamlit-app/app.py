import os
import json
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional, Set, Tuple
from rich.pretty import pprint
from termcolor import cprint
from utils import step_printer
from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger
import uuid

# Load environment variables
load_dotenv()

# Constants
MODEL_ID = "llama32-3b"
MODEL_PROMPT= """You are a helpful assistant. You have access to a number of tools.
Whenever a tool is called, be sure return the Response in a friendly and helpful tone."""

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
        st.toast("Successfully connected to Llama Stack server", icon="✅")
        return client
    except Exception as e:
        st.error(f"Failed to initialize LlamaStackClient: {str(e)}")
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
    
    # Register OpenShift MCP server if URL is provided
    ocp_mcp_url = os.getenv("REMOTE_OCP_MCP_URL")
    if ocp_mcp_url and ocp_mcp_url != "http://openshift-mcp-server:8080":
        try:
            client.toolgroups.register(
                toolgroup_id="mcp::openshift",
                provider_id="model-context-protocol",
                mcp_endpoint={"uri": ocp_mcp_url}
            )
            add_mcp_server("OpenShift MCP", ocp_mcp_url)
            st.toast("Successfully registered OpenShift MCP server", icon="✅")
        except Exception as e:
            st.error(f"Failed to register OpenShift MCP server: {str(e)}")
    
    # Register Slack MCP server if URL is provided
    slack_mcp_url = os.getenv("REMOTE_SLACK_MCP_URL")
    if slack_mcp_url and slack_mcp_url != "http://slack-mcp-server:8080":
        try:
            client.toolgroups.register(
                toolgroup_id="mcp::slack",
                provider_id="model-context-protocol",
                mcp_endpoint={"uri": slack_mcp_url}
            )
            add_mcp_server("Slack MCP", slack_mcp_url)
            st.toast("Successfully registered Slack MCP server", icon="✅")
        except Exception as e:
            st.error(f"Failed to register Slack MCP server: {str(e)}")

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
            },
            'tools': ["builtin::websearch"]
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
            # Create the agent
            agent['agent'] = Agent(
                client=st.session_state.llama_client,
                instructions=MODEL_PROMPT,
                model=agent['config']['model'],
                sampling_params=st.session_state.sampling_params,
                tools=agent['config'].get('tools', []),
                tool_config={"tool_choice":"auto"}
            )
            
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
                    'color': log.color,
                    'content': log.content
                })
            if log.role == None:
                full_response += log.content
                message_placeholder.markdown(full_response + "▌")


        agent['messages'].append({
            'role': 'assistant',
        'content': full_response
        })

        # Store steps for display if available
        if hasattr(response, 'steps'):
            for step in response.steps:
                step_info = {
                    'type': getattr(step, 'type', 'Step'),
                    'content': str(step)
                }
                agent['steps'].append(step_info)
        
    except Exception as e:
        st.error(f"Error processing message: {str(e)}")
        agent['messages'].append({
            'role': 'assistant',
            'content': f"Sorry, I encountered an error: {str(e)}"
        })

def add_mcp_server(name: str, url: str):
    """Add a new MCP server"""
    server_id = name.lower().replace(' ', '_')
    st.session_state.mcp_servers[server_id] = {
        'name': name,
        'url': url
    }

# Sidebar for MCP server management
with st.sidebar:
    st.title("MCP Servers")
    
    # Display existing MCP servers
    for server_id, server in st.session_state.mcp_servers.items():
        with st.expander(f"{server['name']} ({server_id})"):
            st.text(f"URL: {server['url']}")
    
    # Add new MCP server
    with st.expander("Add MCP Server", expanded=False):
        with st.form("add_mcp_server"):
            server_name = st.text_input("Server Name")
            server_url = st.text_input("Server URL")
            if st.form_submit_button("Add Server"):
                if server_name and server_url:
                    add_mcp_server(server_name, server_url)
                    st.rerun()

# Main app
st.title("LLM Agent Chat")

# Agent management
with st.expander("➕ Create New Agent", expanded=False):
    agent_name = st.text_input(
        "Agent Name",
        value=f"Agent {len(st.session_state.agents) + 1}",
        help="Enter a name for your new agent"
    )
    if st.button(
        "✨ Create Agent",
        help="Create a new agent with the specified name",
        type="primary",
        use_container_width=True,
    ):
        create_agent(agent_name)
        st.toast(f"Agent '{agent_name}' created successfully!", icon="🤖")

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
    default_question = "Who did the panthers draft in 2025?"
    prompt = st.chat_input(default_question)
    
    # Check if we should send the default question (first load)
    if 'default_question_sent' not in st.session_state:
        st.session_state.default_question_sent = True
        send_message(st.session_state.active_agent, default_question)
        st.rerun()
    # Handle user input
    elif prompt:
        send_message(st.session_state.active_agent, prompt)
        st.rerun()

else:
    st.info("Create a new agent using the '+' button to get started.")

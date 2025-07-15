# LLM Agent Chat Application

A Streamlit-based chat interface for interacting with LLM agents.

## Features

- Create multiple chat agents
- View processing steps for each response
- Manage MCP servers (OpenShift and Slack registered by default)
- Environment variable configuration

## Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your API keys and URLs.

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

```bash
streamlit run app.py
```

## Usage

1. Click the "+" button to create a new agent
2. Select an agent from the dropdown
3. Type your message and press Enter to send
4. View the response and processing steps in the sidebar
5. Add or manage MCP servers using the sidebar

## Environment Variables

- `REMOTE_BASE_URL`: Base URL for the LLM service
- `TAVILY_SEARCH_API_KEY`: API key for Tavily search
- `OPENSHIFT_MCP_URL`: URL for OpenShift MCP server (default: http://openshift-mcp-server:8080)
- `SLACK_MCP_URL`: URL for Slack MCP server (default: http://slack-mcp-server:8080)

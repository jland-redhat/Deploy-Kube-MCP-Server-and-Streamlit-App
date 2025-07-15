# MCP Server and Streamlit Frontend Helm Chart

This Helm chart deploys two components:
1. An MCP (Model Control Plane) server using the image `quay.io/manusa/kubernetes_mcp_server`
2. A Streamlit frontend for connecting to llama-stack

## Prerequisites

- An OpenShift cluster
- Helm 3
- An instance of llama-stack (required for the Streamlit frontend to function)

## Installing the Chart

To install the chart with the release name `my-release`:

```bash
helm install my-release .
```

## Configuration

The following table lists the configurable parameters of the chart and their default values.

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| `mcp.replicaCount` | Number of MCP server replicas | `1` |
| `mcp.image.repository` | MCP server image repository | `quay.io/manusa/kubernetes_mcp_server` |
| `mcp.image.tag` | MCP server image tag | `latest` |
| `mcp.service.port` | MCP server service port | `8000` |
| `streamlit.replicaCount` | Number of Streamlit frontend replicas | `1` |
| `streamlit.service.port` | Streamlit service port | `8501` |
| `streamlit.route.enabled` | Enable OpenShift Route for Streamlit | `true` |

## Notes

1. The MCP server will be accessible via the internal service on port 8000
2. The Streamlit frontend will be exposed via an OpenShift Route on port 8501
3. Ensure you have a llama-stack instance running before deploying this chart
4. The Streamlit frontend will need to be configured to connect to your llama-stack instance

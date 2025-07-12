# 🔧 Voice-Controlled Multimodal Agent with MCP, Claude, Python & Flowise

## 🧠 Project Overview

This project demonstrates how to build a secure, modular **multimodal AI agent** using the **Model Context Protocol (MCP)**, capable of:

* Accepting **voice input** 🗣️
* Executing **structured tool calls** 🛠️
* Retrieving **image, text, and document resources** 📸
* Performing **RAG (retrieval-augmented generation)** using **Pinecone** 📈
* Delivering results through **Claude Desktop** or **Cursor** 🖥️

## 🧱 Tech Stack

* **Claude Desktop** or **Cursor** (MCP clients)
* **Python** + `modelcontextprotocol` SDK (MCP servers)
* **n8n** for automation triggers
* **Flowise** for tool chaining and vector-based search
* **Pinecone** + SQLite for retrieval
* **OpenAI Whisper** or browser-based voice input

## 💡 Key Capabilities

| Capability | How It's Implemented |
|------------|---------------------|
| **MCP Tool Server** | Built in Python, exposes commands via JSON schema |
| **Multimodal input** | Voice input transcribed and routed via MCP |
| **RAG pipeline** | Search + summarize using Flowise + Pinecone |
| **Security & Privacy** | API key access, SSE monitoring, GDPR-safe logging |
| **Prompt Engineering** | Custom templates defined in Python for Claude/GPT use |
| **Real-time interactivity** | Streamable HTTP + SSE endpoints |
| **Multi-client compatibility** | Tested with Cursor, Claude Desktop, and n8n UI |

## 🗂️ Project Structure

```
Project/
├── mcp_servers/
│   ├── voice_agent_server.py      # Main MCP server
│   ├── multimodal_tools.py        # Image/document processing
│   ├── rag_server.py              # RAG implementation
│   └── security_server.py         # Security & logging
├── config/
│   ├── claude_desktop_config.json # Claude Desktop MCP config
│   ├── cursor_config.json         # Cursor MCP config
│   ├── n8n_workflows.json         # n8n automation workflows
│   └── flowise_config.json        # Flowise tool chains
├── templates/
│   ├── prompt_templates.py        # Custom prompt templates
│   └── response_templates.py      # Response formatting
├── schemas/
│   ├── tool_schemas.json          # JSON schemas for tools
│   └── api_schemas.json           # API endpoint schemas
├── docs/
│   ├── architecture.md            # System architecture
│   ├── setup_guide.md            # Setup instructions
│   └── demo_script.md            # Demo scenarios
├── requirements.txt               # Python dependencies
├── .env.example                  # Environment variables template
└── demo_video_script.md          # Demo video script
```

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start MCP Servers**
   ```bash
   python mcp_servers/voice_agent_server.py
   ```

4. **Configure Claude Desktop/Cursor**
   - Copy `config/claude_desktop_config.json` to your Claude Desktop config
   - Or copy `config/cursor_config.json` to your Cursor config

5. **Test Voice Input**
   - Open Claude Desktop or Cursor
   - Use voice commands like "Search for information about AI"
   - Watch the multimodal agent process your request!

## 🎯 Demo Scenarios

1. **Voice-to-RAG**: "Tell me about machine learning trends"
2. **Image Analysis**: "Analyze this screenshot and explain what's happening"
3. **Document Processing**: "Summarize this PDF and extract key points"
4. **Web Search + Synthesis**: "Research competitors and create a comparison"

## 🔐 Security Features

* API key validation and rotation
* Request filtering and sanitization
* GDPR-compliant logging
* Rate limiting and abuse detection
* Secure credential management

## 📊 Monitoring & Analytics

* Real-time request tracking via SSE
* Performance metrics and latency monitoring
* Error logging and debugging tools
* Usage analytics and reporting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

* Anthropic for Claude and MCP
* OpenAI for Whisper and GPT models
* Pinecone for vector search
* The open-source community for amazing tools

---

**Built with ❤️ for the future of multimodal AI**

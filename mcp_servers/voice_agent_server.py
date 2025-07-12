#!/usr/bin/env python3
"""
Voice-Controlled Multimodal Agent MCP Server
============================================

This server implements the core MCP functionality for the voice-controlled
multimodal agent, including voice input processing, tool orchestration,
and response generation.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime
import base64
import io
import wave
import speech_recognition as sr
from openai import OpenAI
import pinecone
import sqlite3
from pathlib import Path

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    Tool,
    Resource,
    ListResourcesRequest,
    ListResourcesResult,
    ReadResourceRequest,
    ReadResourceResult,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")

if pinecone_api_key:
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)


class VoiceAgentServer:
    """Main server class for the voice-controlled multimodal agent."""
    
    def __init__(self):
        self.server = Server("voice-agent")
        self.db_path = Path("voice_agent.db")
        self.init_database()
        self.setup_handlers()
        
    def init_database(self):
        """Initialize SQLite database for conversation history and caching."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for conversation history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_input TEXT,
                input_type TEXT,
                response TEXT,
                tools_used TEXT,
                session_id TEXT
            )
        ''')
        
        # Create table for cached results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE,
                result TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def setup_handlers(self):
        """Set up MCP server handlers."""
        
        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available tools."""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="transcribe_audio",
                        description="Transcribe audio input to text using Whisper",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "audio_data": {
                                    "type": "string",
                                    "description": "Base64 encoded audio data"
                                },
                                "format": {
                                    "type": "string",
                                    "description": "Audio format (wav, mp3, etc.)",
                                    "default": "wav"
                                }
                            },
                            "required": ["audio_data"]
                        }
                    ),
                    Tool(
                        name="process_voice_command",
                        description="Process voice command and execute appropriate actions",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "Voice command text"
                                },
                                "context": {
                                    "type": "object",
                                    "description": "Additional context for the command",
                                    "default": {}
                                }
                            },
                            "required": ["command"]
                        }
                    ),
                    Tool(
                        name="search_knowledge_base",
                        description="Search the knowledge base using RAG",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query"
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Number of results to return",
                                    "default": 5
                                },
                                "include_metadata": {
                                    "type": "boolean",
                                    "description": "Include metadata in results",
                                    "default": True
                                }
                            },
                            "required": ["query"]
                        }
                    ),
                    Tool(
                        name="analyze_image",
                        description="Analyze image content using multimodal AI",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "image_data": {
                                    "type": "string",
                                    "description": "Base64 encoded image data"
                                },
                                "analysis_type": {
                                    "type": "string",
                                    "description": "Type of analysis to perform",
                                    "enum": ["describe", "extract_text", "identify_objects", "analyze_chart"],
                                    "default": "describe"
                                },
                                "prompt": {
                                    "type": "string",
                                    "description": "Custom prompt for analysis",
                                    "default": ""
                                }
                            },
                            "required": ["image_data"]
                        }
                    ),
                    Tool(
                        name="process_document",
                        description="Process and analyze document content",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "document_data": {
                                    "type": "string",
                                    "description": "Base64 encoded document data"
                                },
                                "document_type": {
                                    "type": "string",
                                    "description": "Type of document",
                                    "enum": ["pdf", "docx", "txt", "md"],
                                    "default": "pdf"
                                },
                                "task": {
                                    "type": "string",
                                    "description": "Task to perform on document",
                                    "enum": ["summarize", "extract_key_points", "answer_questions", "translate"],
                                    "default": "summarize"
                                },
                                "questions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Questions to answer from document"
                                }
                            },
                            "required": ["document_data"]
                        }
                    ),
                    Tool(
                        name="web_search",
                        description="Search the web for information",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query"
                                },
                                "num_results": {
                                    "type": "integer",
                                    "description": "Number of results to return",
                                    "default": 5
                                },
                                "include_snippets": {
                                    "type": "boolean",
                                    "description": "Include content snippets",
                                    "default": True
                                }
                            },
                            "required": ["query"]
                        }
                    ),
                    Tool(
                        name="generate_response",
                        description="Generate contextual response using AI",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "Input prompt"
                                },
                                "context": {
                                    "type": "object",
                                    "description": "Additional context data"
                                },
                                "response_format": {
                                    "type": "string",
                                    "description": "Format of response",
                                    "enum": ["text", "markdown", "json", "summary"],
                                    "default": "text"
                                }
                            },
                            "required": ["prompt"]
                        }
                    ),
                    Tool(
                        name="save_conversation",
                        description="Save conversation to database",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "user_input": {
                                    "type": "string",
                                    "description": "User input"
                                },
                                "response": {
                                    "type": "string",
                                    "description": "Agent response"
                                },
                                "session_id": {
                                    "type": "string",
                                    "description": "Session identifier"
                                },
                                "tools_used": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Tools used in processing"
                                }
                            },
                            "required": ["user_input", "response"]
                        }
                    )
                ]
            )
            
        @self.server.call_tool()
        async def call_tool(request: CallToolRequest) -> CallToolResult:
            """Handle tool calls."""
            try:
                if request.name == "transcribe_audio":
                    return await self.transcribe_audio(request.arguments)
                elif request.name == "process_voice_command":
                    return await self.process_voice_command(request.arguments)
                elif request.name == "search_knowledge_base":
                    return await self.search_knowledge_base(request.arguments)
                elif request.name == "analyze_image":
                    return await self.analyze_image(request.arguments)
                elif request.name == "process_document":
                    return await self.process_document(request.arguments)
                elif request.name == "web_search":
                    return await self.web_search(request.arguments)
                elif request.name == "generate_response":
                    return await self.generate_response(request.arguments)
                elif request.name == "save_conversation":
                    return await self.save_conversation(request.arguments)
                else:
                    raise ValueError(f"Unknown tool: {request.name}")
                    
            except Exception as e:
                logger.error(f"Error in tool call {request.name}: {str(e)}")
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error executing tool {request.name}: {str(e)}"
                    )]
                )
                
        @self.server.list_resources()
        async def list_resources() -> ListResourcesResult:
            """List available resources."""
            return ListResourcesResult(
                resources=[
                    Resource(
                        uri="conversation://history",
                        name="Conversation History",
                        description="Access to conversation history and context",
                        mimeType="application/json"
                    ),
                    Resource(
                        uri="cache://results",
                        name="Cached Results",
                        description="Access to cached search and analysis results",
                        mimeType="application/json"
                    )
                ]
            )
            
        @self.server.read_resource()
        async def read_resource(request: ReadResourceRequest) -> ReadResourceResult:
            """Read resource content."""
            if request.uri == "conversation://history":
                return await self.get_conversation_history()
            elif request.uri == "cache://results":
                return await self.get_cached_results()
            else:
                raise ValueError(f"Unknown resource: {request.uri}")
    
    async def transcribe_audio(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Transcribe audio to text using Whisper."""
        try:
            audio_data = arguments.get("audio_data", "")
            format_type = arguments.get("format", "wav")
            
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            
            # Create temporary file for Whisper
            temp_path = f"temp_audio.{format_type}"
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)
            
            # Transcribe using Whisper
            with open(temp_path, "rb") as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            # Clean up temporary file
            os.remove(temp_path)
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "transcript": transcript,
                        "status": "success",
                        "timestamp": datetime.now().isoformat()
                    })
                )]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "status": "error",
                        "timestamp": datetime.now().isoformat()
                    })
                )]
            )
    
    async def process_voice_command(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Process voice command and determine appropriate actions."""
        try:
            command = arguments.get("command", "")
            context = arguments.get("context", {})
            
            # Analyze command intent
            intent_prompt = f"""
            Analyze the following voice command and determine the appropriate action:
            Command: "{command}"
            Context: {json.dumps(context)}
            
            Determine:
            1. Primary intent (search, analyze, create, etc.)
            2. Required tools
            3. Parameters for tool calls
            4. Expected output format
            
            Return as JSON with structure:
            {{
                "intent": "primary_intent",
                "tools": ["tool1", "tool2"],
                "parameters": {{"key": "value"}},
                "output_format": "text|markdown|json"
            }}
            """
            
            # Use GPT to analyze intent
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that analyzes voice commands and determines appropriate actions."},
                    {"role": "user", "content": intent_prompt}
                ],
                temperature=0.1
            )
            
            intent_analysis = json.loads(response.choices[0].message.content)
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "command": command,
                        "analysis": intent_analysis,
                        "status": "success",
                        "timestamp": datetime.now().isoformat()
                    })
                )]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "status": "error",
                        "timestamp": datetime.now().isoformat()
                    })
                )]
            )
    
    async def search_knowledge_base(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Search knowledge base using RAG."""
        try:
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 5)
            include_metadata = arguments.get("include_metadata", True)
            
            # For demonstration, we'll simulate a search
            # In a real implementation, this would use Pinecone or similar
            results = [
                {
                    "id": f"doc_{i}",
                    "text": f"Knowledge base result {i} for query: {query}",
                    "score": 0.9 - (i * 0.1),
                    "metadata": {"source": f"document_{i}.pdf", "page": i+1} if include_metadata else {}
                }
                for i in range(top_k)
            ]
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "query": query,
                        "results": results,
                        "total_results": len(results),
                        "status": "success",
                        "timestamp": datetime.now().isoformat()
                    })
                )]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "status": "error",
                        "timestamp": datetime.now().isoformat()
                    })
                )]
            )
    
    async def analyze_image(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Analyze image content using multimodal AI."""
        try:
            image_data = arguments.get("image_data", "")
            analysis_type = arguments.get("analysis_type", "describe")
            custom_prompt = arguments.get("prompt", "")
            
            # Create analysis prompt based on type
            prompts = {
                "describe": "Describe what you see in this image in detail.",
                "extract_text": "Extract all text visible in this image.",
                "identify_objects": "Identify and list all objects visible in this image.",
                "analyze_chart": "Analyze this chart or graph and explain the key insights."
            }
            
            prompt = custom_prompt or prompts.get(analysis_type, prompts["describe"])
            
            # For demonstration, we'll return a simulated analysis
            # In a real implementation, this would use GPT-4 Vision or similar
            analysis_result = {
                "analysis_type": analysis_type,
                "prompt": prompt,
                "result": f"Image analysis result for {analysis_type}: This is a simulated analysis of the provided image.",
                "confidence": 0.95,
                "objects_detected": ["object1", "object2", "object3"] if analysis_type == "identify_objects" else [],
                "extracted_text": "Sample extracted text" if analysis_type == "extract_text" else ""
            }
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "analysis": analysis_result,
                        "status": "success",
                        "timestamp": datetime.now().isoformat()
                    })
                )]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "status": "error",
                        "timestamp": datetime.now().isoformat()
                    })
                )]
            )
    
    async def process_document(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Process and analyze document content."""
        try:
            document_data = arguments.get("document_data", "")
            document_type = arguments.get("document_type", "pdf")
            task = arguments.get("task", "summarize")
            questions = arguments.get("questions", [])
            
            # For demonstration, we'll return a simulated document analysis
            # In a real implementation, this would extract and process actual document content
            
            task_results = {
                "summarize": "This is a summary of the document content. The document discusses various topics and provides insights into the subject matter.",
                "extract_key_points": ["Key point 1", "Key point 2", "Key point 3"],
                "answer_questions": {q: f"Answer to: {q}" for q in questions},
                "translate": "This is a translated version of the document content."
            }
            
            result = {
                "document_type": document_type,
                "task": task,
                "result": task_results.get(task, "Unknown task"),
                "word_count": 1500,
                "page_count": 5,
                "processing_time": 2.3
            }
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "document_analysis": result,
                        "status": "success",
                        "timestamp": datetime.now().isoformat()
                    })
                )]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "status": "error",
                        "timestamp": datetime.now().isoformat()
                    })
                )]
            )
    
    async def web_search(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Search the web for information."""
        try:
            query = arguments.get("query", "")
            num_results = arguments.get("num_results", 5)
            include_snippets = arguments.get("include_snippets", True)
            
            # For demonstration, we'll return simulated search results
            # In a real implementation, this would use a search API like Google, Bing, or DuckDuckGo
            
            results = [
                {
                    "title": f"Search Result {i+1} for {query}",
                    "url": f"https://example{i+1}.com/article",
                    "snippet": f"This is a snippet from search result {i+1} about {query}. It provides relevant information..." if include_snippets else "",
                    "domain": f"example{i+1}.com",
                    "timestamp": datetime.now().isoformat()
                }
                for i in range(num_results)
            ]
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "query": query,
                        "results": results,
                        "total_results": len(results),
                        "search_time": 0.5,
                        "status": "success",
                        "timestamp": datetime.now().isoformat()
                    })
                )]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "status": "error",
                        "timestamp": datetime.now().isoformat()
                    })
                )]
            )
    
    async def generate_response(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Generate contextual response using AI."""
        try:
            prompt = arguments.get("prompt", "")
            context = arguments.get("context", {})
            response_format = arguments.get("response_format", "text")
            
            # Enhanced prompt with context
            enhanced_prompt = f"""
            Context: {json.dumps(context)}
            
            User Request: {prompt}
            
            Please provide a comprehensive response in {response_format} format.
            """
            
            # Generate response using GPT
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that provides comprehensive and contextual responses."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=0.7
            )
            
            generated_text = response.choices[0].message.content
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "prompt": prompt,
                        "response": generated_text,
                        "format": response_format,
                        "tokens_used": response.usage.total_tokens,
                        "status": "success",
                        "timestamp": datetime.now().isoformat()
                    })
                )]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "status": "error",
                        "timestamp": datetime.now().isoformat()
                    })
                )]
            )
    
    async def save_conversation(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Save conversation to database."""
        try:
            user_input = arguments.get("user_input", "")
            response = arguments.get("response", "")
            session_id = arguments.get("session_id", "default")
            tools_used = arguments.get("tools_used", [])
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO conversations (user_input, response, session_id, tools_used, input_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_input, response, session_id, json.dumps(tools_used), "voice"))
            
            conn.commit()
            conversation_id = cursor.lastrowid
            conn.close()
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "conversation_id": conversation_id,
                        "status": "success",
                        "timestamp": datetime.now().isoformat()
                    })
                )]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "status": "error",
                        "timestamp": datetime.now().isoformat()
                    })
                )]
            )
    
    async def get_conversation_history(self) -> ReadResourceResult:
        """Get conversation history from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM conversations 
                ORDER BY timestamp DESC 
                LIMIT 50
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            conversations = []
            for row in rows:
                conversations.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "user_input": row[2],
                    "input_type": row[3],
                    "response": row[4],
                    "tools_used": json.loads(row[5]) if row[5] else [],
                    "session_id": row[6]
                })
            
            return ReadResourceResult(
                contents=[TextContent(
                    type="text",
                    text=json.dumps({
                        "conversations": conversations,
                        "total_count": len(conversations),
                        "status": "success"
                    })
                )]
            )
            
        except Exception as e:
            return ReadResourceResult(
                contents=[TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "status": "error"
                    })
                )]
            )
    
    async def get_cached_results(self) -> ReadResourceResult:
        """Get cached results from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM cache 
                WHERE expires_at > datetime('now')
                ORDER BY timestamp DESC 
                LIMIT 20
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            cached_results = []
            for row in rows:
                cached_results.append({
                    "id": row[0],
                    "query_hash": row[1],
                    "result": json.loads(row[2]) if row[2] else {},
                    "timestamp": row[3],
                    "expires_at": row[4]
                })
            
            return ReadResourceResult(
                contents=[TextContent(
                    type="text",
                    text=json.dumps({
                        "cached_results": cached_results,
                        "total_count": len(cached_results),
                        "status": "success"
                    })
                )]
            )
            
        except Exception as e:
            return ReadResourceResult(
                contents=[TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "status": "error"
                    })
                )]
            )


async def main():
    """Main function to start the MCP server."""
    voice_agent = VoiceAgentServer()
    
    async with stdio_server() as (read_stream, write_stream):
        await voice_agent.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="voice-agent",
                server_version="1.0.0",
                capabilities=voice_agent.server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())

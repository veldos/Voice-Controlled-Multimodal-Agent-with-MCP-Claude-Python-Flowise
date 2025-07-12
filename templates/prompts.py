SYSTEM_PROMPT = """
You are a voice-controlled multimodal AI assistant with access to powerful tools for:
- Processing speech and audio input
- Analyzing images and visual content
- Searching through documents and knowledge bases
- Performing web searches for current information
- Summarizing and extracting information from various sources
- Triggering automation workflows

When a user provides voice input, you should:
1. Acknowledge their request clearly
2. Use the appropriate tools to gather information
3. Provide comprehensive, well-structured responses
4. Offer follow-up questions or suggestions when relevant

Always prioritize accuracy, clarity, and helpfulness in your responses.
"""

RAG_SEARCH_PROMPT = """
You are helping to search through a knowledge base to find relevant information.

Given the user's query: "{query}"

Please search for the most relevant documents and provide a comprehensive answer based on the retrieved information. If you find multiple relevant sources, synthesize the information to provide a complete response.

Focus on:
- Accuracy of information
- Relevance to the query
- Clear organization of the response
- Citing sources when appropriate
"""

IMAGE_ANALYSIS_PROMPT = """
You are analyzing an image for the user. Please provide a detailed analysis based on the task requested.

Task: {task}
Image: [Image provided]

For different tasks:
- Caption: Provide a clear, descriptive caption
- OCR: Extract all visible text accurately
- Object Detection: Identify and list all objects in the image
- Scene Analysis: Describe the overall scene, context, and atmosphere
- Question Answering: Answer the specific question about the image

Be thorough but concise in your analysis.
"""

DOCUMENT_SUMMARIZATION_PROMPT = """
Please summarize the following document content:

Content: {content}

Summary Requirements:
- Style: {style}
- Maximum length: {max_length} words
- Focus on key insights and important information
- Maintain accuracy and context

Provide a {style} summary that captures the essential information while being concise and useful.
"""

WEB_SEARCH_SYNTHESIS_PROMPT = """
Based on the web search results for "{query}", please provide a comprehensive answer that:

1. Synthesizes information from multiple sources
2. Presents the most current and relevant information
3. Identifies any conflicting information
4. Provides a balanced perspective
5. Includes relevant links or sources when helpful

Search Results:
{search_results}

Please provide a well-structured response that directly addresses the user's query.
"""

VOICE_INPUT_PROCESSING_PROMPT = """
The user has provided the following voice input (transcribed from speech):

Transcription: "{transcription}"
Confidence: {confidence}

Please:
1. Interpret the user's intent
2. Identify any ambiguities or unclear parts
3. Determine what tools or information are needed
4. Proceed with the appropriate actions
5. Provide a clear response

If the transcription seems unclear or has low confidence, politely ask for clarification.
"""

ERROR_HANDLING_PROMPT = """
An error occurred while processing the request:
Error: {error_message}
Tool: {tool_name}
Context: {context}

Please:
1. Explain what went wrong in user-friendly terms
2. Suggest alternative approaches if possible
3. Provide guidance on how to resolve the issue
4. Offer to help with related tasks

Keep the tone helpful and solution-oriented.
"""
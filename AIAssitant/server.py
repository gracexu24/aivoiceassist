"""
FastAPI server for interacting with the Jarvis agent.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import asyncio
from jarvis.agent import root_agent
from google.adk.agents import invocation_context
from google.adk.agents.session_service import SessionService
import uuid

app = FastAPI(title="Jarvis AI Assistant API", version="1.0.0")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

# In-memory session storage (use a proper database in production)
sessions: dict[str, SessionService] = {}

def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, SessionService]:
    """Get existing session or create a new one."""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]
    
    # Create new session
    new_session_id = str(uuid.uuid4())
    session_service = SessionService()
    sessions[new_session_id] = session_service
    return new_session_id, session_service

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Jarvis AI Assistant API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatMessage):
    """
    Send a message to the agent and get a response.
    
    Args:
        request: ChatMessage containing the message and optional session_id
        
    Returns:
        ChatResponse with the agent's response and session_id
    """
    try:
        # Get or create session
        session_id, session_service = get_or_create_session(request.session_id)
        
        # Create invocation context
        ctx = invocation_context.InvocationContext(
            session_service=session_service,
            invocation_id=str(uuid.uuid4()),
            agent=root_agent,
            session=session_service.get_or_create_session(session_id)
        )
        
        # Add user message to context
        from google.genai.types import Content, Part
        user_content = Content(
            role="user",
            parts=[Part(text=request.message)]
        )
        ctx.session.add_content(user_content)
        
        # Run the agent
        response_text = ""
        async for event in root_agent.run_async(ctx):
            # Collect text from events
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
            elif hasattr(event, 'text'):
                response_text += event.text
        
        # If no response text collected, try to get from session
        if not response_text:
            # Get the last assistant message from session
            contents = ctx.session.get_contents()
            if contents:
                last_content = contents[-1]
                if hasattr(last_content, 'parts'):
                    for part in last_content.parts:
                        if hasattr(part, 'text'):
                            response_text += part.text
        
        if not response_text:
            response_text = "I received your message but couldn't generate a response."
        
        return ChatResponse(response=response_text, session_id=session_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    return {"sessions": list(sessions.keys())}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import Optional

app = FastAPI()

# Configure CORS for your GitHub Pages site
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://easydoer.com", 
        "http://easydoer.com",
        "https://www.easydoer.com",
        "http://www.easydoer.com",
        "http://localhost:3000", 
        "http://localhost:8000",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use Gemini 1.5 Flash (fastest, free tier)
model = genai.GenerativeModel('gemini-2.0-flash')

# Store conversation history in memory (for demo purposes)
# In production, use Redis or a database
conversations = {}

class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

# Your professional information for RAG-like responses
PROFESSIONAL_CONTEXT = """
You are an AI assistant for Prabhakar Raturi's professional website. Here's key information about him:

- IT Professional with 18+ years of experience in technical program management
- Location: Greater Seattle Area, Washington
- Email: aaccela@gmail.com
- Phone: 425-471-2980
- Certifications: PMP, CSM
- Focus: Digital Transformation, Cloud Migration, Identity Governance

Key Accomplishments:
1. Enterprise Legacy Modernization: Led modernization from mainframe to Workday HCM
2. Identity Governance: 5-year SailPoint program, 85% efficiency improvement
3. Manufacturing Digital Transformation: Industry 4.0 across 4 global factories, 40% labor savings
4. Cloud Transformation: 130+ applications migrated to AWS/Azure
5. Cybersecurity Framework: NIST implementation, improved ratings from 1.31 to 2.10

Expertise: ERP (Oracle, Workday, SAP), SailPoint, Python, AWS, Azure, PowerBI, MES, WMS, TMS
Organizations: Washington State, GE, Metrolinx, Glanbia Nutritionals, New York State, AT&T

When answering questions about Prabhakar, use this information. For other topics, respond naturally as a helpful AI assistant.
"""

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_msg: ChatMessage):
    try:
        conversation_id = chat_msg.conversation_id
        
        # Initialize conversation history if new
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        # Add context about Prabhakar for portfolio-related queries
        user_message = chat_msg.message.lower()
        is_portfolio_query = any(keyword in user_message for keyword in 
            ["prabhakar", "experience", "skills", "work", "accomplishment", 
             "project", "contact", "email", "phone", "background", "who are you"])
        
        # Prepare the prompt
        if is_portfolio_query:
            prompt = f"{PROFESSIONAL_CONTEXT}\n\nUser Question: {chat_msg.message}"
        else:
            prompt = chat_msg.message
        
        # Start chat with history
        chat = model.start_chat(history=conversations[conversation_id])
        
        # Get response from Gemini
        response = chat.send_message(prompt)
        
        # Update conversation history (keep last 10 exchanges to stay within limits)
        conversations[conversation_id] = chat.history[-20:]  # 10 exchanges = 20 messages
        
        return ChatResponse(
            response=response.text,
            conversation_id=conversation_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/clear")
async def clear_conversation(conversation_id: str = "default"):
    """Clear conversation history"""
    if conversation_id in conversations:
        del conversations[conversation_id]
    return {"message": "Conversation cleared"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "gemini-1.5-flash"}

@app.get("/")
async def root():
    return {"message": "Gemini Chatbot API is running", "endpoints": ["/chat", "/clear", "/health"]}

# For local testing -
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
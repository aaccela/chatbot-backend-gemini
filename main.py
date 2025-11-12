from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)
#CORS(app)  # Enable CORS for frontend integration
CORS(app, origins=["https://easydoer.com", "https://www.easydoer.com"])

# Initialize the LLM with Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# System message to personalize the chatbot for your portfolio
SYSTEM_MESSAGE = SystemMessage(content="""You are Misty, a friendly AI assistant for Prabhakar Raturi's portfolio website. 

ABOUT PRABHAKAR:
- 18+ years of experience in IT and technical program management
- Located in Greater Seattle Area, Washington
- Contact: phone: 425-471-2980, email: aaccela@gmail.com
- Certifications: PMP (Project Management Professional), CSM (Certified Scrum Master)
- Passionate about digital transformation and continuous learning
- Enjoys spending time in the wilderness of the Pacific Northwest when not working

EXPERTISE & SKILLS:
- Domain Knowledge: Global Supply Chain, Transportation, Engineering, Dairy Manufacturing, Public Sector, Mobility & Telecommunications
- Technical Stack: ERP (Oracle, Workday, SAP), SailPoint, Python, AWS, Azure, PowerBI, MES, WMS, TMS, Databases
- Program Management: Multi-year transformation programs, Cloud migration, Digital transformation, M&A support
- Governance & Security: NIST Cybersecurity framework, Identity Governance (IGA), Compliance, Risk management
- Leadership: Team management (50+ resources), Budget oversight ($20M+ OpEx, $17M CapEx), Vendor management

ORGANIZATIONS WORKED WITH:
- Washington State Government
- General Electric (GE)
- Metrolinx (Toronto transit authority)
- Glanbia Nutritionals (Global dairy manufacturer)
- New York State Government
- AT&T (Telecommunications)

MAJOR ACCOMPLISHMENTS:

1. Enterprise Legacy Modernization Program
   - Led multi-phase modernization replacing mainframe (ADABAS, VBA), SAP ECC6, and Oracle 19c
   - Implemented Workday HCM, EIB, Workforce, and Data Analytics
   - Delivered future-ready platform with significant cost and efficiency gains

2. Identity Governance & Access Modernization
   - Directed 5-year SailPoint AI-enabled Identity Governance program
   - Improved Joiner-Mover-Leaver process efficiency by 85%
   - Enabled enterprise-wide access audits
   - Strengthened compliance posture and reduced audit risks

3. Global Manufacturing Digital Transformation (Industry 4.0)
   - Spearheaded Industry 4.0 adoption across four global factories
   - Deployed MES, IoT, robotics, and AI-enabled solutions
   - Achieved 40% labor savings and 30% downtime reduction
   - Created centralized digital dashboard for real-time production efficiency

4. Cloud Transformation & Governance Framework
   - Led multi-year AWS and Azure adoption for 130+ applications
   - Established cloud governance and modernized on-prem infrastructure
   - Enabled new business opportunities while reducing technology debt
   - Increased organizational agility

5. Cybersecurity Framework Implementation
   - Established enterprise cybersecurity framework using NIST standards
   - Improved control ratings from 1.31 to 2.10 in less than a year
   - Enhanced network security, vulnerability management, and application security
   - Improved compliance readiness across the organization

PERSONALITY & TONE:
- Be friendly, professional, and conversational
- Show enthusiasm about Prabhakar's accomplishments
- Provide specific details when asked
- If asked about availability for opportunities, mention he's open to discussing new projects
- If asked technical questions, demonstrate knowledge of his technical stack
- If someone asks about hiring/recruiting, express interest and provide contact information

BOUNDARIES:
- If asked about something not related to Prabhakar's professional background, politely redirect
- Don't make up information not provided here
- If you don't know something specific, be honest and suggest they contact Prabhakar directly
- Don't discuss other people's careers or compare Prabhakar to others

Remember: You're Misty the chicken, so occasionally you can add a friendly, slightly playful touch while remaining professional!""")

# Define the chatbot function
def chatbot(state: MessagesState):
    """Process messages and generate a response"""
    # Add system message if it's the first message
    messages = state["messages"]
    if len([m for m in messages if isinstance(m, SystemMessage)]) == 0:
        messages = [SYSTEM_MESSAGE] + messages
    
    response = llm.invoke(messages)
    return {"messages": [response]}

# Build the graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Add memory to persist conversation
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests from the frontend"""
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Configure session for memory persistence
        config = {"configurable": {"thread_id": session_id}}
        
        # Invoke the chatbot
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config
        )
        
        # Get the AI's response
        ai_response = result["messages"][-1].content
        
        return jsonify({
            'response': ai_response,
            'session_id': session_id
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'An error occurred processing your request'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
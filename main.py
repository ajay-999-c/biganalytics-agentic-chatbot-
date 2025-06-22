from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
import uuid
from typing import Optional

from agent import get_rag_agent, pre_router_chain
# Import the function that creates our real agent
from agent import get_rag_agent, get_llm
# Import enhanced RAG for auto-initialization
from enhanced_rag_retriever import FreeAdvancedRAG
import config


# Load environment variables from .env file
load_dotenv() 

def auto_initialize_enhanced_vectors():
    """
    Automatically initialize enhanced ChromaDB vectors when FastAPI starts.
    Creates vector store if it doesn't exist.
    """
    try:
        print("🚀 FastAPI Starting - Initializing Enhanced RAG System...")
        
        # Initialize enhanced RAG (will use existing ChromaDB if available)
        rag = FreeAdvancedRAG()
        
        # Check if we have data in the collection
        try:
            collection_count = rag.get_collection_stats()
            
            # If collection is empty, build vector store
            if collection_count == 0:
                print("🔄 Collection is empty, building vector store...")
                success = rag.build_vector_store(config.KNOWLEDGE_BASE_FILE)
                if success:
                    print("✅ Enhanced vector store created successfully!")
                    rag.get_collection_stats()
                else:
                    print("❌ Failed to build vector store")
                    return False
            else:
                print("✅ Existing ChromaDB collection found and ready!")
        except Exception as e:
            print(f"⚠️ Could not check collection status: {e}")
            print("🔄 Attempting to build vector store...")
            success = rag.build_vector_store(config.KNOWLEDGE_BASE_FILE)
            if not success:
                print("❌ Failed to build vector store")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing enhanced RAG system: {e}")
        return False

# Automatically initialize enhanced vectors when FastAPI starts
print("🚀 FastAPI Starting - Initializing Enhanced ChromaDB System...")
embedding_success = auto_initialize_enhanced_vectors()
if embedding_success:
    print("✅ Enhanced RAG System ready! FastAPI can now start.")
else:
    print("⚠️ Warning: Embeddings initialization failed, but continuing...")

# --- SERVER-SIDE MEMORY & PRE-ROUTER SETUP ---
conversation_memories = {}
class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None # We now take just an ID from client instead of full history


app = FastAPI(
    title="Bignalytics RAG Agent API",
    description="Endpoint for interacting with the Bignalytics RAG agent.",
    version="1.0.0"
)


# Initialize the real agent
agent = get_rag_agent()

@app.post("/chat")
def chat_with_agent(request: ChatRequest):
    """
    Handles greetings, manages server-side memory, and invokes the RAG agent.
    """
    # 1. Handle conversation ID
    convo_id = request.conversation_id
    if not convo_id:
        # If client didn't send ID, create a new unique ID
        convo_id = str(uuid.uuid4())
    
    # Assign convo_id back to request if it was newly generated
    if not request.conversation_id:
        request.conversation_id = convo_id

    # 1. First classify the intent
    intent = pre_router_chain.invoke({"question": request.question})

    # 2. Handle greetings and respond quickly
    if "greeting" in intent.lower():
        greeting_response = """Hello! How can I assist you with information about Bignalytics?

Here are our contact details:

Address: Pearl Business Park, 3, Bhawarkua Main Rd, Above Ramesh Dosa, Near Vishnupuri i bus stop, Vishnu Puri Colony, Indore, Madhya Pradesh - 452001 
Contact Number: 093992-00960"""
        return {"answer": greeting_response, "conversation_id": request.conversation_id}

    # 3. Get or create memory for each conversation
    if request.conversation_id not in conversation_memories:
        conversation_memories[request.conversation_id] = ConversationSummaryBufferMemory(
            llm=get_llm(), max_token_limit=1000, return_messages=True
        )           

    memory = conversation_memories[request.conversation_id]

    # 4. Call the main agent
    history_langchain_messages = memory.load_memory_variables({})['history']
    input_data = {
        "user_question": request.question,
        "conversation_history": history_langchain_messages
    }
    response = agent.invoke(input_data)
    final_answer = response.get("final_response")

    # 5. Save the new Q&A to memory
    memory.save_context({"input": request.question}, {"output": final_answer})

    return {"answer": final_answer, "conversation_id": request.conversation_id}

@app.get("/")
def read_root():
    return {"status": "Bignalytics RAG Agent is running."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
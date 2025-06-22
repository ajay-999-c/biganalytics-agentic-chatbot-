import os
from typing import List, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from operator import itemgetter
from langchain_core.prompts import MessagesPlaceholder

# Imports for tool calling functionality
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import create_tool_calling_agent, AgentExecutor, create_react_agent
# caching tools
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
import config
# Import for enhanced RAG retriever system
from enhanced_rag_retriever import FreeAdvancedRAG
# Import prompts from separate file
from prompts_new import (
    get_pre_router_template,
    get_rewriter_template, 
    get_synthesis_template,
    get_react_agent_template,
    get_tool_calling_agent_template
)


# --- LANGSMITH TRACING SETUP ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
langsmith_project = os.getenv('LANGSMITH_PROJECT')
if langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
if langsmith_project:
    os.environ['LANGSMITH_PROJECT'] = langsmith_project

# --- GLOBAL CACHE SETUP ---
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# --- 1. MODULAR COMPONENT LOADERS (No changes) ---

def get_primary_llm():
    """Primary LLM for critical tasks: Main Agent + Rewriter (Uses Gemini)"""
    if config.USE_OPENSOURCE_LLM:
        return Ollama(model="llama3")
    else:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", temperature=0)

def get_secondary_llm():
    """Secondary LLM for less critical tasks: Pre-router + Synthesizer"""
    if config.USE_OPENSOURCE_LLM or not hasattr(config, 'USE_HYBRID_LLM') or not config.USE_HYBRID_LLM:
        # Use same LLM as primary if not in hybrid mode
        return get_primary_llm()
    
    # Use Cloudflare for cost optimization
    try:
        from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI
        # Use a fast, high-performance, and accurate model from Cloudflare's offerings
        # Prefer Llama 3 8B if available, else fallback to Mistral 7B
        # Both are fast, high-performance, and accurate for most tasks
        try:
            # Try Llama 3 8B first
            return CloudflareWorkersAI(
            account_id=config.CLOUDFLARE_ACCOUNT_ID,
            api_token=config.CLOUDFLARE_API_TOKEN,
            model="@cf/meta/llama-3-8b-instruct"
            )
        except Exception as e:
            print("Llama 3 8B not available or failed, falling back to Mistral 7B:", e)
            return CloudflareWorkersAI(
            account_id=config.CLOUDFLARE_ACCOUNT_ID,
            api_token=config.CLOUDFLARE_API_TOKEN,
            model="@cf/mistral/mistral-7b-instruct-v0.2"
            )
    except:
        return get_primary_llm()  # Fallback to primary

def get_llm():
    """Backward compatibility - returns primary LLM"""
    return get_primary_llm()

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
    )

def get_retriever():
    """
    Enhanced retriever using ChromaDB + reranker system.
    Returns the enhanced RAG object for better document retrieval.
    """
    try:
        print("🚀 Initializing Enhanced RAG Retriever...")
        rag = FreeAdvancedRAG()
        print("✅ Enhanced RAG retriever initialized successfully!")
        return rag
    except Exception as e:
        print(f"❌ Error initializing enhanced RAG retriever: {e}")
        print("CRITICAL: Enhanced retriever failed to initialize. The RAG agent may not function correctly.")
        return None


# PRE-ROUTER: Uses secondary LLM (Cloudflare) - Binary Classification
pre_router_chain = get_pre_router_template() | get_secondary_llm() | StrOutputParser()

# --- 2. LANGGRAPH STATE DEFINITION (No changes) ---

class GraphState(TypedDict):
    user_question: str
    conversation_history: list
    rewritten_questions: List[str]
    individual_answers: dict
    final_response: str

# --- 3. OPTIMIZED ARCHITECTURE ---

# Initialize components
primary_llm = get_primary_llm()
secondary_llm = get_secondary_llm()
retriever = get_retriever()

# STEP 1: Create a tool that will search the Knowledge Base
@tool
def bignalytics_knowledge_search(query: str) -> str:
    """
    Comprehensive search tool for Bignalytics Institute knowledge base.
    
    This tool retrieves precise information including FAQ sections, quick reference data, and structured content about:
    - Course offerings (Data Analytics, Data Science, AI/ML) with exact durations and fees
    - Curriculum details, technologies, and tools taught
    - Class schedules, timings, and lab session information
    - Placement statistics and career outcomes
    - Institute information, contact details, and location
    - Faculty expertise and learning methodologies
    
    IMPORTANT: This tool returns information in FAQ format (Q: question, A: answer) and structured sections.
    When the retrieved content contains a direct Q&A match for the user's question, use that exact answer.
    For example, if you see "Q: What is the duration of the AI & ML course?" followed by 
    "A: The Master Program in Artificial Intelligence & Machine Learning has a duration of 12-14 months",
    use that precise answer directly.
    
    Returns comprehensive, contextually relevant information that should be directly extracted and used.
    """
    if retriever is None:
        return "Knowledge base is not accessible. Please contact the institute directly."
    
    try:
        # Enhanced query processing for better factual retrieval
        enhanced_query = query
        
        # For duration queries, add specific keywords to boost keyword search
        duration_keywords = ["duration", "months", "time", "long", "period", "length"]
        if any(keyword in query.lower() for keyword in duration_keywords):
            enhanced_query = f"{query} duration months time period"
        
        # For course-specific queries, add course identifiers
        course_keywords = ["ai", "ml", "data science", "data analytics", "artificial intelligence", "machine learning"]
        if any(keyword in query.lower() for keyword in course_keywords):
            enhanced_query = f"{query} course program training"
        
        # Retrieve relevant documents from the knowledge base using enhanced RAG
        results = retriever.advanced_search(enhanced_query, use_reranking=True, top_k=3)
        
        if not results:
            return "No relevant information found in the knowledge base for your query."
        
        # Process and structure the retrieved information
        knowledge_segments = []
        for i, result in enumerate(results, 1):
            content = result.get('content', '').strip()
            if content:
                knowledge_segments.append(f"📚 Knowledge Section {i}:\n{content}")
        
        # Combine all relevant information
        comprehensive_info = "\n\n" + "="*60 + "\n\n".join(knowledge_segments)
        
        return f"📋 Retrieved Information:\n{comprehensive_info}"
        
    except Exception as e:
        return f"Error accessing knowledge base: {str(e)}. Please try rephrasing your question or contact the institute directly."

# REWRITER: Uses primary LLM (Gemini) for accuracy - Advanced Chain-of-Thought
rewriter_chain = get_rewriter_template() | primary_llm | JsonOutputParser()

# SYNTHESIZER: Uses secondary LLM (Cloudflare) - Structured Response Generation  
synthesis_chain = get_synthesis_template() | secondary_llm | StrOutputParser()


# MAIN AGENT: Uses primary LLM (Gemini) for tool-calling
tools = [bignalytics_knowledge_search]

if config.USE_OPENSOURCE_LLM:
    # Advanced ReAct agent for open-source LLMs with Chain-of-Thought
    from langchain.agents import create_react_agent, AgentExecutor
    agent_prompt = get_react_agent_template()
    tool_agent = create_react_agent(primary_llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=tool_agent, tools=tools, verbose=True)
else:
    # Tool-calling agent for Gemini - Advanced Role-Based Prompting
    agent_prompt = get_tool_calling_agent_template()
    tool_agent = create_tool_calling_agent(primary_llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=tool_agent, tools=tools, verbose=True) 

# Graph nodes (simplified)
def rewrite_query_node(state: GraphState):
    # Handle conversation history properly - extract content from Message objects
    history_text = "No previous context"
    if state["conversation_history"]:
        history_parts = []
        for msg in state["conversation_history"]:
            if hasattr(msg, 'content'):
                # LangChain Message objects have a content attribute
                msg_type = "Human" if "Human" in str(type(msg)) else "AI"
                history_parts.append(f"{msg_type}: {msg.content}")
            else:
                # If it's already a string
                history_parts.append(str(msg))
        history_text = "\n".join(history_parts)
    
    result = rewriter_chain.invoke({
        "history": history_text,
        "question": state["user_question"]
    })
    return {"rewritten_questions": result.get('questions', [state["user_question"]])}

def process_questions_node(state: GraphState):
    answers = {}
    for question in state["rewritten_questions"]:
        response = agent_executor.invoke({"input": question})
        answers[question] = response.get('output', "No response")
    return {"individual_answers": answers}

def decide_to_synthesize_node(state: GraphState):
    if len(state["individual_answers"]) == 1:
        return {"final_response": list(state["individual_answers"].values())[0]}
    return {}

def synthesize_response_node(state: GraphState):
    answers_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in state["individual_answers"].items()])
    try:
        final_response = synthesis_chain.invoke({"answers": answers_str})
        
        # Ensure response doesn't get truncated - limit to reasonable length
        if len(final_response) > 2000:
            # If too long, truncate but add proper ending
            final_response = final_response[:1950] + "... \n\nFor more details, please ask specific questions about what interests you most!"
        
        return {"final_response": final_response}
    except:
        # If synthesis fails, return concatenated answers
        return {"final_response": answers_str}

# Build and compile the RAG agent graph
def get_rag_agent():
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("rewriter", rewrite_query_node)
    graph.add_node("processor", process_questions_node)
    graph.add_node("decider", decide_to_synthesize_node)
    graph.add_node("synthesizer", synthesize_response_node)
    
    # Add edges
    graph.set_entry_point("rewriter")
    graph.add_edge("rewriter", "processor")
    graph.add_edge("processor", "decider")
    graph.add_conditional_edges(
        "decider",
        lambda state: "synthesizer" if len(state["individual_answers"]) > 1 else END,
        {"synthesizer": "synthesizer", END: END}
    )
    graph.add_edge("synthesizer", END)
    
    return graph.compile()
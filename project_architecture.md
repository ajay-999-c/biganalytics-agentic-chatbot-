BIGNALYTICS AGENTIC RAG CHATBOT - FINAL ARCHITECTURE (Hybrid LLM)

+-------------------------------------------------------------------------+
|                                  USER                                   |
| (Sends: "question", "conversation_id" [optional])                       |
+-------------------------------------------------------------------------+
                    |
                    v
+-------------------------------------------------------------------------+
|                  FastAPI Server (`main.py`)                             |
|      - Manages Conversation ID (creates if new)                         |
|      - Handles Server-Side Memory (ConversationSummaryBufferMemory)     |
+-------------------------------------------------------------------------+
                    |
                    v
+-------------------------------------------------------------------------+
|                1. PRE-ROUTER NODE (Simple Keyword Check)                |
|   - Input: "question"                                                   |
|   - Task: Checks if query is a simple greeting (e.g., 'hi', 'hello').   |
|   - Use llm call to intentify greeting.                       |
+-------------------------------------------------------------------------+
       |                                      |
       '--> IF 'greeting'                      '--> IF 'information_seeking'
           |                                      |
+--------------------------+                      v
|   Return Hardcoded     |      +-------------------------------------------+
|   Greeting Response    |      |         MAIN LANGGRAPH AGENT INVOKED        |
+--------------------------+      |    (Receives: "user_question",            |
           |                      |     "conversation_history" from memory)   |
           '--------------------->|-------------------------------------------|
                                  |         GRAPH EXECUTION STARTS...         |
                                  |                                           |
                                  |   +-----------------------------------+   |
                                  |   |   2. REWRITER NODE                |   |
                                  |   |   - Task: Breaks complex queries. |   |
                                  |   |   - Uses LLM (Cloudflare)         |   |
                                  |   +-----------------------------------+   |
                                  |                   |                       |
                                  |                   v                       |
                                  |   +-----------------------------------+   |
                                  |   |   3. PROCESSOR NODE               |   |
                                  |   |   - Loops through each question & |   |
                                  |   |     invokes Tool-Calling Agent.   |   |
                                  |   |                                   |   |
                                  |   | --- Inside AgentExecutor Logic ---|   |
                                  |   | | LLM (Gemini) analyzes query:  |   |
                                  |   | +-------------------------------+   |
                                  |   |   |                               |   |
                                  |   |   '-> IF Bignalytics query?      |   |
                                  |   |        |                          |   |
                                  |   |        v                          |   |
                                  |   |      [Invoke RAG Tool]            |   |
                                  |   |   |                               |   |
                                  |   |   '-> IF General CS/IT query?    |   |
                                  |   |        |                          |   |
                                  |   |        v                          |   |
                                  |   |      [Answer from own knowledge]  |   |
                                  |   |   |                               |   |
                                  |   |   '-> IF Out-of-Scope query?     |   |
                                  |   |        |                          |   |
                                  |   |        v                          |   |
                                  |   |      [Generate Fallback Message]  |   |
                                  |   |-----------------------------------|   |
                                  |   +-----------------------------------+   |
                                  |                   |                       |
                                  |                   v                       |
                                  |   +-----------------------------------+   |
                                  |   |   4. DECIDER NODE                 |   |
                                  |   |   - Checks number of answers.     |   |
                                  |   +-----------------------------------+   |
                                  |       |                             |       |
                                  |       '--> IF 1 Answer             '--> IF >1 Answers
                                  |           |                             |
                                  '-----------'                             v
                                                                  +---------------------+
                                                                  | 5. SYNTHESIZER NODE |
                                                                  | - Combines multiple |
                                                                  |   answers into one. |
                                                                  | - Uses LLM (Cloudflare) |
                                                                  +---------------------+
                                                                          |
                                                                          v
+-------------------------------------------------------------------------+
|                  FINAL RESPONSE                                         |
|  (Final answer and conversation_id sent back to User)                   |
+-------------------------------------------------------------------------+


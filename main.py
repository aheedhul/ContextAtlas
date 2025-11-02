import os
import logging
import chromadb
import json

# Core LlamaIndex imports
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core import Settings, VectorStoreIndex, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever

# LLM and Embedding imports
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Reranker import

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Configuration (override via environment variables when needed)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "contextatlas_collection")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
KNOWLEDGE_SOURCE_NAME = os.getenv("KNOWLEDGE_SOURCE_NAME", "ContextAtlas Corpus")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Globals ---
app = FastAPI()
index = None
reranker = None 
"""response_synthesizer is not global; we prompt the LLM for JSON directly."""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/response models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]

# Strict JSON prompt used for answer synthesis
STRICT_JSON_PROMPT = """
You are an expert RAG (Retrieval-Augmented Generation) processor.
Your task is to analyze a user's query and a list of "Candidate Contexts" from "{knowledge_source}" and generate a perfect, high-faithfulness answer in a single, valid JSON object, and nothing else.

You MUST follow this internal step-by-step process:
1.  *Analyze Query:* Deeply understand the user's question.
2.  *Analyze Contexts:* Read all "Candidate_Contexts" provided below.
3.  *Re-Rank and Select:* Identify the most relevant and useful contexts needed to answer the query. You must select exactly {top_k_requested} contexts, unless fewer are retrieved or none are relevant.
4.  *Generate Answer:* Formulate a concise and correct "answer" using only the information from the contexts you selected.
5.  *Extract Citations:* Note the page numbers from only the contexts you used.

### CRITICAL RULES (FOR HIGH RAGAS SCORE):
1.  **STRICT FAITHFULNESS (15%):** The "answer" field MUST be generated using ONLY the information from the final selected contexts. Do NOT use any external knowledge.
2.  **THE "I DON'T KNOW" RULE:** If NONE of the contexts are relevant or sufficient to answer the query, you MUST set the "answer" field to "I do not have enough information to answer this question." and the "contexts" field to an empty list [].
3.  **SAFETY:** If the query requests personal medical, legal, or financial advice, you MUST follow Rule #2.
4.  **CITATION BLOCK:** If an answer is generated, the final "answer" MUST end with a citation block using only the page numbers: "\\n(See Pages: [page_num_1], [page_num_2])".
5.  **OUTPUT FORMAT:** Your entire response MUST be a single, valid JSON object, and nothing else.

### Inputs:
---
"User_Query": "{user_query}"
---
"Output_Top_K": {top_k_requested} 
---
"Knowledge_Source": "{knowledge_source}"
---
"Candidate_Contexts": [
{contexts_list}
]
---

### Required JSON Output Format:
(Your response MUST be this JSON object)

{{
    "answer": "Generate the concise, correct answer here, ending with the citation block. (e.g., Anemia is referenced in the text. \\n(See Pages: 158, 203))",
    "contexts": [
        // Copy the full string of *only* the relevant context(s) you selected here.
    ]
}}
"""
# --- END FINAL SYSTEM PROMPT ---


# --- Initialization ---
def initialize_rag_pipeline():
    global index, reranker
    logger.info("Initializing RAG pipeline components...")
    check_collection = None
    
    # DB check & setup
    try:
        logger.info(f"[DB Check] Attempting to connect to ChromaDB at: {DB_PATH}")
        if not os.path.exists(DB_PATH):
             raise FileNotFoundError(f"Database path not found: {DB_PATH}")
        check_client = chromadb.PersistentClient(path=DB_PATH)
        check_collection = check_client.get_collection(COLLECTION_NAME)
        collection_count = check_collection.count()
        logger.info(f"[DB Check] SUCCESS: Collection '{COLLECTION_NAME}' found with {collection_count} items.")
        if collection_count == 0:
             logger.warning("[DB Check] Collection is empty! No data to retrieve.")
    except Exception as db_check_error:
        logger.error(f"[DB Check] FAILED: Could not connect or read ChromaDB collection. Error: {db_check_error}", exc_info=True)
        raise RuntimeError(f"ChromaDB check failed: {db_check_error}") from db_check_error

    try:
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}...")
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

        google_api_key = GOOGLE_API_KEY
        if not google_api_key:
             raise ValueError("GOOGLE_API_KEY not set in the environment.")
        logger.info(f"Initializing LLM with GoogleGenAI: {GEMINI_MODEL_NAME}...")
        Settings.llm = GoogleGenAI(model_name=GEMINI_MODEL_NAME, api_key=google_api_key)

        logger.info(f"Connecting to ChromaDB via LlamaIndex wrapper...")
        vector_store = ChromaVectorStore(chroma_collection=check_collection)

        logger.info("Loading index from vector store...")
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        logger.info("Initializing FlagEmbedding Reranker...")
        reranker = FlagEmbeddingReranker(
            top_n=3, # This is the internal scoring limit, overridden by effective_top_k_return
            model="BAAI/bge-reranker-base"
        )
        
        logger.info("RAG components initialization complete.")
    except Exception as e:
        logger.error(f"Error during RAG component initialization: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize RAG components: {e}") from e

# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    # Deprecation warning handled by the warning log above
    initialize_rag_pipeline()

# --- API Endpoint ---
@app.post("/query", response_model=QueryResponse)
async def query_route(req: QueryRequest):
    global index, reranker
    
    if index is None or reranker is None:
        logger.error("RAG components not initialized.")
        raise HTTPException(status_code=503, detail="Service Unavailable: RAG components not ready.")

    try:
        logger.info(f"Received query: '{req.query}', requested top_k: {req.top_k}")
        
        effective_top_k_retrieve = max(5, (req.top_k or 3) + 2) # Retrieve 5-7 nodes
        effective_top_k_return = max(1, req.top_k or 3)       # Return 3-5 nodes

    # Retrieve initial chunks (slightly more for better reranking)
        retriever = VectorIndexRetriever(index=index, similarity_top_k=effective_top_k_retrieve)
        query_bundle = QueryBundle(req.query)
        retrieved_nodes = await retriever.aretrieve(query_bundle)

        # 1.5. Rerank and Filter
        reranker.top_n = effective_top_k_return 
        reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle=query_bundle)
        
       # Prepare input for JSON prompt
        context_list_for_prompt = []
        for i, node in enumerate(reranked_nodes):
           # Include page label and index for LLM citation
             page_label = node.metadata.get("page_label", "N/A")
             context_content = node.get_content().replace('"', '\\"') # Escape quotes
             
           # Format: "[Context X] [Page Y] text..."
             context_list_for_prompt.append(
                 f'"[Context {i}] [Page {page_label}] {context_content}"'
             )
        
        # Fill the JSON prompt template
        final_prompt = STRICT_JSON_PROMPT.format(
            user_query=req.query,
            top_k_requested=effective_top_k_return,
            knowledge_source=KNOWLEDGE_SOURCE_NAME,
            contexts_list=",\n".join(context_list_for_prompt) # Join into the list format
        )
        
        # Send the prompt to the LLM (expects raw JSON)
        logger.info("Synthesizing response via strict JSON prompt...")
        # Use the Settings.llm global object directly
        raw_response = await Settings.llm.acomplete(final_prompt)
        raw_answer_text = str(raw_response)

        # Extract and validate JSON
        try:
            # Clean LLM output (remove common markdown wrappers)
            json_str = raw_answer_text.strip()
            # Clean off markdown wrapper if present (e.g., ```json ... ```)
            if json_str.startswith("```"):
                 json_str = json_str.replace("```json", "").replace("```", "").strip()
            
            response_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Failed: {e}. Raw output: {raw_answer_text}")
            # If JSON fails, return a safe, compliant failure message
            return QueryResponse(answer=f"LLM Error: Could not generate valid JSON. Please check LLM response for malformed characters.", contexts=[])


        # 5. Final Return (The evaluator will use this JSON)
        answer = response_json.get("answer", "Error: Answer field missing.")
        contexts = response_json.get("contexts", []) 
        
        logger.info(f"Generated answer length: {len(answer)}")
        logger.info(f"Returning {len(contexts)} contexts (from JSON output).")

        return QueryResponse(answer=answer, contexts=contexts)

    except Exception as e:
        logger.error(f"Error processing query '{req.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# --- Health Check ---
@app.get("/health")
async def health_check():
    if index and reranker:
        return {"status": "ok", "message": "RAG components initialized."}
    else:
        raise HTTPException(status_code=503, detail="Service Unavailable: RAG components not initialized.")

# --- Local runner ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server locally...")
    if not GOOGLE_API_KEY:
         print("\n\nERROR: GOOGLE_API_KEY environment variable is missing.\n")
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

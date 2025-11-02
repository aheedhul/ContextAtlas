import fitz  # PyMuPDF
import os
import time
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import chromadb
from bisect import bisect_right

print("Starting ContextAtlas hierarchical ingest...")

# Runtime configuration (override via environment variables)
print("Loading embedding model (BAAI/bge-small-en-v1.5)...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None
DATA_DIR = os.getenv("DATA_DIR", "data")
DB_DIR = os.getenv("CHROMA_DB_PATH", "chroma_db")
DB_COLLECTION = os.getenv("CHROMA_COLLECTION", "contextatlas_collection")

# --- Database Setup ---
print(f"Setting up ChromaDB at: {DB_DIR}")
db = chromadb.PersistentClient(path=DB_DIR)
chroma_collection = db.get_or_create_collection(DB_COLLECTION)

def get_pdf_toc(filepath):
    """Extract ToC (document outline) from a PDF as (page_num, title, level)."""
    print(f"Extracting ToC from: {os.path.basename(filepath)}")
    toc_data = []
    doc = fitz.open(filepath)
    # doc.get_toc() returns [level, title, page_num, ...]
    toc = doc.get_toc(simple=False)
    if not toc:
        print(f"WARNING: No ToC (Document Outline) found for {os.path.basename(filepath)}")
        return None
        
    for level, title, page_num, _ in toc:
        # We store (page_num, title, level) for easy sorting and lookup
        toc_data.append((int(page_num), title.strip(), int(level)))
    
    # Sort by page number
    toc_data.sort()
    return toc_data

def enrich_nodes_with_toc(nodes, toc_map):
    """Add hierarchical metadata to each node based on its page number."""
    print("Enriching nodes with ToC metadata...")
    if not toc_map:
        print("No ToC map available. Skipping enrichment.")
        return nodes

    # toc_map is { "filename": [(page, title, level), ...], ... }
    
    for node in nodes:
        filename = node.metadata.get("source_book")
        if not filename or filename not in toc_map:
            continue # Skip if node's source file has no ToC

        toc_data = toc_map[filename]
        page_num_str = node.metadata.get("page_label", "0")
        
        try:
            page_num = int(page_num_str)
        except ValueError:
            print(f"Warning: Could not parse page number '{page_num_str}'")
            continue

        # 'toc_data' is sorted by page number.
        toc_pages = [entry[0] for entry in toc_data]
        
        i = bisect_right(toc_pages, page_num)
        
        if i == 0:
            continue # Page is before the first ToC entry

        current_hierarchy = {
            "L1_Section": None,
            "L2_Chapter": None,
            "L3_Subchapter": None,
            "L4_Topic": None
        }
        
        _, current_title, current_level = toc_data[i - 1]
        
        current_hierarchy[f"L{current_level}_Topic"] = current_title
        
        parent_level = current_level - 1
        for j in range(i - 2, -1, -1):
            _, parent_title, parent_lvl = toc_data[j]
            if parent_lvl == parent_level:
                key = f"L{parent_level}_Topic"
                if parent_level > 3: key = "L4_Topic"
                if parent_level == 3: key = "L3_Subchapter"
                if parent_level == 2: key = "L2_Chapter"
                if parent_level == 1: key = "L1_Section"
                
                # Only add if it's not already filled
                if not current_hierarchy.get(key):
                    current_hierarchy[key] = parent_title
                    parent_level -= 1
            
            if parent_level == 0:
                break # We've reached the top
                
    # Add the clean, non-null metadata to the node
        clean_hierarchy = {k: v for k, v in current_hierarchy.items() if v}
        node.metadata.update(clean_hierarchy)

    print("Enrichment complete.")
    return nodes

# --- Main Execution ---
def main():
    start_time = time.time()
    
    if chroma_collection.count() > 0:
        print(f"Database at '{DB_DIR}' already contains data.")
        print(f"Delete the '{DB_DIR}' folder and re-run to re-process.")
        return

    toc_map = {}
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            toc = get_pdf_toc(filepath)
            if toc:
                toc_map[filename] = toc

    def file_metadata_func(file_path: str) -> dict:
        return {"source_book": os.path.basename(file_path)}

    reader = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        required_exts=[".pdf"],
        file_metadata=file_metadata_func,
    )
    
    print("Loading documents from PDF files...")
    documents = reader.load_data(show_progress=True)
    
    if not documents:
        print(f"Error: No documents found in {DATA_DIR}.")
        return

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    print("Initializing semantic splitter...")
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=Settings.embed_model
    )
    
    pipeline = IngestionPipeline(
        transformations=[splitter, Settings.embed_model],
    )

    print("Running chunking pipeline (in memory)...")
    nodes = pipeline.run(documents=documents, show_progress=True)

    nodes = enrich_nodes_with_toc(nodes, toc_map)

    print("Sanitizing metadata for ChromaDB compatibility...")
    sanitized_count = 0
    for node in nodes:
        for key, value in node.metadata.items():
            if isinstance(value, str):
                sanitized_value = value.encode('utf-8', 'replace').decode('utf-8')
                if sanitized_value != value:
                    sanitized_count += 1
                node.metadata[key] = sanitized_value
    if sanitized_count > 0:
        print(f"Sanitized {sanitized_count} potentially problematic metadata string(s).")
    else:
        print("Metadata seems clean.")
    
    print("Adding enriched nodes to vector database...")
    vector_store.add(nodes, show_progress=True)
    
    end_time = time.time()
    print(f"--- Processing Complete ---")
    print(f"Total *semantically chunked* and *hierarchically tagged* nodes indexed: {chroma_collection.count()}")
    print(f"Database saved to: {DB_DIR}")
    print(f"Total time taken: {(end_time - start_time) / 60:.2f} minutes")
    
    print("\n--- Example Metadata Check ---")
    example = chroma_collection.peek(limit=1)
    if example and example.get('metadatas'):
        print(f"Example chunk metadata: {example['metadatas'][0]}")
    else:
        print("Could not retrieve example metadata. Check DB.")

if __name__ == "__main__":
    main()
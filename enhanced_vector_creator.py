#!/usr/bin/env python3
"""
Enhanced Vector Creation Tool
Uses the new enhanced RAG system for better vector store creation
"""

from enhanced_rag_retriever import FreeAdvancedRAG
import config

def create_enhanced_vectors():
    """Create vectors using the enhanced RAG system"""
    print("🚀 Creating Enhanced Vector Store...")
    print("Using ChromaDB + all-mpnet-base-v2 embeddings")
    
    try:
        # Initialize enhanced RAG
        rag = FreeAdvancedRAG()
        
        # Build vector store from knowledge base
        success = rag.build_vector_store(config.KNOWLEDGE_BASE_FILE)
        
        if success:
            print("✅ Enhanced vector store created successfully!")
            print("📊 Collection stats:")
            rag.get_collection_stats()
        else:
            print("❌ Failed to create enhanced vector store")
            
        return success
        
    except Exception as e:
        print(f"❌ Error creating enhanced vectors: {e}")
        return False

def test_enhanced_retrieval():
    """Test the enhanced retrieval system"""
    print("\n🧪 Testing Enhanced Retrieval...")
    
    try:
        rag = FreeAdvancedRAG()
        
        # Test queries
        test_queries = [
            "What are the fees for data analytics course?",
            "What is the duration of data science course?",
            "What topics are covered in AI ML curriculum?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            results = rag.advanced_search(query, top_k=2)
            
            if results:
                print(f"✅ Found {len(results)} results")
                print(f"📝 Top result: {results[0]['content'][:100]}...")
            else:
                print("❌ No results found")
                
    except Exception as e:
        print(f"❌ Error testing retrieval: {e}")

if __name__ == "__main__":
    print("🔧 Enhanced Vector Creation & Testing Tool")
    print("=" * 50)
    
    # Create vectors
    success = create_enhanced_vectors()
    
    if success:
        # Test retrieval
        test_enhanced_retrieval()
        
        print(f"\n🎉 Enhanced retriever system is ready!")
        print("💡 Use this for better vector creation and retrieval")
    else:
        print(f"\n❌ Enhanced system setup failed")

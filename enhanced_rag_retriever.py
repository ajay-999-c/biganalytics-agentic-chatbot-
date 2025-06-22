#!/usr/bin/env python3
"""
🆓 Free Advanced RAG Pipeline - Zero Cost Solution
ChromaDB + Better Embeddings + Free Reranking + Smart Chunking
"""

import os
import re
import chromadb
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, CrossEncoder
import config

class FreeAdvancedRAG:
    def __init__(self):
        """Initialize the free advanced RAG pipeline"""
        print("🚀 Initializing Free Advanced RAG Pipeline...")
        
        # Better embedding model (free)
        self.embedding_model_name = "sentence-transformers/all-mpnet-base-v2"  # Better than current
        self.embedding_model = None
        
        # Free reranking model
        self.reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        self.reranker = None
        
        # ChromaDB client (free, persistent)
        self.chroma_client = None
        self.collection_name = "bignalytics_advanced"
        self.collection = None
        
        # Initialize components
        self._initialize_models()
        self._initialize_chromadb()
    
    def _initialize_models(self):
        """Initialize embedding and reranking models"""
        try:
            print(f"📥 Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            print(f"🎯 Loading reranking model: {self.reranker_model_name}")
            self.reranker = CrossEncoder(self.reranker_model_name)
            
            print("✅ Models loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB with persistence"""
        try:
            # Create persistent ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                print(f"📂 Using existing ChromaDB collection: {self.collection_name}")
            except:
                self.collection = self.chroma_client.create_collection(name=self.collection_name)
                print(f"🆕 Created new ChromaDB collection: {self.collection_name}")
            
        except Exception as e:
            print(f"❌ Error initializing ChromaDB: {e}")
            raise
    
    def _smart_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Advanced FAQ-aware chunking strategy"""
        chunks = []
        
        # Strategy 1: Extract FAQ sections
        faq_pattern = r'\*\*Q:(.*?)\*\*\s*A:(.*?)(?=\*\*Q:|$)'
        faq_matches = re.findall(faq_pattern, text, re.DOTALL)
        
        for i, (question, answer) in enumerate(faq_matches):
            q_clean = question.strip()
            a_clean = answer.strip()
            
            # Create FAQ chunk with metadata
            faq_chunk = {
                'content': f"**Q: {q_clean}**\nA: {a_clean}",
                'type': 'faq',
                'question': q_clean,
                'answer': a_clean,
                'chunk_id': f'faq_{i}'
            }
            chunks.append(faq_chunk)
        
        # Strategy 2: Section-based chunking for non-FAQ content
        # Remove FAQ content first
        text_without_faqs = re.sub(faq_pattern, '', text, flags=re.DOTALL)
        
        # Split by sections (headings)
        section_pattern = r'(#{1,3}[^#\n]+)'
        sections = re.split(section_pattern, text_without_faqs)
        
        current_section = ""
        section_title = ""
        
        for i, section in enumerate(sections):
            if section.strip():
                if section.startswith('#'):
                    # This is a heading
                    if current_section.strip():
                        # Save previous section
                        chunks.append({
                            'content': current_section.strip(),
                            'type': 'section',
                            'section_title': section_title,
                            'chunk_id': f'section_{len(chunks)}'
                        })
                    section_title = section.strip()
                    current_section = section + "\n"
                else:
                    # This is content
                    current_section += section
        
        # Add final section
        if current_section.strip():
            chunks.append({
                'content': current_section.strip(),
                'type': 'section',
                'section_title': section_title,
                'chunk_id': f'section_{len(chunks)}'
            })
        
        # Strategy 3: Further split large sections
        final_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Optimized size
            chunk_overlap=50,  # Less overlap for FAQ content
            separators=["\n\n", "\n", ". ", " "]
        )
        
        for chunk in chunks:
            if len(chunk['content']) > 800:  # Split large chunks
                sub_docs = text_splitter.split_text(chunk['content'])
                for j, sub_content in enumerate(sub_docs):
                    sub_chunk = chunk.copy()
                    sub_chunk['content'] = sub_content
                    sub_chunk['chunk_id'] = f"{chunk['chunk_id']}_sub_{j}"
                    final_chunks.append(sub_chunk)
            else:
                final_chunks.append(chunk)
        
        print(f"📝 Smart chunking created {len(final_chunks)} chunks ({len(faq_matches)} FAQs)")
        return final_chunks
    
    def build_vector_store(self, file_path: str):
        """Build ChromaDB vector store with smart chunking"""
        print(f"🔄 Building vector store from: {file_path}")
        
        try:
            # Load document
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Smart chunking
            chunks = self._smart_chunking(text)
            
            if not chunks:
                print("❌ No chunks created")
                return False
            
            # Generate embeddings
            print("🧠 Generating embeddings...")
            contents = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_model.encode(contents, convert_to_numpy=True)
            
            # Prepare data for ChromaDB
            ids = [chunk['chunk_id'] for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                metadata = {
                    'type': chunk['type'],
                    'source': file_path
                }
                
                if chunk['type'] == 'faq':
                    metadata['question'] = chunk['question'][:500]  # ChromaDB limit
                    metadata['answer'] = chunk['answer'][:500]
                elif chunk['type'] == 'section':
                    metadata['section_title'] = chunk.get('section_title', '')[:500]
                
                metadatas.append(metadata)
            
            # Clear existing collection
            try:
                self.collection.delete()
                print("🗑️ Cleared existing collection")
            except:
                pass
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Vector store built with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error building vector store: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_documents(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search documents using ChromaDB"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )
            
            # Format results
            documents = []
            for i in range(len(results['documents'][0])):
                doc = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'id': results['ids'][0][i]
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank documents using free cross-encoder model"""
        try:
            if len(documents) <= 1:
                return documents
            
            # Prepare query-document pairs
            pairs = [(query, doc['content']) for doc in documents]
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Add rerank scores to documents
            for doc, score in zip(documents, rerank_scores):
                doc['rerank_score'] = float(score)
            
            # Sort by rerank score
            reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
            
            return reranked[:top_k]
            
        except Exception as e:
            print(f"❌ Error reranking: {e}")
            return documents[:top_k]
    
    def advanced_search(self, query: str, use_reranking: bool = True, top_k: int = 5) -> List[Dict[str, Any]]:
        """Advanced search with reranking"""
        print(f"🔍 Searching for: '{query}'")
        
        # Initial search
        documents = self.search_documents(query, n_results=15)
        
        if not documents:
            print("❌ No documents found")
            return []
        
        print(f"📄 Found {len(documents)} initial results")
        
        # Rerank if requested
        if use_reranking:
            print("🎯 Reranking results...")
            documents = self.rerank_documents(query, documents, top_k=top_k)
            print(f"📊 Reranked to top {len(documents)} results")
        else:
            documents = documents[:top_k]
        
        return documents
    
    def get_collection_stats(self):
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            print(f"📊 Collection '{self.collection_name}' contains {count} documents")
            return count
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return 0

def test_free_rag():
    """Test the free advanced RAG system"""
    print("🧪 Testing Free Advanced RAG Pipeline")
    print("=" * 50)
    
    try:
        # Initialize
        rag = FreeAdvancedRAG()
        
        # Build vector store
        print("\n1. Building vector store...")
        success = rag.build_vector_store(config.KNOWLEDGE_BASE_FILE)
        
        if not success:
            print("❌ Failed to build vector store")
            return
        
        # Get stats
        rag.get_collection_stats()
        
        # Test queries
        test_queries = [
            "What are the main topics covered in Data Analytics curriculum?",
            "What are the fees for Data Analytics course?",
            "Who are the faculty members?",
            "What technologies are taught in Data Science course?",
            "What is the duration of AI ML course?",
            "Do you provide placement assistance?"
        ]
        
        print(f"\n2. Testing {len(test_queries)} queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: '{query}' ---")
            
            # Search with reranking
            results = rag.advanced_search(query, use_reranking=True, top_k=3)
            
            if results:
                print(f"✅ Found {len(results)} relevant results:")
                for j, doc in enumerate(results, 1):
                    similarity = doc.get('similarity', 0)
                    rerank_score = doc.get('rerank_score', 0)
                    doc_type = doc['metadata'].get('type', 'unknown')
                    
                    print(f"\n  {j}. Type: {doc_type} | Similarity: {similarity:.3f} | Rerank: {rerank_score:.3f}")
                    print(f"     Content: {doc['content'][:150]}...")
                    
                    if doc_type == 'faq':
                        question = doc['metadata'].get('question', '')
                        if question:
                            print(f"     FAQ Question: {question[:100]}...")
            else:
                print("❌ No results found")
        
        print(f"\n🎉 Testing complete!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_free_rag()

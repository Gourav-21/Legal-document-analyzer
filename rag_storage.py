# Fix for SQLite3 compatibility with ChromaDB on Streamlit Cloud
import sys
import os

# Apply SQLite3 fix before importing chromadb
try:
    # Try to import pysqlite3 first
    import pysqlite3
    # Replace sqlite3 with pysqlite3
    sys.modules['sqlite3'] = pysqlite3
    print("✅ Successfully replaced sqlite3 with pysqlite3")
except ImportError:
    print("⚠️ pysqlite3-binary not available, trying fallback sqlite_fix")
    try:
        import sqlite_fix
    except ImportError:
        print("⚠️ sqlite_fix not available")

import chromadb
from chromadb.config import Settings
import uuid
import json
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session
from database import get_db, Law, Judgement
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()

class RAGLegalStorage:
    """
    Retrieval-Augmented Generation storage for legal documents.
    Uses ChromaDB for vector storage and PostgreSQL for text and summary storage.
    Enhanced with AI-powered summarization using Google Gemini and recursive text chunking.
    """
    
    def __init__(self, persist_directory: str = "./legal_rag_db"):
        self.persist_directory = persist_directory
        
        # Initialize database connection
        DATABASE_URL = os.getenv("DATABASE_URL")
        if DATABASE_URL:
            engine = create_engine(DATABASE_URL)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            self.db_session = SessionLocal()
        else:
            raise Exception("DATABASE_URL not found in environment variables")
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize Google Gemini AI for summarization
        self.gemini_api_key = os.getenv("GOOGLE_CLOUD_VISION_API_KEY")
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.ai_model = genai.GenerativeModel('gemini-2.5-flash')
                self.ai_enabled = True
                print("✅ AI summarization enabled with Gemini API")
            except Exception as e:
                print(f"⚠️ Warning: Failed to initialize Gemini AI: {e}")
                self.ai_model = None
                self.ai_enabled = False
        else:
            print("⚠️ Warning: GEMINI_API_KEY not found. AI summarization disabled.")
            print("   To enable AI summaries, set GEMINI_API_KEY in your .env file")
            self.ai_model = None
            self.ai_enabled = False
        
        # Initialize ChromaDB client
        # self.client = chromadb.PersistentClient(
        #     path=persist_directory,
        #     settings=Settings(
        #         anonymized_telemetry=False,
        #         allow_reset=True
        #     )
        # )
        chromadb_host = os.getenv("CHROMADB_HOST", "127.0.0.1")
        chromadb_port = int(os.getenv("CHROMADB_PORT", "3333"))
        self.client = chromadb.HttpClient(host=chromadb_host, port=chromadb_port)

        
        # Initialize OpenAI client for embeddings
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                print("✅ OpenAI embeddings enabled")
            except Exception as e:
                print(f"⚠️ Warning: Failed to initialize OpenAI client: {e}")
                self.openai_client = None
        else:
            print("⚠️ Warning: OPENAI_API_KEY not found. Please set it in your .env file")
            self.openai_client = None
        
        # Create collections for law and judgement chunks
        try:
            self.laws_collection = self.client.get_collection("law")
        except:
            self.laws_collection = self.client.create_collection(
                name="law",
                metadata={"description": "Chunked legal laws and regulations"}
            )
            
        try:
            self.judgements_collection = self.client.get_collection("judgement")
        except:
            self.judgements_collection = self.client.create_collection(
                name="judgement",
                metadata={"description": "Chunked legal judgements and precedents"}
            )
    
    def __del__(self):
        """Close database session when object is destroyed"""
        if hasattr(self, 'db_session') and self.db_session:
            self.db_session.close()
    
    def generate_law_summary(self, law_text: str) -> str:
        """Generate a concise AI summary of a law using Gemini AI"""
        if not self.ai_enabled or not self.ai_model:
            return "AI summarization not available. Please configure GEMINI_API_KEY."
            
        try:
            prompt = f"""
            Analyze the following legal text and create a summary that helps an AI detect violations.
            The summary should clearly outline the conditions that lead to a violation.

            Guidelines:
            1.  **Focus on Violations**: What actions are prohibited? What obligations are mandated?
            2.  **Clarity for AI**: Frame the summary to make it easy to check if a situation violates the law.
            3.  **Concise**: 2-4 sentences.
            4.  **Language**: Hebrew.
            5.  **Direct Output**: Do not add any introductory text. Provide only the summary.

            Example:
            - **Original Law Concept**: The law specifies requirements for severance pay.
            - **Violation-Focused Summary**: A violation occurs if an employer fails to pay severance to an employee who was dismissed after working for at least one year, or who resigned under specific circumstances like worsening of conditions.

            Legal Text:
            {law_text}

            Violation-Focused Summary (in Hebrew):
            """
            
            response = self.ai_model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Error generating law summary: {e}")
            return "Summary generation failed due to technical error."

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI's text-embedding-3-small model"""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized. Please set OPENAI_API_KEY.")
            
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise e

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI"""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized. Please set OPENAI_API_KEY.")
            
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            raise e

    def add_law(self, law_text: str, metadata: Optional[Dict] = None) -> str:
        """Add a law to both database and vector database with AI-generated summary and chunking"""
        law_id = str(uuid.uuid4())
        
        try:
            # Generate AI summary
            summary = self.generate_law_summary(law_text)
            
            # Store law in database (simplified schema: id, full_text, summary)
            db_law = Law(
                id=law_id,
                full_text=law_text,
                summary=summary
            )
            self.db_session.add(db_law)
            self.db_session.commit()
            
            # Chunk the text for vector storage
            chunks = self.text_splitter.split_text(law_text)
            
            # Generate embeddings for chunks
            if chunks:
                embeddings = self.get_embeddings_batch(chunks)
                
                # Prepare chunk IDs and metadata
                chunk_ids = [f"{law_id}_chunk_{i}" for i in range(len(chunks))]
                chunk_metadatas = []
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        "type": "law_chunk",
                        "law_id": law_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "created_at": datetime.now().isoformat(),
                        "summary": summary
                    }
                    if metadata:
                        chunk_metadata.update(metadata)
                    chunk_metadatas.append(chunk_metadata)
                
                # Add chunks to vector database
                self.laws_collection.add(
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=chunk_metadatas,
                    ids=chunk_ids
                )
            
            print(f"✅ Added law {law_id} with {len(chunks)} chunks to database and vector store")
            return law_id
            
        except Exception as e:
            # Rollback database changes if vector storage fails
            self.db_session.rollback()
            print(f"Error adding law: {e}")
            raise e
    
    def add_judgement(self, judgement_text: str, metadata: Optional[Dict] = None) -> str:
        """Add a judgement to both database and vector database with chunking"""
        judgement_id = str(uuid.uuid4())
        
        try:
            # Store judgement in database (simplified schema: id, full_text)
            db_judgement = Judgement(
                id=judgement_id,
                full_text=judgement_text
            )
            self.db_session.add(db_judgement)
            self.db_session.commit()
            
            # Chunk the text for vector storage
            chunks = self.text_splitter.split_text(judgement_text)
            
            # Generate embeddings for chunks
            if chunks:
                embeddings = self.get_embeddings_batch(chunks)
                
                # Prepare chunk IDs and metadata
                chunk_ids = [f"{judgement_id}_chunk_{i}" for i in range(len(chunks))]
                chunk_metadatas = []
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        "type": "judgement_chunk",
                        "judgement_id": judgement_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "created_at": datetime.now().isoformat()
                    }
                    if metadata:
                        chunk_metadata.update(metadata)
                    chunk_metadatas.append(chunk_metadata)
                
                # Add chunks to vector database
                self.judgements_collection.add(
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=chunk_metadatas,
                    ids=chunk_ids
                )
            
            print(f"✅ Added judgement {judgement_id} with {len(chunks)} chunks to database and vector store")
            return judgement_id
            
        except Exception as e:
            # Rollback database changes if vector storage fails
            self.db_session.rollback()
            print(f"Error adding judgement: {e}")
            raise e

    def search_laws(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant law chunks using semantic similarity"""
        if not self.laws_collection.count():
            return []
            
        query_embedding = self.get_embedding(query)
        
        try:
            results = self.laws_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results * 3, self.laws_collection.count()),  # Get more chunks to group by law_id
                include=["documents", "metadatas", "distances"]
            )
            
            # Group chunks by law_id and get the best chunks for each law
            law_chunks = {}
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    law_id = metadata.get('law_id')
                    distance = results['distances'][0][i]
                    
                    if law_id not in law_chunks:
                        law_chunks[law_id] = []
                    
                    law_chunks[law_id].append({
                        "chunk": doc,
                        "metadata": metadata,
                        "distance": distance,
                        "chunk_index": metadata.get('chunk_index', 0)
                    })
            
            # Sort chunks within each law and combine relevant chunks
            formatted_results = []
            for law_id, chunks in law_chunks.items():
                # Sort chunks by distance (best first) and then by chunk_index
                chunks.sort(key=lambda x: (x['distance'], x['chunk_index']))
                
                # Take the best chunks for this law (up to 3 chunks per law)
                best_chunks = chunks[:3]
                
                # Get law info from database
                db_law = self.db_session.query(Law).filter(Law.id == law_id).first()
                
                combined_text = "\n\n".join([chunk['chunk'] for chunk in best_chunks])
                
                result = {
                    "id": law_id,
                    "text": combined_text,
                    "full_text": db_law.full_text if db_law else combined_text,
                    "summary": db_law.summary if db_law else best_chunks[0]['metadata'].get('summary', ''),
                    "metadata": best_chunks[0]['metadata'],
                    "distance": best_chunks[0]['distance'],
                    "chunk_count": len(best_chunks)
                }
                formatted_results.append(result)
            
            # Sort by best distance and limit results
            formatted_results.sort(key=lambda x: x['distance'])
            return formatted_results[:n_results]
            
        except Exception as e:
            print(f"Error searching laws: {e}")
            return []

    def search_judgements(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant judgement chunks using semantic similarity"""
        if not self.judgements_collection.count():
            return []
            
        query_embedding = self.get_embedding(query)
        
        try:
            results = self.judgements_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results * 3, self.judgements_collection.count()),  # Get more chunks to group by judgement_id
                include=["documents", "metadatas", "distances"]
            )
            
            # Group chunks by judgement_id and get the best chunks for each judgement
            judgement_chunks = {}
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    judgement_id = metadata.get('judgement_id')
                    distance = results['distances'][0][i]
                    
                    if judgement_id not in judgement_chunks:
                        judgement_chunks[judgement_id] = []
                    
                    judgement_chunks[judgement_id].append({
                        "chunk": doc,
                        "metadata": metadata,
                        "distance": distance,
                        "chunk_index": metadata.get('chunk_index', 0)
                    })
            
            # Sort chunks within each judgement and combine relevant chunks
            formatted_results = []
            for judgement_id, chunks in judgement_chunks.items():
                # Sort chunks by distance (best first) and then by chunk_index
                chunks.sort(key=lambda x: (x['distance'], x['chunk_index']))
                
                # Take the best chunks for this judgement (up to 3 chunks per judgement)
                best_chunks = chunks[:3]
                
                # Get judgement info from database
                db_judgement = self.db_session.query(Judgement).filter(Judgement.id == judgement_id).first()
                
                combined_text = "\n\n".join([chunk['chunk'] for chunk in best_chunks])
                
                result = {
                    "id": judgement_id,
                    "text": combined_text,
                    "full_text": db_judgement.full_text if db_judgement else combined_text,
                    "metadata": best_chunks[0]['metadata'],
                    "distance": best_chunks[0]['distance'],
                    "chunk_count": len(best_chunks)
                }
                formatted_results.append(result)
            
            # Sort by best distance and limit results
            formatted_results.sort(key=lambda x: x['distance'])
            return formatted_results[:n_results]
            
        except Exception as e:
            print(f"Error searching judgements: {e}")
            return []

    def get_all_laws(self) -> List[Dict]:
        """Get all laws from the database"""
        try:
            db_laws = self.db_session.query(Law).all()
            
            formatted_results = []
            for law in db_laws:
                formatted_results.append({
                    "id": law.id,
                    "text": law.full_text,
                    "summary": law.summary,
                    "created_at": law.created_at.isoformat() if law.created_at else None
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error getting all laws: {e}")
            return []

    def get_all_law_summaries(self) -> List[str]:
        """Get only the AI-generated summaries of all laws for system prompts"""
        try:
            db_laws = self.db_session.query(Law).filter(Law.summary.isnot(None)).all()
            summaries = [law.summary for law in db_laws if law.summary and law.summary.strip()]
            return summaries
        except Exception as e:
            print(f"Error getting law summaries: {e}")
            return []

    def get_all_judgements(self) -> List[Dict]:
        """Get all judgements from the database"""
        try:
            db_judgements = self.db_session.query(Judgement).all()
            
            formatted_results = []
            for judgement in db_judgements:
                formatted_results.append({
                    "id": judgement.id,
                    "text": judgement.full_text,
                    "created_at": judgement.created_at.isoformat() if judgement.created_at else None
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error getting all judgements: {e}")
            return []

    def delete_law(self, law_id: str) -> bool:
        """Delete a law from both database and vector database"""
        try:
            # Delete from database
            db_law = self.db_session.query(Law).filter(Law.id == law_id).first()
            if db_law:
                self.db_session.delete(db_law)
                self.db_session.commit()
            
            # Delete all chunks from vector database
            try:
                # Get all chunk IDs for this law
                chunk_results = self.laws_collection.get(
                    where={"law_id": law_id},
                    include=["metadatas"]
                )
                
                if chunk_results['ids']:
                    self.laws_collection.delete(ids=chunk_results['ids'])
                    print(f"✅ Deleted law {law_id} and {len(chunk_results['ids'])} chunks")
                else:
                    print(f"✅ Deleted law {law_id} (no chunks found)")
                    
            except Exception as ve:
                print(f"Warning: Error deleting law chunks from vector DB: {ve}")
            
            return True
        except Exception as e:
            self.db_session.rollback()
            print(f"Error deleting law {law_id}: {e}")
            return False

    def delete_judgement(self, judgement_id: str) -> bool:
        """Delete a judgement from both database and vector database"""
        try:
            # Delete from database
            db_judgement = self.db_session.query(Judgement).filter(Judgement.id == judgement_id).first()
            if db_judgement:
                self.db_session.delete(db_judgement)
                self.db_session.commit()
            
            # Delete all chunks from vector database
            try:
                # Get all chunk IDs for this judgement
                chunk_results = self.judgements_collection.get(
                    where={"judgement_id": judgement_id},
                    include=["metadatas"]
                )
                
                if chunk_results['ids']:
                    self.judgements_collection.delete(ids=chunk_results['ids'])
                    print(f"✅ Deleted judgement {judgement_id} and {len(chunk_results['ids'])} chunks")
                else:
                    print(f"✅ Deleted judgement {judgement_id} (no chunks found)")
                    
            except Exception as ve:
                print(f"Warning: Error deleting judgement chunks from vector DB: {ve}")
            
            return True
        except Exception as e:
            self.db_session.rollback()
            print(f"Error deleting judgement {judgement_id}: {e}")
            return False

    def update_law(self, law_id: str, new_text: str, metadata: Optional[Dict] = None) -> Optional[str]:
        """Update a law in both local storage and vector database. Returns the new summary if successful, else None."""
        try:

            db_law = self.db_session.query(Law).filter(Law.id == law_id).first()
            if not db_law:
                print(f"Law with ID {law_id} not found")
                return None
            
            # Generate new summary
            new_summary = self.generate_law_summary(new_text)
            
            db_law.full_text = new_text
            db_law.summary = new_summary
            self.db_session.commit()

            # Delete old chunks from vector database
            try:
                chunk_results = self.laws_collection.get(
                    where={"law_id": law_id},
                    include=["metadatas"]
                )
                if chunk_results['ids']:
                    self.laws_collection.delete(ids=chunk_results['ids'])
            except Exception as ve:
                print(f"Warning: Error deleting old law chunks: {ve}")

            # Add new chunks to vector database
            chunks = self.text_splitter.split_text(new_text)
            if chunks:
                embeddings = self.get_embeddings_batch(chunks)
                chunk_ids = [f"{law_id}_chunk_{i}" for i in range(len(chunks))]
                chunk_metadatas = []
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        "type": "law_chunk",
                        "law_id": law_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "updated_at": datetime.now().isoformat(),
                        "summary": new_summary
                    }
                    if metadata:
                        chunk_metadata.update(metadata)
                    chunk_metadatas.append(chunk_metadata)
                self.laws_collection.add(
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=chunk_metadatas,
                    ids=chunk_ids
                )
            print(f"✅ Updated law {law_id} with {len(chunks)} new chunks")
            return new_summary
        except Exception as e:
            self.db_session.rollback()
            print(f"Error updating law {law_id}: {e}")
            return None

    def update_judgement(self, judgement_id: str, new_text: str, metadata: Optional[Dict] = None) -> bool:
        """Update a judgement in both database and vector database"""
        try:
            # Get existing judgement from database
            db_judgement = self.db_session.query(Judgement).filter(Judgement.id == judgement_id).first()
            if not db_judgement:
                print(f"Judgement with ID {judgement_id} not found")
                return False
            
            # Update judgement in database (simplified schema: id, full_text)
            db_judgement.full_text = new_text
            
            self.db_session.commit()
            
            # Delete old chunks from vector database
            try:
                chunk_results = self.judgements_collection.get(
                    where={"judgement_id": judgement_id},
                    include=["metadatas"]
                )
                
                if chunk_results['ids']:
                    self.judgements_collection.delete(ids=chunk_results['ids'])
            except Exception as ve:
                print(f"Warning: Error deleting old judgement chunks: {ve}")
            
            # Add new chunks to vector database
            chunks = self.text_splitter.split_text(new_text)
            
            if chunks:
                embeddings = self.get_embeddings_batch(chunks)
                
                # Prepare chunk IDs and metadata
                chunk_ids = [f"{judgement_id}_chunk_{i}" for i in range(len(chunks))]
                chunk_metadatas = []
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        "type": "judgement_chunk",
                        "judgement_id": judgement_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "updated_at": datetime.now().isoformat()
                    }
                    if metadata:
                        chunk_metadata.update(metadata)
                    chunk_metadatas.append(chunk_metadata)
                
                # Add updated chunks to vector database
                self.judgements_collection.add(
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=chunk_metadatas,
                    ids=chunk_ids
                )
            
            print(f"✅ Updated judgement {judgement_id} with {len(chunks)} new chunks")
            return True
            
        except Exception as e:
            self.db_session.rollback()
            print(f"Error updating judgement {judgement_id}: {e}")
            return False

    def get_relevant_context(self, query: str, max_laws: int = 3, max_judgements: int = 3) -> Tuple[List[Dict], List[Dict]]:
        """Get relevant laws and judgements for a given query"""
        relevant_laws = self.search_laws(query, n_results=max_laws)
        relevant_judgements = self.search_judgements(query, n_results=max_judgements)
        
        return relevant_laws, relevant_judgements

    def format_laws_for_prompt(self, laws: List[Dict]) -> str:
        """Format laws for use in AI prompts"""
        if not laws:
            return "No relevant laws found."
        
        formatted_laws = []
        for i, law in enumerate(laws):
            # Get the text content - prefer full_text from DB, then text from search results
            law_text = ""
            if 'full_text' in law and law['full_text']:
                law_text = law['full_text']
            elif 'text' in law and law['text']:
                law_text = law['text']
            else:
                law_text = "Text unavailable"
            
            formatted_laws.append(f"{i+1}. LAW TEXT: {law_text}")

        return "\n\n".join(formatted_laws)

    def format_judgements_for_prompt(self, judgements: List[Dict]) -> str:
        """Format judgements for use in AI prompts"""
        if not judgements:
            return "No relevant judgements found."
        
        formatted_judgements = []
        for i, judgement in enumerate(judgements):
            # Get the text content - prefer full_text from DB, then text from search results
            judgement_text = ""
            if 'full_text' in judgement and judgement['full_text']:
                judgement_text = judgement['full_text']
            elif 'text' in judgement and judgement['text']:
                judgement_text = judgement['text']
            else:
                judgement_text = "Text unavailable"
            
            formatted_judgements.append(f"{i+1}. JUDGEMENT TEXT: {judgement_text}")

        return "\n\n".join(formatted_judgements)


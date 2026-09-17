"""
RAG Engine using chromadb
"""

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

# Use the correct import paths for newer langchain
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ Using langchain_huggingface")
except ImportError:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        print("✅ Using langchain.embeddings (legacy)")
    except ImportError:
        print("⚠️ HuggingFaceEmbeddings not available")

# Correct import for text splitter
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print("✅ Using langchain.text_splitter")
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        print("✅ Using langchain_text_splitters")
    except ImportError:
        print("⚠️ Text splitter not available")

# Correct import for document loaders
try:
    from langchain.document_loaders import DirectoryLoader, TextLoader
    print("✅ Using langchain.document_loaders")
except ImportError:
    try:
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        print("✅ Using langchain_community.document_loaders")
    except ImportError:
        print("⚠️ Document loaders not available")

class RAGEngine:
    def __init__(self, persist_directory="./data/vector_store"):
        self.persist_directory = persist_directory
        self.collection = None
        self.embeddings = None
        self.embedding_fn = None  # <-- ADD THIS LINE

        try:
            # Try to use langchain embeddings
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                print("✅ Using HuggingFaceEmbeddings from langchain_huggingface")
            except ImportError:
                # Fallback to chromadb's built-in embedding
                self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                print("✅ Using chromadb SentenceTransformerEmbeddingFunction")

            # Initialize client
            self.client = chromadb.PersistentClient(path=persist_directory)

            # Try to get existing collection
            try:
                self.collection = self.client.get_collection("logs")
                print(f"✅ RAG loaded collection with {self.collection.count()} documents")
            except:
                print("⚠️ No existing collection found")

        except Exception as e:
            print(f"⚠️ RAG initialization error: {e}")
            self.client = None

    def load_logs(self, log_directory="./data/logs"):
        """Load log files into vector store"""
        # Try to import with fallbacks
        try:
            from langchain_community.document_loaders import DirectoryLoader, TextLoader
        except ImportError:
            try:
                from langchain.document_loaders import DirectoryLoader, TextLoader
            except ImportError:
                print("❌ Document loaders not available. Install: pip install langchain-community")
                return 0

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            try:
                from langchain.text_splitter import RecursiveCharacterTextSplitter
            except ImportError:
                print("❌ Text splitter not available. Install: pip install langchain-text-splitters")
                return 0

        if not self.client:
            print("❌ chromadb not available")
            return 0

        try:
            # Load documents
            loader = DirectoryLoader(
                log_directory,
                glob="*.log",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            documents = loader.load()

            if not documents:
                print("⚠️ No log files found")
                return 0

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            texts = text_splitter.split_documents(documents)

            # Create or get collection
            try:
                self.client.delete_collection("logs")
            except:
                pass

            # Use chromadb's embedding function
            self.collection = self.client.create_collection(
                name="logs",
                embedding_function=self.embedding_fn  # <-- This now exists!
            )

            # Add documents
            for i, doc in enumerate(texts):
                self.collection.add(
                    documents=[doc.page_content],
                    ids=[f"doc_{i}"]
                )

            print(f"✅ Loaded {len(texts)} chunks into vector store")
            return len(texts)

        except Exception as e:
            print(f"❌ Log load error: {e}")
            return 0

    def retrieve(self, query: str, k: int = 5):
        """Retrieve relevant log chunks"""
        if not self.collection:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return []

import os
import logging
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
from pathlib import Path
import aiohttp
from urllib.parse import quote_plus, urljoin
import html
import openai
from bs4 import BeautifulSoup

# Check for required packages and provide helpful error messages
missing_packages = []

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
except ImportError:
    missing_packages.append("python-telegram-bot")

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
    print("⚠️  PyPDF2 not found - PDF processing will be disabled")

try:
    import docx
except ImportError:
    docx = None
    print("⚠️  python-docx not found - DOCX processing will be disabled")

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("⚠️  sentence-transformers/sklearn not found - using simple text search")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  openai not found - using simple response generation")

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env file loaded")
except ImportError:
    print("⚠️  python-dotenv not found - make sure to set environment variables manually")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

if missing_packages:
    print(f"❌ Missing required packages: {', '.join(missing_packages)}")
    print("Please install them using: pip install " + " ".join(missing_packages))
    exit(1)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class WebSearchEngine:
    """Web search engine using multiple sources"""
    async def _scrape_wikipedia_events(self, year: str) -> Optional[str]:
        try:
            session = await self._get_session()
            url = f"https://en.wikipedia.org/wiki/{year}"
            async with session.get(url) as response:
                if response.status == 200:
                    html_text = await response.text()
                    soup = BeautifulSoup(html_text, 'html.parser')
                
                    # Find the Events section
                    events_header = soup.find(id="Events")
                    if not events_header:
                        return None

                    # Find the next <ul> after Events heading
                    events_list = events_header.find_next("ul")
                    if not events_list:
                        return None

                    # Return first few events as a bullet list
                    events = events_list.find_all("li")[:5]
                    event_text = "\n".join(f"• {e.get_text()}" for e in events)
                    return f"📅 **Major Events in {year}:**\n\n{event_text}\n\n🔗 [Wikipedia]({url})"
        except Exception as e:
            logging.error(f"Error scraping Wikipedia events for {year}: {e}")
            return None
    
    def __init__(self):
        self.session = None
        self.search_engines = {
            'duckduckgo': self._search_duckduckgo,
            'wikipedia': self._search_wikipedia,
            'searx': self._search_searx,
            'brave': self._search_brave,
        }

    async def _get_session(self):
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            )
        return self.session

    async def _search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        try:
            session = await self._get_session()
            url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    if data.get('Abstract'):
                        results.append({
                            'title': data.get('AbstractSource', 'DuckDuckGo'),
                            'snippet': data['Abstract'],
                            'url': data.get('AbstractURL', ''),
                            'source': 'duckduckgo_abstract'
                        })
                    for topic in data.get('RelatedTopics', [])[:3]:
                        if isinstance(topic, dict) and topic.get('Text'):
                            results.append({
                                'title': topic.get('FirstURL', '').split('/')[-1].replace('_', ' '),
                                'snippet': topic['Text'],
                                'url': topic.get('FirstURL', ''),
                                'source': 'duckduckgo_related'
                            })
                    return results[:max_results]
        except Exception as e:
            logging.error(f"DuckDuckGo search error: {e}")
        return []

    async def _search_wikipedia(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        try:
            session = await self._get_session()
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(query)}"
            async with session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('extract'):
                        return [{
                            'title': data.get('title', query),
                            'snippet': data['extract'],
                            'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                            'source': 'wikipedia'
                        }]
        except Exception as e:
            logging.error(f"Wikipedia search error: {e}")
        return []

    async def _search_searx(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        try:
            session = await self._get_session()
            searx_url = f"https://searx.be/search?q={quote_plus(query)}&format=json"
            async with session.get(searx_url) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    for item in data.get('results', [])[:max_results]:
                        results.append({
                            'title': item.get('title', 'Searx Result'),
                            'snippet': item.get('content', ''),
                            'url': item.get('url', ''),
                            'source': 'searx'
                        })
                    return results
        except Exception as e:
            logging.error(f"Searx search error: {e}")
        return []

    async def _search_brave(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        brave_api_key = os.getenv("BRAVE_API_KEY")
        if not brave_api_key:
            return []
        try:
            session = await self._get_session()
            headers = {"Accept": "application/json", "X-Subscription-Token": brave_api_key}
            url = f"https://api.search.brave.com/res/v1/web/search?q={quote_plus(query)}"
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    for item in data.get('web', {}).get('results', [])[:max_results]:
                        results.append({
                            'title': item.get('title', 'Brave Search Result'),
                            'snippet': item.get('description', ''),
                            'url': item.get('url', ''),
                            'source': 'brave'
                        })
                    return results
        except Exception as e:
            logging.error(f"Brave search error: {e}")
        return []

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        all_results = []
        tasks = [func(query, max_results) for func in self.search_engines.values()]
        try:
            search_results = await asyncio.gather(*tasks, return_exceptions=True)
            for results in search_results:
                if isinstance(results, list):
                    all_results.extend(results)
                elif isinstance(results, Exception):
                    logging.error(f"Search engine error: {results}")
            seen_urls = set()
            unique_results = []
            for result in all_results:
                url = result.get('url', '')
                if url not in seen_urls or not url:
                    seen_urls.add(url)
                    unique_results.append(result)
                    if len(unique_results) >= max_results:
                        break
            return unique_results
        except Exception as e:
            logging.error(f"Web search error: {e}")
            return []

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

class DocumentProcessor:
    """Handles extraction of text from various document formats"""
    
    def __init__(self):
        self.supported_formats = ['.txt']
        if PyPDF2:
            self.supported_formats.append('.pdf')
        if docx:
            self.supported_formats.append('.docx')
    
    def extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        if not PyPDF2:
            return "PDF processing not available - install PyPDF2"
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting PDF {file_path}: {e}")
            return ""
    
    def extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        if not docx:
            return "DOCX processing not available - install python-docx"
        
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX {file_path}: {e}")
            return ""
    
    def extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting TXT {file_path}: {e}")
            return ""
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document and return structured data"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return {}
        
        if file_path.suffix.lower() not in self.supported_formats:
            logger.warning(f"Unsupported file format: {file_path.suffix}")
            return {}
        
        # Extract text based on file extension
        if file_path.suffix.lower() == '.pdf':
            text = self.extract_from_pdf(str(file_path))
        elif file_path.suffix.lower() == '.docx':
            text = self.extract_from_docx(str(file_path))
        elif file_path.suffix.lower() == '.txt':
            text = self.extract_from_txt(str(file_path))
        else:
            return {}
        
        if not text.strip():
            logger.warning(f"No text extracted from {file_path}")
            return {}
        
        # Basic metadata extraction
        document_data = {
            'filename': file_path.name,
            'filepath': str(file_path),
            'text': text,
            'word_count': len(text.split()),
            'processed_at': datetime.now().isoformat()
        }
        
        # Try to extract key information using regex patterns
        document_data.update(self._extract_metadata(text))
        
        return document_data
    
    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text using patterns"""
        metadata = {}
        
        # Extract dates (various formats)
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b'
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        metadata['dates'] = list(set(dates))
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        metadata['emails'] = re.findall(email_pattern, text)
        
        # Extract URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        metadata['urls'] = re.findall(url_pattern, text)
        
        # Extract amounts/funding (simple pattern)
        amount_pattern = r'€?\$?[0-9,]+(?:\.[0-9]{2})?\s*(?:EUR|USD|euro|dollar|k|K|million|M)?'
        metadata['amounts'] = re.findall(amount_pattern, text)
        
        return metadata

class SimpleSearchEngine:
    """Simple keyword-based search engine"""
    
    def __init__(self):
        self.documents = []
        self.chunks = []
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the search index"""
        self.documents = documents
        
        # Create text chunks
        self.chunks = []
        for doc in documents:
            text = doc['text']
            paragraphs = text.split('\n\n')
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    chunk = {
                        'text': paragraph.strip().lower(),
                        'original_text': paragraph.strip(),
                        'document': doc,
                        'chunk_id': f"{doc['filename']}_{i}"
                    }
                    self.chunks.append(chunk)
        
        logger.info(f"Indexed {len(self.chunks)} text chunks")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Simple keyword-based search"""
        if not self.chunks:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        for chunk in self.chunks:
            # Simple keyword matching
            chunk_words = set(chunk['text'].split())
            common_words = query_words.intersection(chunk_words)
            
            if common_words:
                score = len(common_words) / len(query_words)
                result = {
                    'chunk': {
                        'text': chunk['original_text'],
                        'document': chunk['document'],
                        'chunk_id': chunk['chunk_id']
                    },
                    'similarity': score,
                    'document': chunk['document']
                }
                results.append(result)
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

class EnhancedAnswerGenerator:
    """Enhanced answer generator with web search and AI"""

    async def wikipedia_fallback(self, query: str) -> Optional[str]:
        try:
            # Direct page lookup
            wiki_result = await self.web_search._search_wikipedia(query)
            if wiki_result:
                snippet = wiki_result[0]['snippet']
                url = wiki_result[0].get('url', '')
                return f"📘 **{wiki_result[0]['title']}**\n\n{snippet}\n\n🔗 [Wikipedia]({url})"
        except Exception as e:
            logging.error(f"Wikipedia fallback failed: {e}")
            return None

    async def smart_year_handler(self, query: str) -> Optional[str]:
        match = re.fullmatch(r"(?:what happened in )?(\d{4})", query.strip().lower())
        if match:
            year = match.group(1)
            detailed = await self.web_search._scrape_wikipedia_events(year)
            if detailed:
                return detailed
        return None
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.web_search = WebSearchEngine()
        self.use_openai = False
        
        if OPENAI_AVAILABLE and openai_api_key and openai_api_key != "your_openai_api_key_here":
            try:
                self.client = openai.OpenAI(api_key=openai_api_key)
                # Test the connection
                self.client.models.list()
                self.use_openai = True
                logger.info("✅ OpenAI integration enabled")
            except Exception as e:
                logger.warning(f"OpenAI initialization failed: {e}")
                self.use_openai = False
    
    async def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]] = None, use_web_search: bool = True) -> str:
        """Generate comprehensive answer using documents and web search"""

        # 0. Try handling common queries like years first
        special_case = await self.smart_year_handler(query)
        if special_case:
            return special_case

        sources_info = []
        # 1. Check local documents first
        if context_chunks:
            for chunk_data in context_chunks[:2]:
                chunk = chunk_data['chunk']
                sources_info.append({
                    'type': 'document',
                    'title': f"Document: {chunk['document']['filename']}",
                    'content': chunk['text'][:500],
                    'source': chunk['document']['filename']
                })
        
        # 2. Web search for additional information
        web_results = []
        if use_web_search:
            try:
                web_results = await self.web_search.search(query, max_results=3)
                for result in web_results:
                    sources_info.append({
                        'type': 'web',
                        'title': result['title'],
                        'content': result['snippet'][:500],
                        'source': result.get('url', result.get('source', 'Web'))
                    })
            except Exception as e:
                logger.error(f"Web search error: {e}")
        
        # 3. Generate answer based on available information
        try:
            if self.use_openai:
                return await self._generate_ai_answer(query, sources_info)
            
            # Fallback to Wikipedia if no OpenAI
            wiki_answer = await self.wikipedia_fallback(query)
            if wiki_answer:
                return wiki_answer
            
            # Final fallback - simple answer from available sources
            return await self._generate_simple_answer(query, sources_info, web_results)
        except Exception as e:
            logger.error(f"Answer generation error: {e}", exc_info=True)
            return f"⚠️ I encountered an error while processing your question. Please try again later."


    async def _generate_ai_answer(self, query: str, sources_info: List[Dict[str, Any]]) -> str:
        """Generate AI-powered answer (even without sources)"""
        try:
            # If sources are available, build context from them
            if sources_info:
                context = "Based on the following information sources:\n\n"
                sources = []
                for i, source in enumerate(sources_info[:4], 1):
                    context += f"Source {i} ({source['type']} - {source['title']}):\n{source['content']}\n\n"
                    sources.append(source['title'])

                prompt = f"""You are a helpful assistant. Answer the user's question based on the provided context.
Be comprehensive but concise. If the context doesn't fully answer the question, provide what information you can and mention what might be missing.

Context:
{context}

Question: {query}

Please provide a helpful, accurate answer based on the available information."""
            else:
                # No sources — answer using general knowledge
                prompt = f"""You are a knowledgeable assistant. Please answer the following user question as clearly and accurately as possible:

Question: {query}

You may use your general knowledge. If the answer depends on specific context not provided, explain what would help clarify it."""

                sources = []

            # Call OpenAI
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a knowledgeable assistant that provides accurate, helpful answers."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.3
                )
            )

            answer = response.choices[0].message.content.strip()

            # Add sources if available
            if sources:
                answer += f"\n\n📚 **Sources:**\n" + "\n".join([f"• {source}" for source in sources[:3]])

            return answer

        except Exception as e:
            logger.error(f"AI answer generation error: {e}")
            return f"⚠️ I encountered an error while generating an AI response. Please try again or rephrase your question."

    
    async def _generate_simple_answer(self, query: str, sources_info: List[Dict[str, Any]], web_results: List[Dict[str, Any]]) -> str:
        """Generate simple answer without AI"""
        if not sources_info:
            return f"I couldn't find specific information about '{query}'. You might want to try rephrasing your question or checking online resources."
        
        answer = f"Here's what I found about '{query}':\n\n"
        
        # Add information from sources
        for i, source in enumerate(sources_info[:3], 1):
            content = source['content']
            if len(content) > 300:
                content = content[:300] + "..."
            
            icon = "📄" if source['type'] == 'document' else "🌐"
            answer += f"{icon} **{source['title']}**\n{content}\n\n"
        
        # Add sources
        sources = [source['source'] for source in sources_info if source['source']]
        if sources:
            answer += f"📚 **Sources:** {', '.join(sources[:3])}"
        
        return answer
    
    async def close(self):
        """Close resources"""
        await self.web_search.close()

class TelegramBot:
    """Enhanced Telegram bot class"""
    
    def __init__(self, telegram_token: str, openai_api_key: Optional[str] = None, documents_path: str = "documents"):
        self.telegram_token = telegram_token
        self.documents_path = Path(documents_path)
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        
        # Choose search engine based on available dependencies
        if EMBEDDING_AVAILABLE:
            logger.info("Using embedding-based search engine")
            from sentence_transformers import SentenceTransformer
            self.search_engine = EmbeddingSearchEngine()
        else:
            logger.info("Using simple keyword-based search engine")
            self.search_engine = SimpleSearchEngine()
        
        # Enhanced answer generator with web search
        self.answer_generator = EnhancedAnswerGenerator(openai_api_key)
        
        # Load documents on initialization
        self._load_documents()
    
    def _load_documents(self):
        """Load and process all documents in the documents directory"""
        if not self.documents_path.exists():
            logger.warning(f"Documents directory not found: {self.documents_path}")
            self.documents_path.mkdir(parents=True, exist_ok=True)
            return
        
        documents = []
        supported_extensions = self.doc_processor.supported_formats
        
        for file_path in self.documents_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                logger.info(f"Processing document: {file_path}")
                doc_data = self.doc_processor.process_document(file_path)
                if doc_data:
                    documents.append(doc_data)
        
        if documents:
            self.search_engine.add_documents(documents)
            logger.info(f"Loaded {len(documents)} documents")
        else:
            logger.warning("No documents were loaded")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        ai_status = "AI-powered with web search" if self.answer_generator.use_openai else "Enhanced with web search"
        
        welcome_message = f"""
🤖 **Enhanced Information Assistant Bot**

Hello! I can help you find information about **anything** you ask!

**Enhanced capabilities:**
✅ Web search integration
✅ Document analysis: {', '.join(self.doc_processor.supported_formats)}
✅ Answer generation: {ai_status}
✅ Multiple information sources

**Ask me about anything:**
• Current events and news
• Technical questions
• General knowledge
• Specific research topics
• Your uploaded documents
• And much more!

Just type your question and I'll search through documents and the web to give you comprehensive answers! 🔍🌐
        """
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_message = """
🆘 **How to use this enhanced bot:**

1. **Ask anything**: Type any question you have
2. **Get comprehensive answers**: I'll search documents + web
3. **Multiple sources**: You'll get information from various sources

**Example questions:**
• "What's the latest news about AI?"
• "How does photosynthesis work?"
• "What are the best programming languages in 2024?"
• "Explain quantum computing"
• "What's happening in climate change?"

**Commands:**
/start - Welcome message
/help - This help message  
/status - Check bot status
/reload - Reload documents
/websearch [query] - Force web-only search
/docsearch [query] - Search only documents

**Features:**
🌐 Web search for current information
📄 Document analysis for your files
🤖 AI-powered comprehensive answers
        """
        await update.message.reply_text(help_message, parse_mode='Markdown')
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        num_docs = len(self.search_engine.documents) if hasattr(self.search_engine, 'documents') else 0
        num_chunks = len(self.search_engine.chunks) if hasattr(self.search_engine, 'chunks') else 0
        ai_status = "AI + Web Search" if self.answer_generator.use_openai else "Enhanced with Web Search"
        
        status_message = f"""
📊 **Enhanced Bot Status:**

📁 Documents loaded: {num_docs}
📝 Text chunks indexed: {num_chunks}
🔍 Search engine: {'Embedding' if EMBEDDING_AVAILABLE else 'Keyword'}
🤖 Answer generator: {ai_status}
🌐 Web search: ✅ Enabled
📂 Document formats: {', '.join(self.doc_processor.supported_formats)}

**Search capabilities:**
• DuckDuckGo integration
• Wikipedia integration  
• Document analysis
• Multi-source answers
        """
        await update.message.reply_text(status_message, parse_mode='Markdown')
    
    async def reload_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reload command"""
        await update.message.reply_text("🔄 Reloading documents...")
        self._load_documents()
        await update.message.reply_text("✅ Documents reloaded successfully!")
    
    async def websearch_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /websearch command - web search only"""
        if not context.args:
            await update.message.reply_text("Please provide a search query. Example: `/websearch latest AI news`", parse_mode='Markdown')
            return
        
        query = ' '.join(context.args)
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
        
        try:
            # Web search only
            response = await self.answer_generator.generate_answer(query, context_chunks=None, use_web_search=True)
            await update.message.reply_text(f"🌐 **Web Search Results:**\n\n{response}", parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Web search error: {e}")
            await update.message.reply_text("Sorry, I encountered an error during web search. Please try again.")
    
    async def docsearch_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /docsearch command - document search only"""
        if not context.args:
            await update.message.reply_text("Please provide a search query. Example: `/docsearch funding opportunities`", parse_mode='Markdown')
            return
        
        query = ' '.join(context.args)
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
        
        try:
            # Document search only
            search_results = self.search_engine.search(query, top_k=5)
            response = await self.answer_generator.generate_answer(query, context_chunks=search_results, use_web_search=False)
            await update.message.reply_text(f"📄 **Document Search Results:**\n\n{response}", parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Document search error: {e}")
            await update.message.reply_text("Sorry, I encountered an error during document search. Please try again.")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages with enhanced capabilities"""
        user_query = update.message.text.strip()
        
        if not user_query:
            await update.message.reply_text("Please ask me anything - I can search both documents and the web!")
            return
        
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
        
        try:
            # Search documents first
            document_results = self.search_engine.search(user_query, top_k=3)
            
            # Generate comprehensive answer using both documents and web search
            response = await self.answer_generator.generate_answer(
                user_query, 
                context_chunks=document_results, 
                use_web_search=True
            )
            
            # Split long messages if needed
            if len(response) > 4000:
                chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
                for chunk in chunks:
                    await update.message.reply_text(chunk, parse_mode='Markdown')
            else:
                await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await update.message.reply_text("Sorry, I encountered an error while processing your request. Please try again.")
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.answer_generator.close()
    
    def run(self):
        """Run the Telegram bot"""
        # Create application
        application = Application.builder().token(self.telegram_token).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("status", self.status_command))
        application.add_handler(CommandHandler("reload", self.reload_command))
        application.add_handler(CommandHandler("websearch", self.websearch_command))
        application.add_handler(CommandHandler("docsearch", self.docsearch_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Start the bot
        logger.info("Starting Enhanced Telegram bot with web search...")
        
        try:
            application.run_polling(allowed_updates=Update.ALL_TYPES)
        finally:
            # Cleanup
            asyncio.run(self.cleanup())

# Add embedding search engine class (if available)
if EMBEDDING_AVAILABLE:
    class EmbeddingSearchEngine:
        """Embedding-based search engine for documents"""
        
        def __init__(self):
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.documents = []
            self.embeddings = None
            self.chunks = []
        
        def add_documents(self, documents: List[Dict[str, Any]]):
            """Add documents to the search index"""
            self.documents = documents
            
            # Create text chunks for better search granularity
            self.chunks = []
            for doc in documents:
                text = doc['text']
                paragraphs = text.split('\n\n')
                for i, paragraph in enumerate(paragraphs):
                    if paragraph.strip():
                        chunk = {
                            'text': paragraph.strip(),
                            'document': doc,
                            'chunk_id': f"{doc['filename']}_{i}"
                        }
                        self.chunks.append(chunk)
            
            # Create embeddings
            if self.chunks:
                texts = [chunk['text'] for chunk in self.chunks]
                self.embeddings = self.model.encode(texts)
                logger.info(f"Created embeddings for {len(self.chunks)} text chunks")
        
        def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
            """Search for relevant documents/chunks"""
            if not self.chunks or self.embeddings is None:
                return []
            
            # Encode query
            query_embedding = self.model.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    result = {
                        'chunk': self.chunks[idx],
                        'similarity': float(similarities[idx]),
                        'document': self.chunks[idx]['document']
                    }
                    results.append(result)
            
            return results

def main():
    """Main function to run the enhanced bot"""
    # Load environment variables
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    DOCUMENTS_PATH = os.getenv('DOCUMENTS_PATH', 'documents')
    
    print(f"🔍 Checking environment variables...")
    print(f"📱 Telegram token: {'✅ Set' if TELEGRAM_TOKEN else '❌ Missing'}")
    print(f"🤖 OpenAI API key: {'✅ Set' if OPENAI_API_KEY and OPENAI_API_KEY != 'your_openai_api_key_here' else '⚠️ Missing or placeholder'}")
    print(f"📁 Documents path: {DOCUMENTS_PATH}")
    print(f"🌐 Web search: ✅ Enabled")
    print(f"🔍 Advanced search: {'✅ Embedding-based' if EMBEDDING_AVAILABLE else '⚠️ Keyword-based'}")
    
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN environment variable is required")
        print("\n❌ Please set TELEGRAM_TOKEN in your .env file")
        print("Get your token from @BotFather on Telegram")
        return
    
    # Create and run enhanced bot
    bot = TelegramBot(
        telegram_token=TELEGRAM_TOKEN,
        openai_api_key=OPENAI_API_KEY,
        documents_path=DOCUMENTS_PATH
    )
    
    print("\n🚀 Starting Enhanced Telegram Bot with:")
    print("   • Web search capabilities")
    print("   • Document analysis")
    print("   • AI-powered answers (if OpenAI key provided)")
    print("   • Multi-source information gathering")
    print("\n📝 You can now ask your bot about ANYTHING!")
    
    bot.run()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
API FastAPI super otimizada para RAG com papers em Raspberry Pi usando Ollama.
Gerenciamento inteligente de contexto para 32k window com múltiplos papers.
"""
import os
import asyncio
import aiohttp
import json
import hashlib
from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from collections import deque
import time

# Configuração de logging ultra-otimizada
logging.basicConfig(
    level=logging.ERROR,  # Só erros críticos
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Desabilitar paralelismo desnecessário
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# --- Configurações Otimizadas ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"

# Gestão inteligente de contexto para 32k window
MAX_CONTEXT_TOKENS = 28000  # Reserva espaço para prompt e resposta
TOKENS_PER_CHAR = 0.25     # Aproximação conservadora
MAX_CONTEXT_CHARS = int(MAX_CONTEXT_TOKENS / TOKENS_PER_CHAR)
OPTIMAL_CHUNKS = 12        # Número ideal de chunks para análise
CHUNK_OVERLAP_REDUCTION = 0.7  # Reduz sobreposição para economizar espaço

# Performance para Raspberry Pi
OLLAMA_TIMEOUT = 180
REQUEST_TIMEOUT = 8
CACHE_MAX_SIZE = 200
PRECOMPUTE_CACHE_SIZE = 50

@dataclass
class ContextChunk:
    content: str
    relevance_score: float
    source: str
    chunk_id: str
    
    def estimated_tokens(self) -> int:
        return int(len(self.content) * TOKENS_PER_CHAR)

class SmartCache:
    """Cache inteligente com TTL e priorização por relevância"""
    def __init__(self, max_size: int = CACHE_MAX_SIZE):
        self.cache: Dict[str, Tuple[str, float, float]] = {}  # key: (response, timestamp, score)
        self.max_size = max_size
        self.access_count: Dict[str, int] = {}
    
    def get(self, query: str) -> Optional[str]:
        key = self._hash_query(query)
        if key in self.cache:
            response, timestamp, score = self.cache[key]
            # Cache válido por 1 hora
            if time.time() - timestamp < 3600:
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return response
            else:
                del self.cache[key]
                self.access_count.pop(key, None)
        return None
    
    def set(self, query: str, response: str, relevance_score: float = 1.0):
        key = self._hash_query(query)
        
        if len(self.cache) >= self.max_size:
            # Remove entrada menos usada e mais antiga
            worst_key = min(
                self.cache.keys(),
                key=lambda k: (self.access_count.get(k, 0), self.cache[k][1])
            )
            del self.cache[worst_key]
            self.access_count.pop(worst_key, None)
        
        self.cache[key] = (response, time.time(), relevance_score)
        self.access_count[key] = 1
    
    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

class ContextOptimizer:
    """Otimizador inteligente de contexto para maximizar relevância"""
    
    @staticmethod
    def smart_truncate(text: str, max_chars: int) -> str:
        """Trunca texto preservando frases completas"""
        if len(text) <= max_chars:
            return text
        
        # Tenta cortar em ponto final
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        if last_period > max_chars * 0.8:  # Se o ponto estiver nos últimos 20%
            return truncated[:last_period + 1]
        
        # Senão, corta em espaço
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.9:
            return truncated[:last_space]
        
        return truncated + "..."
    
    @staticmethod
    def rank_chunks(chunks: List[ContextChunk], query: str) -> List[ContextChunk]:
        """Reordena chunks por relevância usando heurísticas simples"""
        query_words = set(query.lower().split())
        
        for chunk in chunks:
            content_words = set(chunk.content.lower().split())
            # Score baseado em sobreposição de palavras
            overlap = len(query_words.intersection(content_words))
            chunk.relevance_score = overlap / len(query_words) if query_words else 0
        
        return sorted(chunks, key=lambda x: x.relevance_score, reverse=True)
    
    @staticmethod
    def optimize_context_window(chunks: List[ContextChunk], max_chars: int) -> str:
        """Otimiza contexto para caber na janela disponível"""
        if not chunks:
            return ""
        
        selected_chunks = []
        total_chars = 0
        
        # Prioriza chunks mais relevantes
        for chunk in chunks:
            chunk_chars = len(chunk.content)
            
            if total_chars + chunk_chars <= max_chars:
                selected_chunks.append(chunk)
                total_chars += chunk_chars
            else:
                # Tenta incluir versão truncada do chunk
                remaining_chars = max_chars - total_chars
                if remaining_chars > 200:  # Mínimo para ser útil
                    truncated_content = ContextOptimizer.smart_truncate(
                        chunk.content, remaining_chars
                    )
                    selected_chunks.append(ContextChunk(
                        content=truncated_content,
                        relevance_score=chunk.relevance_score,
                        source=chunk.source,
                        chunk_id=chunk.chunk_id + "_trunc"
                    ))
                break
        
        # Monta contexto final com separadores eficientes
        context_parts = []
        for i, chunk in enumerate(selected_chunks):
            context_parts.append(f"[Doc{i+1}] {chunk.content}")
        
        return "\n\n".join(context_parts)

# Instâncias globais
smart_cache = SmartCache()
context_optimizer = ContextOptimizer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia ciclo de vida com inicialização assíncrona otimizada"""
    global chat_session, aiohttp_session
    aiohttp_session = None # Garantir que a variável exista no escopo
    
    try:
        logger.info("Carregando memória Memvid...")
        await load_memory_async()
        
        # Sessão HTTP otimizada para Raspberry Pi
        connector = aiohttp.TCPConnector(
            limit=5,  # Menos conexões simultâneas
            limit_per_host=3,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            # force_close=True, # REMOVIDO: Conflitava com keepalive_timeout
            ttl_dns_cache=600
        )
        
        timeout = aiohttp.ClientTimeout(
            total=OLLAMA_TIMEOUT,
            connect=REQUEST_TIMEOUT
        )
        
        aiohttp_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        logger.info("API otimizada pronta!")
        yield
        
    finally:
        if aiohttp_session and not aiohttp_session.closed:
            await aiohttp_session.close()

# Aplicação FastAPI
app = FastAPI(
    title="Memvid RAG API - Raspberry Pi Optimized",
    description="API ultra-otimizada para RAG com papers usando Gemma2:1b",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# Modelos Pydantic
class ChatQuery(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    use_cache: bool = Field(True)
    detail_level: str = Field("medium", pattern="^(brief|medium|detailed)$")

class ChatResponse(BaseModel):
    answer: str
    cached: bool = False
    context_chunks: int = 0
    processing_time: float = 0.0
    context_chars_used: int = 0

# Variáveis globais
chat_session = None
aiohttp_session = None

async def load_memory_async():
    """Carregamento assíncrono da memória"""
    global chat_session
    
    # Adiciona path para memvid
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from memvid import MemvidChat
    from memvid.config import get_codec_parameters
    
    codec_usado_na_memoria = 'mp4v'
    video_extension = get_codec_parameters(codec_usado_na_memoria)["video_file_type"]
    VIDEO_FILE = f"output/memory.{video_extension}"
    INDEX_FILE = "output/memory_index.json"
    
    if not os.path.exists(VIDEO_FILE) or not os.path.exists(INDEX_FILE):
        raise RuntimeError(
            f"Arquivos de memória não encontrados!\n"
            f"Execute 'python build_memory.py' primeiro."
        )
    
    loop = asyncio.get_event_loop()
    chat_session = await loop.run_in_executor(
        None, 
        lambda: MemvidChat(video_file=VIDEO_FILE, index_file=INDEX_FILE)
    )
    
    await loop.run_in_executor(None, chat_session.start_session)

async def search_and_optimize_context(query: str, detail_level: str) -> Tuple[str, int]:
    """Busca e otimiza contexto baseado no nível de detalhe"""
    # Ajusta número de chunks baseado no nível de detalhe
    chunk_counts = {"brief": 6, "medium": 10, "detailed": 15}
    target_chunks = chunk_counts.get(detail_level, 10)
    
    loop = asyncio.get_event_loop()
    raw_results = await loop.run_in_executor(
        None,
        lambda: chat_session.search_context(query, top_k=target_chunks)
    )
    
    if not raw_results:
        return "", 0
    
    # Converte para objetos ContextChunk
    chunks = [
        ContextChunk(
            content=result,
            relevance_score=0.0,  # Será calculado
            source=f"paper_{i}",
            chunk_id=f"chunk_{i}"
        )
        for i, result in enumerate(raw_results)
    ]
    
    # Otimiza ordem por relevância
    ranked_chunks = context_optimizer.rank_chunks(chunks, query)
    
    # Otimiza para caber na janela de contexto
    max_context_chars = int(MAX_CONTEXT_CHARS * 0.85)  # Margem de segurança
    optimized_context = context_optimizer.optimize_context_window(
        ranked_chunks, max_context_chars
    )
    
    return optimized_context, len(ranked_chunks)

async def query_ollama_optimized(prompt: str, detail_level: str) -> str:
    """Query otimizada para Ollama com configurações por nível de detalhe"""
    
    # Configurações por nível de detalhe
    configs = {
        "brief": {"num_predict": 300, "temperature": 0.2},
        "medium": {"num_predict": 600, "temperature": 0.3},
        "detailed": {"num_predict": 1200, "temperature": 0.4}
    }
    
    config = configs.get(detail_level, configs["medium"])
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": MAX_CONTEXT_TOKENS,
            "temperature": config["temperature"],
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_predict": config["num_predict"],
            "stop": ["PERGUNTA:", "CONTEXTO:", "---END---"]
        }
    }
    
    try:
        async with aiohttp_session.post(OLLAMA_API_URL, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("response", "").strip()
            
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timeout - query muito complexa")
    except Exception as e:
        logger.error(f"Erro Ollama: {e}")
        raise HTTPException(status_code=503, detail="Erro no processamento")

def create_optimized_prompt(context: str, query: str, detail_level: str) -> str:
    """Cria prompt otimizado baseado no contexto e nível de detalhe"""
    
    detail_instructions = {
        "brief": "Responda de forma concisa e direta, focando nos pontos principais.",
        "medium": "Forneça uma resposta equilibrada com explicações claras dos conceitos principais.",
        "detailed": "Analise profundamente, explicando metodologias, resultados e implicações."
    }
    
    instruction = detail_instructions.get(detail_level, detail_instructions["medium"])
    
    return f"""Você é um especialista em análise de papers acadêmicos e documentos técnicos.

CONTEXTO DOS DOCUMENTOS:
{context}

PERGUNTA: {query}

INSTRUÇÕES: {instruction}
- Use apenas informações do contexto fornecido
- Cite detalhes específicos quando relevante
- Se não houver informação suficiente, mencione claramente
- Mantenha resposta estruturada e precisa

RESPOSTA:"""

@app.get("/health")
async def health_check():
    """Status da API"""
    return {
        "status": "healthy",
        "model": OLLAMA_MODEL,
        "memory_loaded": chat_session is not None,
        "cache_entries": len(smart_cache.cache),
        "max_context_chars": MAX_CONTEXT_CHARS
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatQuery):
    """Endpoint principal para chat RAG otimizado"""
    start_time = time.time()
    
    if not chat_session:
        raise HTTPException(status_code=503, detail="Sessão não iniciada")
    
    query = request.query.strip()
    
    # Verifica cache primeiro
    if request.use_cache:
        cached_response = smart_cache.get(query)
        if cached_response:
            return ChatResponse(
                answer=cached_response,
                cached=True,
                processing_time=time.time() - start_time
            )
    
    try:
        # Busca e otimiza contexto
        context, num_chunks = await search_and_optimize_context(
            query, request.detail_level
        )
        
        if not context:
            no_context_answer = "Não encontrei informações relevantes nos documentos para responder sua pergunta."
            return ChatResponse(
                answer=no_context_answer,
                context_chunks=0,
                processing_time=time.time() - start_time
            )
        
        # Cria prompt otimizado
        prompt = create_optimized_prompt(context, query, request.detail_level)
        
        # Consulta Ollama
        answer = await query_ollama_optimized(prompt, request.detail_level)
        
        # Adiciona ao cache
        if request.use_cache and answer:
            smart_cache.set(query, answer, 1.0)
        
        return ChatResponse(
            answer=answer,
            cached=False,
            context_chunks=num_chunks,
            processing_time=time.time() - start_time,
            context_chars_used=len(context)
        )
        
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")
        raise HTTPException(status_code=500, detail="Erro interno")

@app.get("/stats")
async def get_stats():
    """Estatísticas da API"""
    memory_stats = {}
    if chat_session:
        stats = chat_session.retriever.get_stats()
        memory_stats = stats.get('index_stats', {})
    
    return {
        "memory_stats": memory_stats,
        "cache_stats": {
            "size": len(smart_cache.cache),
            "max_size": smart_cache.max_size,
            "access_counts": len(smart_cache.access_count)
        },
        "config": {
            "model": OLLAMA_MODEL,
            "max_context_chars": MAX_CONTEXT_CHARS,
            "optimal_chunks": OPTIMAL_CHUNKS
        }
    }

@app.delete("/cache")
async def clear_cache():
    """Limpa cache"""
    cache_size = len(smart_cache.cache)
    smart_cache.cache.clear()
    smart_cache.access_count.clear()
    return {"message": f"Cache limpo. {cache_size} entradas removidas."}

if __name__ == "__main__":
    import uvicorn
    
    # Configuração ultra-otimizada para Raspberry Pi
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="asyncio",
        log_level="error",  # Só erros
        access_log=False,
        limit_concurrency=8,  # Limite baixo para Raspberry Pi
        timeout_keep_alive=60,
        h11_max_incomplete_event_size=16384  # Economiza memória
    )
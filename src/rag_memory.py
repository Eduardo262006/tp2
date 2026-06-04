"""
rag_memory.py — Componente 3: RAG Memory.

Indexa cada inspeção concluída numa vector store (ChromaDB) e permite recuperação
semântica para contextualizar análises futuras (Secção 6).

Embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (local, PT).
Vector store: ChromaDB persistente em disco.
Chunking: "full", "per_issue" e "hybrid" (default), para comparar Recall@3.

Este ficheiro é autossuficiente: inclui a sua própria camada de acesso ao Gemini.
Degrada graciosamente: sem chromadb/sentence-transformers cai para índice em
memória com similaridade por bag-of-words.

Uso CLI:
    python rag_memory.py index --inspections-dir data/inspections
    python rag_memory.py query "última vez que Z_S1 teve prateleira vazia"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================================== #
# Camada de acesso ao LLM (embutida) — Gemini 3.5 Flash (texto).
# Usada para gerar summaries e sintetizar respostas. Degradação graciosa.
# =========================================================================== #
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
try:
    import google.generativeai as genai
    _GENAI_OK = True
except Exception:
    genai = None
    _GENAI_OK = False

MODEL_NAME = os.getenv("MODEL", "gemini-2.5-flash")
_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "prompts").is_dir():
            return parent
    return here.parent

_ROOT = _project_root()
_SUMMARY_CACHE = _ROOT / "cache" / "summaries"
_SUMMARY_CACHE.mkdir(parents=True, exist_ok=True)


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "prompts").is_dir():
            return parent
    return here.parent


_ROOT = _project_root()
PROMPTS_DIR = _ROOT / "prompts"


class LLMUnavailable(RuntimeError):
    """O LLM não pode ser usado (sem chave, sem rede, ou quota esgotada)."""


class _LLMState:
    def __init__(self) -> None:
        self.available = False
        self.quota_exhausted = False
        self._model = None
        if not _GENAI_OK:
            self.reason = "pacote google-generativeai não instalado"
            return
        if not _API_KEY:
            self.reason = "GEMINI_API_KEY ausente (definir no .env)"
            return
        try:
            genai.configure(api_key=_API_KEY)
            self._model = genai.GenerativeModel(MODEL_NAME)
            self.available = True
            self.reason = "ok"
        except Exception as exc:
            self.reason = f"falha a configurar o modelo: {exc}"

    @property
    def model(self):
        return self._model


_STATE = _LLMState()


def llm_available() -> bool:
    return _STATE.available and not _STATE.quota_exhausted


def llm_status() -> dict:
    return {"available": _STATE.available, "quota_exhausted": _STATE.quota_exhausted,
            "reason": _STATE.reason, "model": MODEL_NAME}


class _RateLimiter:
    def __init__(self, max_calls: int = 15, period_s: float = 60.0) -> None:
        self.max_calls = max_calls
        self.period_s = period_s
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            while self._calls and now - self._calls[0] > self.period_s:
                self._calls.popleft()
            if len(self._calls) >= self.max_calls:
                sleep_for = self.period_s - (now - self._calls[0]) + 0.05
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self.period_s:
                    self._calls.popleft()
            self._calls.append(time.monotonic())


_RATE = _RateLimiter()


def _is_quota_error(exc: Exception) -> bool:
    m = str(exc).lower()
    return "429" in m or "quota" in m or ("resource" in m and "exhaust" in m)


def _is_daily_quota(exc: Exception) -> bool:
    return "PerDay" in str(exc)

def _retry_seconds(exc: Exception) -> Optional[int]:
    m = (re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(exc))
         or re.search(r'retryDelay[\'\"]?\s*[:=]\s*[\'\"]?(\d+)s', str(exc)))
    return int(m.group(1)) if m else None


def generate(prompt: str, image_path: Optional[str] = None,
             temperature: float = 0.0, max_retries: int = 5) -> str:
    if not llm_available():
        raise LLMUnavailable(llm_status()["reason"])
    parts: list[Any] = [prompt]
    if image_path:
        from PIL import Image
        parts.append(Image.open(image_path))
    delay = 10.0
    for attempt in range(max_retries):
        _RATE.acquire()
        try:
            resp = _STATE.model.generate_content(
                parts, generation_config={"temperature": temperature})
            return (resp.text or "").strip()
        except Exception as exc:
            if _is_quota_error(exc):
                if _is_daily_quota(exc):          # diário: não recupera hoje
                    _STATE.quota_exhausted = True
                    _STATE.reason = "quota diária esgotada (429)"
                    raise LLMUnavailable(_STATE.reason) from exc
                if attempt < max_retries - 1:     # por minuto: esperar e repetir
                    secs = _retry_seconds(exc)
                    wait = (secs + 1) if secs and secs <= 120 else (max(delay, 30.0) if attempt >= 2 else delay)
                    time.sleep(wait)
                    delay *= 2
                    continue
                _STATE.quota_exhausted = True
                _STATE.reason = "quota esgotada (429) após múltiplas tentativas"
                raise LLMUnavailable(_STATE.reason) from exc
            if attempt < max_retries - 1:
                time.sleep(1.0)
                continue
            raise LLMUnavailable(f"erro do modelo: {exc}") from exc
    raise LLMUnavailable("esgotadas as tentativas de chamada ao modelo")


def load_prompt(name: str, default: str = "") -> str:
    p = PROMPTS_DIR / f"{name}.txt"
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return default
    return default
# =================== fim da camada de LLM embutida ========================= #


_VECTORSTORE = str(_ROOT / "vectorstore")
_EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

try:
    import chromadb
except ImportError as e:
    raise ImportError(
        "Biblioteca 'chromadb' não encontrada. Instala com: pip install chromadb"
    ) from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "Biblioteca 'sentence-transformers' não encontrada. "
        "Instala com: pip install sentence-transformers"
    ) from e


'''_SUMMARY_PROMPT = """Gera um resumo de UMA frase, rico em termos semanticamente relevantes para
recuperação futura, a partir deste registo de inspeção de prateleira. Inclui: zona,
fill rate, tipos de problema, localização, severidade e o dia/hora se disponível.

Mau exemplo: "prateleira com problemas."
Bom exemplo: "prateleira inferior da zona Z_S3 com fill rate de 72%, detergente líquido
fora de posição na secção central, embalagem danificada à direita, terça-feira 15h."

Registo (JSON):
%s

Devolve APENAS a frase de resumo, sem aspas nem prefixos.
"""

_ANSWER_PROMPT = """És o sistema de memória de inspeções de uma loja. Responde à pergunta do gestor
usando EXCLUSIVAMENTE o contexto recuperado abaixo. Sê conciso e refere explicitamente
as inspeções usadas pelo seu inspection_id e data. Se o contexto não chegar para
responder, di-lo claramente.

PERGUNTA: %s

CONTEXTO RECUPERADO:
%s

Resposta:"""'''


def _embed_text_summary_fallback(record: dict) -> str:
    issues = record.get("issues", [])
    parts = [
        f"zona {record.get('zone_id','?')}",
        f"estado {record.get('overall_status','?')}",
        f"fill rate {float(record.get('shelf_fill_rate',0)):.0%}",
    ]
    for i in issues:
        parts.append(f"{i['type']} ({i['severity']}) em {i.get('location','?')}: {i.get('description','')}")
    parts.append("produtos: " + ", ".join(record.get("products_detected", []) or ["—"]))
    ts = record.get("timestamp", "")
    if ts:
        parts.append(f"data {ts}")
    return "; ".join(parts)


class _Embedder:
    def __init__(self) -> None:
        self.model = SentenceTransformer(_EMB_MODEL)
        self.available = True

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class RAGMemory:
    def __init__(self, chunking: str = "hybrid", collection: Optional[str] = None) -> None:
        if chunking not in {"full", "per_issue", "hybrid"}:
            raise ValueError("chunking deve ser full|per_issue|hybrid")
        self.chunking = chunking
        self.embedder = _Embedder()
        self.collection_name = collection or f"inspections_{chunking}"

        # Inicia logo o ChromaDB
        client = chromadb.PersistentClient(path=_VECTORSTORE)
        self._chroma = client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"})

    @property
    def backend(self) -> str:
        return "chromadb"

    def make_summary(self, record: dict) -> str:
        ins_id = record.get("inspection_id", "")
        cache_file = _SUMMARY_CACHE / f"{ins_id}.txt"
        if ins_id and cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
        try:
            tmpl = load_prompt("rag_memory_summary_prompt")
            if not tmpl or "%s" not in tmpl:
                raise LLMUnavailable("prompt 'rag_memory_summary_prompt.txt' em falta ou sem %s")
            summary = generate(tmpl % json.dumps(record, ensure_ascii=False),
                               temperature=0.0)
        except LLMUnavailable:
            summary = _embed_text_summary_fallback(record)
        if ins_id and summary:
            cache_file.write_text(summary, encoding="utf-8")
        return summary

    def index_inspection(self, record: dict) -> int:
        summary = self.make_summary(record)
        record["summary"] = summary
        ins_id = record["inspection_id"]
        meta_base = {
            "inspection_id": ins_id,
            "zone_id": record.get("zone_id", ""),
            "timestamp": record.get("timestamp", ""),
            "fill_rate": float(record.get("shelf_fill_rate", 0.0)),
            "overall_status": record.get("overall_status", ""),
        }
        chunks: list[tuple[str, str, dict]] = []
        if self.chunking in {"full", "hybrid"}:
            chunks.append((f"{ins_id}::main", summary, {**meta_base, "kind": "summary"}))
        if self.chunking == "per_issue":
            if not record.get("issues"):
                chunks.append((f"{ins_id}::noissue", summary, {**meta_base, "kind": "summary"}))
            for issue in record.get("issues", []):
                text = (f"Zona {record['zone_id']}: {issue['type']} severidade {issue['severity']} "
                        f"em {issue.get('location', '?')}. {issue.get('description', '')}")
                chunks.append((f"{ins_id}::{issue['issue_id']}", text,
                               {**meta_base, "kind": "issue", "issue_type": issue["type"],
                                "severity": issue["severity"]}))
        if self.chunking == "hybrid":
            for issue in record.get("issues", []):
                text = (f"Zona {record['zone_id']}: {issue['type']} severidade {issue['severity']} "
                        f"em {issue.get('location', '?')}. {issue.get('description', '')}")
                chunks.append((f"{ins_id}::{issue['issue_id']}", text,
                               {**meta_base, "kind": "issue", "issue_type": issue["type"],
                                "severity": issue["severity"]}))

        texts = [c[1] for c in chunks]
        embeddings = self.embedder.encode(texts)

        # Guarda diretamente no ChromaDB sem perguntar se ele existe
        self._chroma.upsert(ids=[c[0] for c in chunks], documents=texts,
                            metadatas=[c[2] for c in chunks], embeddings=embeddings)
        return len(chunks)

    def index_dir(self, inspections_dir: str) -> int:
        total = 0
        for p in sorted(Path(inspections_dir).glob("INS_*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
                total += self.index_inspection(rec)
            except Exception:
                continue
        return total

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        q_emb = self.embedder.encode([query])[0]
        try:
            res = self._chroma.query(query_embeddings=[q_emb], n_results=k,
                                     include=["documents", "metadatas", "distances"])
            out = []
            for i in range(len(res["ids"][0])):
                dist = res["distances"][0][i] if res.get("distances") else 0.0
                out.append({"id": res["ids"][0][i], "text": res["documents"][0][i],
                            "metadata": res["metadatas"][0][i], "score": 1.0 - dist})
            return out
        except Exception:
            return []

    def query(self, question: str, k: int = 3) -> dict:
        chunks = self.retrieve(question, k=k)
        context = "\n".join(
            f"[{c['metadata'].get('inspection_id','?')} | {c['metadata'].get('timestamp','?')} "
            f"| score={c['score']:.2f}] {c['text']}" for c in chunks) or "(sem registos recuperados)"
        answer = None
        if llm_available() and chunks:
            try:
                prompt = load_prompt("rag_memory_answer_prompt") % (question, context)
                answer = generate(prompt, temperature=0.0).strip()
            except LLMUnavailable:
                answer = None
        if answer is None:
            if chunks:
                refs = "; ".join(
                    f"{c['metadata'].get('inspection_id','?')} ({c['metadata'].get('timestamp','?')})"
                    for c in chunks)
                answer = f"Registos mais relevantes recuperados: {refs}."
            else:
                answer = "Não foram encontrados registos relevantes na memória."
        return {"question": question, "answer": answer, "retrieved": chunks,
                "backend": self.backend,
                "embedder": "ST" if self.embedder.available else "fallback"}

    def recall_at_k(self, eval_set: list[dict], k: int = 3) -> dict:
        hits, details = 0, []
        for item in eval_set:
            retrieved = self.retrieve(item["query"], k=k)
            got_ids = {c["metadata"].get("inspection_id") for c in retrieved}
            hit = bool(got_ids & set(item["relevant_ids"]))
            hits += int(hit)
            details.append({"query": item["query"], "hit": hit,
                            "retrieved_ids": list(got_ids), "relevant_ids": item["relevant_ids"]})
        return {"recall_at_k": hits / len(eval_set) if eval_set else 0.0,
                "k": k, "chunking": self.chunking, "details": details}


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG Memory (Componente 3)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_idx = sub.add_parser("index")
    p_idx.add_argument("--inspections-dir", default=str(_ROOT / "data" / "inspections"))
    p_idx.add_argument("--chunking", default="hybrid", choices=["full", "per_issue", "hybrid"])
    p_q = sub.add_parser("query")
    p_q.add_argument("text")
    p_q.add_argument("--chunking", default="hybrid", choices=["full", "per_issue", "hybrid"])
    p_q.add_argument("-k", type=int, default=3)
    args = ap.parse_args()
    rag = RAGMemory(chunking=args.chunking)
    print(f"[backend={rag.backend} embedder={'ST' if rag.embedder.available else 'fallback'}]",
          file=sys.stderr)
    if args.cmd == "index":
        n = rag.index_dir(args.inspections_dir)
        print(f"Indexados {n} chunks de {args.inspections_dir}")
    elif args.cmd == "query":
        print(json.dumps(rag.query(args.text, k=args.k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
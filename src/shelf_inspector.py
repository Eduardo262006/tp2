

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================================== #
# Camada de acesso ao LLM (embutida) — Gemini 3.5 Flash.
# Configuração via .env, rate-limiting 15 req/min com backoff, cache MD5,
# parsing JSON robusto e degradação graciosa (sem chave/rede → modo offline).
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
CACHE_DIR = _ROOT / "cache"
PROMPTS_DIR = _ROOT / "prompts"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
    """Janela deslizante: no máximo `max_calls` chamadas em `period_s` segundos."""

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


_RATE = _RateLimiter(max_calls=15, period_s=60.0)


def file_md5(path: str | os.PathLike) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def text_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class DiskCache:
    """Cache simples key->JSON em disco (uma entrada por ficheiro)."""

    def __init__(self) -> None:
        self.dir = CACHE_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{text_md5(key)}.json"

    def get(self, key: str) -> Optional[dict]:
        p = self._path(key)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def set(self, key: str, value: dict) -> None:
        try:
            self._path(key).write_text(json.dumps(value, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        except Exception:
            pass


def _is_quota_error(exc: Exception) -> bool:
    m = str(exc).lower()
    return "429" in m or "quota" in m or ("resource" in m and "exhaust" in m)


def _load_image(image_path: str):
    from PIL import Image
    return Image.open(image_path)


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


def extract_json(text: str) -> dict:
    """Extrai o primeiro objeto JSON de uma resposta (tolerante a markdown/ruído)."""
    if not text:
        raise ValueError("resposta vazia do modelo")
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("nenhum objeto JSON encontrado na resposta")
    depth = 0
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                cand = re.sub(r",\s*([}\]])", r"\1", cleaned[start:i + 1])
                try:
                    return json.loads(cand)
                except Exception:
                    continue
    raise ValueError("objeto JSON malformado na resposta")


def load_prompt(name: str, default: str) -> str:
    """Lê prompts/<name>.txt se existir; caso contrário usa o default embutido."""
    p = PROMPTS_DIR / f"{name}.txt"
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return default
    return default
# =================== fim da camada de LLM embutida ========================= #


ISSUE_TYPES = {"empty_shelf", "wrong_product", "damaged", "misaligned", "label_missing", "other"}
SEVERITIES = {"low", "medium", "high"}
STATUSES = {"ok", "warning", "critical"}

_STRATEGY_PROMPTS = {
    "A": ("shelf_inspector_zero_shot", ""),
    "B": ("shelf_inspector_chain_of_thought", ""),
    "C": ("shelf_inspector_few_shot", ""),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gen_inspection_id(seq: int = 1) -> str:
    return f"INS_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seq:03d}"


class ShelfInspector:
    """Inspetor visual de prateleiras com 3 estratégias e cache."""

    def __init__(self, strategy: str = "B") -> None:
        if strategy not in _STRATEGY_PROMPTS:
            raise ValueError(f"estratégia inválida: {strategy} (usar A, B ou C)")
        self.strategy = strategy
        self.cache = DiskCache()

    def _prompt(self) -> str:
        name, default = _STRATEGY_PROMPTS[self.strategy]
        return load_prompt(name, default)

    def _cache_key(self, image_path: str) -> str:
        return f"{file_md5(image_path)}::{self.strategy}::{MODEL_NAME}"

    def inspect(self, image_path: str, zone_id: str = "Z_UNKNOWN",
                seq: int = 1, use_cache: bool = True) -> dict:
        """Analisa uma imagem e devolve o inspection record completo. Nunca lança."""
        if not os.path.exists(image_path):
            return self._error_record(image_path, zone_id, seq, "imagem inexistente")
        if use_cache:
            cached = self.cache.get(self._cache_key(image_path))
            if cached is not None:
                cached["_from_cache"] = True
                cached["zone_id"] = zone_id
                return cached
        try:
            raw = generate(self._prompt(), image_path=image_path, temperature=0.0)
            partial = extract_json(raw)
        except (LLMUnavailable, ValueError) as exc:
            return self._error_record(image_path, zone_id, seq, str(exc))
        record = self._normalize(partial, image_path, zone_id, seq)
        if use_cache:
            self.cache.set(self._cache_key(image_path), record)
        return record

    def inspect_dir(self, images_dir: str, zone_id: str = "Z_UNKNOWN") -> list[dict]:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in exts)
        return [self.inspect(str(p), zone_id=zone_id, seq=i + 1) for i, p in enumerate(files)]

    def _normalize(self, partial: dict, image_path: str, zone_id: str, seq: int) -> dict:
        issues = []
        for j, raw_issue in enumerate(partial.get("issues", []) or []):
            itype = str(raw_issue.get("type", "other")).lower()
            if itype not in ISSUE_TYPES:
                itype = "other"
            sev = str(raw_issue.get("severity", "low")).lower()
            if sev not in SEVERITIES:
                sev = "low"
            issues.append({
                "issue_id": f"ISS_{j + 1:03d}",
                "type": itype,
                "location": str(raw_issue.get("location", "não especificado")),
                "severity": sev,
                "description": str(raw_issue.get("description", "")),
                "confidence": _clamp(raw_issue.get("confidence", 0.0), 0.0, 1.0),
                "affected_area_pct": _clamp(raw_issue.get("affected_area_pct", 0.0), 0.0, 100.0),
            })
        status = str(partial.get("overall_status", "")).lower()
        if status not in STATUSES:
            status = self._infer_status(issues)
        return {
            "inspection_id": _gen_inspection_id(seq),
            "timestamp": _now_iso(),
            "image_path": image_path,
            "zone_id": zone_id,
            "overall_status": status,
            "issues": issues,
            "shelf_fill_rate": _clamp(partial.get("shelf_fill_rate", 0.0), 0.0, 1.0),
            "products_detected": [str(p) for p in (partial.get("products_detected") or [])],
            "model_reasoning": str(partial.get("model_reasoning", "")),
            #"strategy": self.strategy,
            #"model": MODEL_NAME,
            #"_from_cache": False,
        }

    @staticmethod
    def _infer_status(issues: list[dict]) -> str:
        if any(i["severity"] == "high" for i in issues):
            return "critical"
        if issues:
            return "warning"
        return "ok"

    def _error_record(self, image_path: str, zone_id: str, seq: int, reason: str) -> dict:
        return {
            "inspection_id": _gen_inspection_id(seq),
            "timestamp": _now_iso(),
            "image_path": image_path,
            "zone_id": zone_id,
            "overall_status": "warning",
            "issues": [],
            "shelf_fill_rate": 0.0,
            "products_detected": [],
            "model_reasoning": f"[análise indisponível] {reason}",
            #"strategy": self.strategy,
            #"model": MODEL_NAME,
            "error": reason,
            #"_from_cache": False,
        }


def _clamp(value, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, v))


def save_inspection(record: dict, out_dir: str = None) -> str:
    record.pop("_from_cache", None)
    out_dir = out_dir or str(_ROOT / "data" / "inspections")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{record['inspection_id']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


STATUS_TO_TIER = {"critical": "high", "warning": "medium", "ok": "low"}

_METRICS_CANDIDATES = ["data/metrics.json", "metrics.json", "data/Projeto1/metrics.json"]


def _find_metrics(path: Optional[str] = None) -> Optional[Path]:
    if path:
        p = Path(path)
        return p if p.exists() else None
    for rel in _METRICS_CANDIDATES:
        p = _ROOT / rel
        if p.exists():
            return p
    return None


def load_metrics(path: Optional[str] = None) -> Optional[dict]:
    p = _find_metrics(path)
    if p is None:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def zone_affluence_scores(metrics: dict) -> dict:
    """
    Devolve {zona: {"pct_of_total": float, "total_visitors": int}}.
    Usa funnel.funnel_by_zone (pct_of_total) e zone_stats (total_visitors).
    """
    out: dict[str, dict] = {}
    funnel = (metrics.get("funnel", {}) or {}).get("funnel_by_zone", {}) or {}
    stats = (metrics.get("zones", {}) or {}).get("zone_stats", {}) or {}
    for z in set(funnel) | set(stats):
        pct = float(funnel.get(z, {}).get("pct_of_total", 0.0))
        visitors = int(stats.get(z, {}).get("total_visitors",
                                            funnel.get(z, {}).get("visitors", 0)))
        out[z] = {"pct_of_total": pct, "total_visitors": visitors}
    return out


def build_affluence_tiers(metrics: dict) -> dict:
    """
    Divide as zonas em três níveis (high/medium/low) por tercis de afluência
    (pct_of_total). Cada nível é uma lista ordenada por afluência decrescente.
    """
    scores = zone_affluence_scores(metrics)
    if not scores:
        return {"high": [], "medium": [], "low": []}
    ranked = sorted(scores, key=lambda z: scores[z]["pct_of_total"], reverse=True)
    n = len(ranked)
    cut = max(1, n // 3)
    return {"high": ranked[:cut], "medium": ranked[cut:2 * cut], "low": ranked[2 * cut:]}


class ZoneAssigner:


    def __init__(self, metrics: Optional[dict]) -> None:
        self.tiers = build_affluence_tiers(metrics) if metrics else {"high": [], "medium": [], "low": []}
        self.scores = zone_affluence_scores(metrics) if metrics else {}
        self._cursors = {"high": 0, "medium": 0, "low": 0}

    @property
    def available(self) -> bool:
        return any(self.tiers.values())

    def assign(self, overall_status: str) -> dict:
        """Devolve {'zone_id', 'pct_of_total', 'total_visitors', 'tier'}."""
        tier = STATUS_TO_TIER.get(overall_status, "medium")
        zones = self.tiers.get(tier) or []
        if not zones:  # nível vazio (ex.: poucas zonas) → recorrer a outro nível
            for alt in ("medium", "high", "low"):
                if self.tiers.get(alt):
                    tier, zones = alt, self.tiers[alt]
                    break
        if not zones:
            return {"zone_id": "Z_UNKNOWN", "pct_of_total": None,
                    "total_visitors": None, "tier": None}
        idx = self._cursors[tier] % len(zones)
        self._cursors[tier] += 1
        zone = zones[idx]
        sc = self.scores.get(zone, {})
        return {"zone_id": zone, "pct_of_total": sc.get("pct_of_total"),
                "total_visitors": sc.get("total_visitors"), "tier": tier}


def run_inspection_session(images_dir: Optional[str] = None,
                           metrics_path: Optional[str] = None,
                           strategy: str = "B",
                           out_dir: Optional[str] = None,
                           use_cache: bool = True,
                           also_individual: bool = False) -> Optional[str]:

    images_dir = images_dir or str(_ROOT / "data" / "images")
    out_dir = out_dir or str(_ROOT / "data" / "inspections")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(metrics_path)
    assigner = ZoneAssigner(metrics)
    if not assigner.available:
        print("⚠ metrics.json não encontrado/legível — zonas ficam Z_UNKNOWN. "
              "Usa --metrics <caminho> ou coloca o ficheiro em data/metrics.json.",
              file=sys.stderr)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = (sorted(p for p in Path(images_dir).glob("*") if p.suffix.lower() in exts)
              if Path(images_dir).is_dir() else [])
    if not images:
        print(f"⚠ Sem imagens em {images_dir}.", file=sys.stderr)
        return None

    inspector = ShelfInspector(strategy=strategy)
    records = []
    for i, img in enumerate(images):
        rec = inspector.inspect(str(img), zone_id="Z_UNKNOWN", seq=i + 1, use_cache=use_cache)
        info = assigner.assign(rec["overall_status"])  # zona ← estado + afluência
        rec["zone_id"] = info["zone_id"]

        records.append(rec)
        if also_individual:
            save_inspection(rec, out_dir=out_dir)

    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = Path(out_dir) / f"{session_id}_estrategia_{strategy}.json"

    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspeção visual de prateleiras (Componente 1)")
    ap.add_argument("--image", help="analisar uma única imagem (modo manual)")
    ap.add_argument("--images-dir", help="diretório de imagens (modo manual, zona fixa via --zone)")
    ap.add_argument("--zone", default="Z_UNKNOWN", help="zona fixa para o modo manual")
    ap.add_argument("--strategy", default="B", choices=["A", "B", "C"])
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--save", action="store_true", help="modo manual: gravar cada inspeção individual")
    ap.add_argument("--session", action="store_true",
                    help="percorre data/images, atribui zona por afluência e grava 1 ficheiro datado")
    ap.add_argument("--metrics", help="caminho do metrics.json (default: data/metrics.json)")
    ap.add_argument("--also-individual", action="store_true",
                    help="modo sessão: gravar também um ficheiro por inspeção (INS_*.json)")
    args = ap.parse_args()

    print(f"LLM: {llm_status()}", file=sys.stderr)

    # Modo sessão: por omissão (sem --image e sem --images-dir) ou com --session.
    if args.session or not (args.image or args.images_dir):
        out = run_inspection_session(
            images_dir=args.images_dir, metrics_path=args.metrics,
            strategy=args.strategy, use_cache=not args.no_cache,
            also_individual=args.also_individual)
        if out:
            print(f"Sessão concluída → {out}")
        return

    # Modo manual (compatível com a versão anterior).
    inspector = ShelfInspector(strategy=args.strategy)
    records = []
    if args.image:
        records.append(inspector.inspect(args.image, zone_id=args.zone, use_cache=not args.no_cache))
    if args.images_dir:
        records.extend(inspector.inspect_dir(args.images_dir, zone_id=args.zone))
    for rec in records:
        if args.save:
            save_inspection(rec)
        print(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
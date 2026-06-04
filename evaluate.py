"""
evaluate.py — Harness de Avaliação (Secção 9).

Executável com um único comando:
    python evaluate.py --images-dir test_images/ --output evaluation_report.json

Calcula as métricas obrigatórias de análise visual, RAG e Rule Engine, incluindo
LLM-as-judge (Hallucination Rate, Faithfulness, Answer Relevance).

Ground truth esperado em <images-dir>/ground_truth.json:
  {"img1.jpg": {"zone_id":"Z_S3","overall_status":"warning",
                "issues":[{"type":"empty_shelf","severity":"medium"}]}, ...}
Opcional <images-dir>/rag_eval.json: {"queries":[{"query","relevant_ids"}]}.

Este ficheiro é autossuficiente para o acesso ao LLM (camada embutida). Importa os
componentes (ShelfInspector, RuleEngine, RAGMemory) sem depender de llm.py.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================================== #
# Camada de acesso ao LLM (embutida) — Gemini 1.5 Flash (texto + imagem).
# Usada pelos avaliadores LLM-as-judge. Degradação graciosa.
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
_SKIP_JUDGE = False  # definido a partir de args no main()


class LLMUnavailable(RuntimeError):
    pass


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
                    print(f"[Rate Limiter] Limite de {self.max_calls} chamadas por minuto atingido. A aguardar {sleep_for:.1f}s...")
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


def extract_json(text: str) -> dict:
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
# =================== fim da camada de LLM embutida ========================= #

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from src.shelf_inspector import ShelfInspector, save_inspection  # noqa: E402
from src.rule_engine import RuleEngine  # noqa: E402
from src.rag_memory import RAGMemory  # noqa: E402

_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = _ROOT / "prompts"

def load_prompt(name: str, default: str = "") -> str:
    p = PROMPTS_DIR / f"{name}.txt"
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return default
    return default

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

'''_JUDGE_HALLUCINATION = """És um avaliador rigoroso. Dada a DESCRIÇÃO de um problema gerada por um modelo de
visão e a imagem original, decide se a descrição é verificável na imagem ou se contém
afirmações não fundamentadas (alucinação).
Responde APENAS com JSON: {"hallucinated": true/false, "justification": "..."}.

Descrição a avaliar: "%s"
"""

_JUDGE_RELEVANCE = """Avalia, de 0 a 1, se a RESPOSTA responde adequadamente à PERGUNTA do gestor de loja.
Responde APENAS com JSON: {"relevance": 0.0, "justification": "..."}.

PERGUNTA: %s
RESPOSTA: %s
"""

_JUDGE_FAITHFULNESS = """Avalia se a RESPOSTA é integralmente suportada pelo CONTEXTO fornecido (sem inventar
factos). Responde APENAS com JSON: {"faithfulness": 0.0, "justification": "..."}.

CONTEXTO:
%s

RESPOSTA: %s
"""'''


def _judge(prompt: str, image_path: Optional[str] = None) -> Optional[dict]:
    if _SKIP_JUDGE or not llm_available():
        return None
    try:
        return extract_json(generate(prompt, image_path=image_path, temperature=0.0))
    except (LLMUnavailable, ValueError, TypeError):
        return None


def evaluate_visual(images_dir: str, ground_truth: dict, strategy: str = "B") -> dict:
    inspector = ShelfInspector(strategy=strategy)
    images = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in _IMG_EXTS)
    parsed_ok = 0
    tp = fp = fn = 0
    sev_correct = sev_total = 0
    halluc_flagged = halluc_total = 0
    per_image = []
    for img in images:
        rec = inspector.inspect(str(img), zone_id=ground_truth.get(img.name, {}).get("zone_id", "Z_EVAL"))
        save_inspection(rec)
        is_parsed = "error" not in rec
        parsed_ok += int(is_parsed)
        if not is_parsed:
            per_image.append({"image": img.name, "parsed": False,
                              "error": rec.get("error")})
            continue
        gt = ground_truth.get(img.name)
        detail = {"image": img.name, "parsed": is_parsed,
                  "predicted_issue_types": [i["type"] for i in rec.get("issues", [])]}
        if gt:
            gt_types = Counter(i["type"] for i in gt.get("issues", []))
            pred_types = Counter(i["type"] for i in rec.get("issues", []))
            for t, cnt in gt_types.items():
                tp += min(cnt, pred_types.get(t, 0))
                fn += max(0, cnt - pred_types.get(t, 0))
            for t, cnt in pred_types.items():
                fp += max(0, cnt - gt_types.get(t, 0))
            gt_sev = {i["type"]: i.get("severity") for i in gt.get("issues", [])}
            for issue in rec.get("issues", []):
                if issue["type"] in gt_sev:
                    sev_total += 1
                    sev_correct += int(issue["severity"] == gt_sev[issue["type"]])
            detail["gt_issue_types"] = list(gt_types.elements())
        for issue in rec.get("issues", []):
            verdict = _judge(load_prompt("evaluate_judge_hallucination") % issue.get("description", ""), str(img))
            if verdict is not None:
                halluc_total += 1
                halluc_flagged += int(bool(verdict.get("hallucinated")))
        per_image.append(detail)
    n = len(images) or 1
    has_gt = bool(ground_truth)
    return {
        "strategy": strategy,
        "images_evaluated": len(images),
        "json_parse_rate": parsed_ok / n,
        "issue_detection_rate_recall": (tp / (tp + fn)) if (tp + fn) else (None if not has_gt else 0.0),
        "false_positive_rate": (fp / (tp + fp)) if (tp + fp) else (None if not has_gt else 0.0),
        "severity_accuracy": (sev_correct / sev_total) if sev_total else (None if not has_gt else 0.0),
        "hallucination_rate": (halluc_flagged / halluc_total) if halluc_total else None,
        "ground_truth_available": has_gt,
        "per_image": per_image,
    }

def _build_rag_eval_from_inspections(inspections_dir: Path) -> list[dict]:
    """
    Gera automaticamente um conjunto de queries de avaliação RAG a partir
    das inspeções geradas na sessão atual. Cada query é construída a partir
    dos dados reais dos inspection records (zona, issue_type, fill_rate),
    e os relevant_ids são os IDs das inspeções que contêm esses dados.

    Isto garante que o rag_eval funciona com qualquer conjunto de imagens
    fornecido pelo professor, sem depender de um ficheiro pré-definido.
    """
    records = []
    for p in sorted(inspections_dir.glob("INS_*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            if "error" in rec or not (rec.get("inspection_id") and rec.get("zone_id")):
                continue
            records.append(rec)
        except Exception:
            continue

    if not records:
        return []

    queries = []

    # Query 1 — por zona: "quais as inspeções da zona X?"
    # Para cada zona única, cria uma query e os IDs relevantes são todas as
    # inspeções dessa zona.
    zones_seen = {}
    for rec in records:
        z = rec["zone_id"]
        zones_seen.setdefault(z, []).append(rec["inspection_id"])

    for zone, ids in list(zones_seen.items())[:3]:  # máx 3 zonas
        queries.append({
            "query": f"Que inspeções foram realizadas na zona {zone}?",
            "relevant_ids": ids,
            "auto_generated": True,
        })

    # Query 2 — por tipo de issue: "quais as inspeções com prateleira vazia?"
    issue_type_map: dict[str, list[str]] = {}
    for rec in records:
        for issue in rec.get("issues", []):
            t = issue.get("type")
            if t:
                issue_type_map.setdefault(t, []).append(rec["inspection_id"])

    issue_labels = {
        "empty_shelf": "prateleira vazia",
        "misaligned": "produto desalinhado",
        "damaged": "embalagem danificada",
        "wrong_product": "produto errado",
        "label_missing": "etiqueta em falta",
        "other": "outro tipo de problema",
    }
    for issue_type, ids in list(issue_type_map.items())[:2]:  # máx 2 tipos
        label = issue_labels.get(issue_type, issue_type)
        queries.append({
            "query": f"Quais as inspeções que detetaram {label}?",
            "relevant_ids": list(set(ids)),
            "auto_generated": True,
        })

    # Query 3 — por fill rate baixo: inspeções com fill_rate < 0.75
    low_fill = [r["inspection_id"] for r in records
                if float(r.get("shelf_fill_rate", 1.0)) < 0.75]
    if low_fill:
        queries.append({
            "query": "Que inspeções tiveram fill rate abaixo de 75%?",
            "relevant_ids": low_fill,
            "auto_generated": True,
        })

    # Query 4 — por status crítico ou warning
    critical_warn = [r["inspection_id"] for r in records
                     if r.get("overall_status") in ("critical", "warning")]
    if critical_warn:
        queries.append({
            "query": "Quais as inspeções com problemas detetados (warning ou critical)?",
            "relevant_ids": critical_warn,
            "auto_generated": True,
        })

    # Query 5 — inspeção mais recente de cada zona
    for zone, ids in list(zones_seen.items())[:2]:
        queries.append({
            "query": f"Qual foi a última inspeção realizada na zona {zone}?",
            "relevant_ids": [ids[-1]],  # último ID = mais recente (ficheiros ordenados)
            "auto_generated": True,
        })

    return queries


def evaluate_rag(inspections_dir: Path) -> dict:

    if not inspections_dir.exists() or not any(inspections_dir.glob("INS_*.json")):
        return {"status": "N/A", "reason": "nenhuma inspeção disponível para avaliar o RAG"}

    eval_set = _build_rag_eval_from_inspections(inspections_dir)

    if not eval_set:
        return {"status": "N/A", "reason": "sem dados suficientes para gerar queries RAG"}

    run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}
    for chunking in ["full", "hybrid"]:
        rag = RAGMemory(chunking=chunking, collection=f"eval_{chunking}_{run_id}")
        rag.index_dir(str(inspections_dir))
        recall = rag.recall_at_k(eval_set, k=3)
        faith_scores, rel_scores = [], []
        for item in eval_set:
            res = rag.query(item["query"], k=3)
            ctx = "\n".join(c["text"] for c in res["retrieved"])
            f = _judge(load_prompt("evaluate_judge_faithfulness") % (ctx, res["answer"]))
            r = _judge(load_prompt("evaluate_judge_relevance") % (item["query"], res["answer"]))
            if f:
                faith_scores.append(float(f.get("faithfulness", 0)))
            if r:
                rel_scores.append(float(r.get("relevance", 0)))
        results[chunking] = {
            "recall_at_3": recall["recall_at_k"],
            "faithfulness": sum(faith_scores) / len(faith_scores) if faith_scores else None,
            "answer_relevance": sum(rel_scores) / len(rel_scores) if rel_scores else None,
        }
    return {"status": "ok", "by_chunking": results, "queries_evaluated": len(eval_set)}


_RULE_TEST_CASES = [
    ("Quero ser alertado quando a prateleira inferior de qualquer zona estiver mais de 30% vazia.", False),
    ("Na zona Z_S1, se não houver laticínios visíveis, é crítico e preciso de saber imediatamente.", False),
    ("Quando o fill rate de uma prateleira cair abaixo de 60% entre as 10h e as 13h, avisa-me mas não é urgente.", False),
    ("Se um produto estiver tombado, considera sempre severidade alta.", False),
    ("Avisa-me quando a prateleira estiver vazia.", True),
]

_SYNTHETIC_INSPECTION = {
    "inspection_id": "INS_SYNTH_001", "zone_id": "Z_S1",
    "timestamp": "2025-03-17T11:30:00Z", "overall_status": "warning",
    "shelf_fill_rate": 0.5,
    "issues": [{"issue_id": "ISS_001", "type": "empty_shelf", "severity": "medium",
                "location": "prateleira inferior, lado esquerdo", "description": "vazio",
                "confidence": 0.9, "affected_area_pct": 35.0}],
    "products_detected": [],
}


def evaluate_rules() -> dict:
    eng = RuleEngine()
    parse_ok = exec_ok = amb_ok = 0
    details = []
    for text, expect_ambiguous in _RULE_TEST_CASES:
        try:
            rule = eng.convert(text)
        except Exception as e:
            details.append({"rule": text[:60] + "...", "parsed": False,
                            "executed": False, "ambiguity_match": False,
                            "error": str(e)})
            continue
        parsed = rule["validation"]["is_valid"] and "conditions" in rule
        parse_ok += int(parsed)
        try:
            res = eng.evaluate(rule, _SYNTHETIC_INSPECTION)
            executed = isinstance(res.get("fired"), bool)
        except Exception:
            executed = False
        exec_ok += int(executed)
        detected_ambiguous = len(rule["validation"]["ambiguities"]) > 0
        amb_ok += int(detected_ambiguous == expect_ambiguous)
        details.append({"rule": text[:60] + "...", "parsed": parsed,
                        "executed": executed,
                        "ambiguity_match": detected_ambiguous == expect_ambiguous})
    n = len(_RULE_TEST_CASES)
    return {"cases": n, "rule_parse_rate": parse_ok / n, "rule_correctness": exec_ok / n,
            "ambiguity_detection": amb_ok / n, "details": details}


def main() -> None:
    ap = argparse.ArgumentParser(description="Harness de avaliação do sistema (Secção 9)")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--output", default="evaluation_report.json")
    ap.add_argument("--strategy", default="B", choices=["A", "B", "C"])
    ap.add_argument("--compare-strategies", action="store_true")
    ap.add_argument("--skip-judge", action="store_true",
                    help="não usar LLM-as-judge (poupa quota durante o desenvolvimento)")
    args = ap.parse_args()

    required_prompts = ["evaluate_judge_hallucination",
                        "evaluate_judge_faithfulness",
                        "evaluate_judge_relevance"]

    missing = [n for n in required_prompts if not load_prompt(n)]
    if missing and not args.skip_judge:
        print(f"⚠ prompts em falta em prompts/: {missing} — juízes desativados.",
              file=sys.stderr)
        args.skip_judge = True

    if not os.path.isdir(args.images_dir):
        print(f"⚠ diretório inexistente: {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    gt_path = Path(args.images_dir) / "ground_truth.json"
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}

    report = {
        "generated_at": _dt.datetime.now().isoformat(),
        "llm_status": llm_status(),
        "ground_truth_available": bool(ground_truth),
    }
    strategies = ["A", "B", "C"] if args.compare_strategies else [args.strategy]
    print(f"\n  A iniciar avaliação visual...")

    try:
        report["visual_analysis"] = {s: evaluate_visual(args.images_dir, ground_truth, s) for s in strategies}

    except Exception as e:
        report["visual_analysis"] = {"error": str(e)}

    inspections_dir = _ROOT / "data" / "inspections"
    print(f"\n  A iniciar avaliação RAG...")

    try:
        report["rag"] = evaluate_rag(inspections_dir)

    except Exception as e:
        report["rag"] = {"status": "error", "reason": str(e)}

    print(f"\n  A iniciar avaliação do Rule Engine...")

    try:
        report["rule_engine"] = evaluate_rules()
    except Exception as e:
        report["rule_engine"] = {"error": str(e)}

    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Avaliação concluída → {args.output}")
    va = report["visual_analysis"]
    if "error" in va:
        print(f"  [visual] erro: {va['error']}")
    else:
        for s, m in va.items():
            print(f"  [visual {s}] parse={m['json_parse_rate']:.0%} "
                  f"recall={m['issue_detection_rate_recall']} "
                  f"fpr={m['false_positive_rate']} sev={m['severity_accuracy']}")
    re_ = report["rule_engine"]
    if "error" in re_:
        print(f"  [rules] erro: {re_['error']}")
    else:
        print(f"  [rules] parse={re_['rule_parse_rate']:.0%} "
              f"correct={re_['rule_correctness']:.0%} "
              f"amb={re_['ambiguity_detection']:.0%}")


if __name__ == "__main__":
    main()
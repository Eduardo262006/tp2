"""
rule_engine.py — Componente 2: Rule Engine.

A ponte entre linguagem natural e deteção estruturada. O LLM converte regras em
português para o schema de configuração (Secção 5.3), detetando ambiguidades
(Secção 5.4). Após cada inspeção, o executor percorre as regras guardadas e gera
notificações para as que disparam, produzindo logs (Secção 5.5).

Este ficheiro é autossuficiente: inclui a sua própria camada de acesso ao Gemini.

Uso CLI:
    python rule_engine.py add "Avisa-me quando a prateleira inferior estiver mais de 40% vazia"
    python rule_engine.py list
    python rule_engine.py delete RULE_003
    python rule_engine.py test RULE_001 --inspection data/inspections/INS_xxx.json
"""

from __future__ import annotations

import argparse
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
# Conversão de texto; rate-limiting 15 req/min com backoff; degradação graciosa.
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


def load_prompt(name: str, default: str = "") -> str:
    p = PROMPTS_DIR / f"{name}.txt"
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return default
    return default
# =================== fim da camada de LLM embutida ========================= #


RULES_DIR = _ROOT / "data" / "rules"
RULES_DIR.mkdir(parents=True, exist_ok=True)

SEV_ORDER = {"low": 0, "medium": 1, "high": 2}
ISSUE_TYPES = {"empty_shelf", "wrong_product", "damaged", "misaligned", "label_missing", "other"}
LOCATION_KEYWORDS = {
    "bottom": ["inferior", "baixo", "fundo", "bottom"],
    "middle": ["meio", "central", "centro", "middle"],
    "top": ["superior", "topo", "cima", "top"],
}

'''CONVERT_PROMPT = """És um conversor de regras de negócio de retalho. O gestor de loja escreve uma regra
em português; converte-a para um objeto JSON com EXATAMENTE este schema (sem markdown,
sem texto antes ou depois):

{
  "natural_language": "<texto original da regra>",
  "description": "reformulação clara e inequívoca em português formal",
  "conditions": {
    "zone_filter": [],
    "time_filter": {"hours_start": null, "hours_end": null},
    "issue_types": [],
    "severity_threshold": null,
    "fill_rate_threshold": null,
    "location_filter": "any"
  },
  "action": {
    "alert_level": "info|warning|critical",
    "notification_message": "template com {zone_id}, {fill_rate}, {issue_type}, {severity}, {location}"
  },
  "validation": {
    "is_valid": true,
    "ambiguities": [],
    "assumptions": []
  }
}

Regras de conversão:
- zone_filter: lista de zonas (ex.: ["Z_S1"]). Vazio = todas as zonas.
- time_filter: horas inteiras 0-23; null se a regra não referir horário.
- issue_types: subconjunto de [empty_shelf, wrong_product, damaged, misaligned, label_missing, other].
- severity_threshold: low|medium|high (mínimo) ou null.
- fill_rate_threshold: fração 0-1; a regra dispara quando o fill rate fica ABAIXO deste valor.
  "X% vazia" deve ser convertido para fill_rate_threshold = 1 - X/100.
- location_filter: bottom|middle|top|any.
- ambiguities: lista TODA a informação em falta ou ambígua (ex.: "vazia" sem percentagem,
  urgência não indicada, zonas não especificadas). Se houver ambiguidades, NÃO inventes valores
  silenciosamente: regista o pressuposto assumido em assumptions e a dúvida em ambiguities.
- is_valid = false só se a regra for incompreensível.

Regra do gestor:
\"\"\"%s\"\"\"
"""'''

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RuleEngine:
    def __init__(self) -> None:
        self.rules_dir = RULES_DIR

    def convert(self, natural_language: str) -> dict:
        rule_id = self._next_rule_id()
        try:
            prompt = load_prompt("rule_engine_convert")
            if not prompt:
                raise LLMUnavailable("prompt 'rule_engine_convert.txt' em falta em prompts/")
            partial = extract_json(generate(prompt % natural_language, temperature=0.0))
            return self._normalize(partial, natural_language, rule_id)

        except (LLMUnavailable, ValueError) as e:
            raise LLMUnavailable(f"conversão da regra falhou: {e}") from e

    def _normalize(self, partial: dict, nl: str, rule_id: str) -> dict:
        cond = partial.get("conditions", {}) or {}
        tf = cond.get("time_filter", {}) or {}
        sev = cond.get("severity_threshold")
        if sev not in SEV_ORDER:
            sev = None
        loc = cond.get("location_filter", "any")
        if loc not in {"bottom", "middle", "top", "any"}:
            loc = "any"
        action = partial.get("action", {}) or {}
        alert = action.get("alert_level", "warning")
        if alert not in {"info", "warning", "critical"}:
            alert = "warning"
        validation = partial.get("validation", {}) or {}
        return {
            "rule_id": rule_id,
            "created_at": _now_iso(),
            "natural_language": nl,
            "description": str(partial.get("description", nl)),
            "conditions": {
                "zone_filter": [str(z).upper() for z in (cond.get("zone_filter") or [])],
                "time_filter": {
                    "hours_start": _opt_int(tf.get("hours_start")),
                    "hours_end": _opt_int(tf.get("hours_end")),
                },
                "issue_types": [t for t in (cond.get("issue_types") or []) if t in ISSUE_TYPES],
                "severity_threshold": sev,
                "fill_rate_threshold": _opt_float(cond.get("fill_rate_threshold")),
                "location_filter": loc,
            },
            "action": {
                "alert_level": alert,
                "notification_message": str(
                    action.get("notification_message", "Regra disparada na zona {zone_id}.")),
            },
            "validation": {
                "is_valid": bool(validation.get("is_valid", True)),
                "ambiguities": [str(a) for a in (validation.get("ambiguities") or [])],
                "assumptions": [str(a) for a in (validation.get("assumptions") or [])],
            },
        }

    def save(self, rule: dict) -> str:
        path = self.rules_dir / f"{rule['rule_id']}.json"
        path.write_text(json.dumps(rule, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def load_all(self) -> list[dict]:
        rules = []
        for p in sorted(self.rules_dir.glob("RULE_*.json")):
            try:
                rules.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
        return rules

    def get(self, rule_id: str) -> Optional[dict]:
        p = self.rules_dir / f"{rule_id}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def delete(self, rule_id: str) -> bool:
        p = self.rules_dir / f"{rule_id}.json"
        if p.exists():
            p.unlink()
            return True
        return False

    def _next_rule_id(self) -> str:
        existing = [int(p.stem.split("_")[1]) for p in self.rules_dir.glob("RULE_*.json")
                    if p.stem.split("_")[1].isdigit()]
        return f"RULE_{(max(existing) + 1) if existing else 1:03d}"

    def evaluate(self, rule: dict, inspection: dict) -> dict:
        """
        Avalia uma regra contra um inspection record.
        Zona e horário são FILTROS de elegibilidade; issue_types, severidade,
        fill_rate e localização são CONDIÇÕES de disparo. A regra dispara se passar
        os filtros E pelo menos uma condição de disparo for satisfeita.
        """
        cond = rule["conditions"]
        log: list[str] = []
        reasons: list[str] = []

        zone = inspection.get("zone_id", "")
        if cond["zone_filter"] and zone not in cond["zone_filter"]:
            log.append(f"zona {zone} fora de {cond['zone_filter']} → não elegível")
            return self._no_fire(rule, log)
        log.append(f"zona {zone}: elegível")

        hs, he = cond["time_filter"]["hours_start"], cond["time_filter"]["hours_end"]
        if hs is not None and he is not None:
            hour = _hour_of(inspection.get("timestamp"))
            if hour is None or not (hs <= hour < he):
                log.append(f"hora {hour} fora de [{hs},{he}) → não elegível")
                return self._no_fire(rule, log)
            log.append(f"hora {hour} dentro de [{hs},{he})")

        fired_fill = False
        if cond["fill_rate_threshold"] is not None:
            fr = float(inspection.get("shelf_fill_rate", 0.0))
            if fr < cond["fill_rate_threshold"]:
                fired_fill = True
                reasons.append(f"fill_rate {fr:.2f} < {cond['fill_rate_threshold']:.2f}")
            log.append(f"fill_rate {fr:.2f} vs limiar {cond['fill_rate_threshold']:.2f}")

        fired_issue = False
        matched_issue = None
        if cond["issue_types"] or cond["severity_threshold"] or cond["location_filter"] != "any":
            for issue in inspection.get("issues", []):
                if cond["issue_types"] and issue["type"] not in cond["issue_types"]:
                    continue
                if cond["severity_threshold"]:
                    if SEV_ORDER.get(issue["severity"], 0) < SEV_ORDER[cond["severity_threshold"]]:
                        continue
                if cond["location_filter"] != "any":
                    if not _location_matches(issue.get("location", ""), cond["location_filter"]):
                        continue
                fired_issue = True
                matched_issue = issue
                reasons.append(
                    f"issue '{issue['type']}' sev={issue['severity']} @ {issue.get('location','?')}")
                break
            log.append(f"correspondência de issue: {'sim' if fired_issue else 'não'}")

        only_filters = not (cond["fill_rate_threshold"] is not None
                            or cond["issue_types"] or cond["severity_threshold"]
                            or cond["location_filter"] != "any")
        if only_filters and inspection.get("issues"):
            fired_issue = True
            matched_issue = inspection["issues"][0]
            reasons.append("regra sem condições específicas; disparou por existir issue")

        fired = fired_fill or fired_issue
        if not fired:
            log.append("nenhuma condição de disparo satisfeita")
            return self._no_fire(rule, log)

        notification = self._render(rule, inspection, matched_issue)
        log.append(f"DISPAROU → {rule['action']['alert_level']}")
        return {"rule_id": rule["rule_id"], "fired": True,
                "alert_level": rule["action"]["alert_level"], "reasons": reasons,
                "notification": notification, "log": log}

    def evaluate_all(self, inspection: dict, rules: Optional[list[dict]] = None) -> list[dict]:
        rules = rules if rules is not None else self.load_all()
        return [self.evaluate(r, inspection) for r in rules]

    def _no_fire(self, rule: dict, log: list[str]) -> dict:
        return {"rule_id": rule["rule_id"], "fired": False, "alert_level": None,
                "reasons": [], "notification": None, "log": log}

    @staticmethod
    def _render(rule: dict, inspection: dict, issue: Optional[dict]) -> str:
        tmpl = rule["action"]["notification_message"]
        ctx = {
            "zone_id": inspection.get("zone_id", "?"),
            "fill_rate": f"{float(inspection.get('shelf_fill_rate', 0.0)):.0%}",
            "issue_type": issue["type"] if issue else "—",
            "severity": issue["severity"] if issue else "—",
            "location": issue.get("location", "—") if issue else "—",
            "timestamp": inspection.get("timestamp", "?"),
        }
        try:
            return tmpl.format(**ctx)
        except Exception:
            return tmpl


def _opt_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _opt_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _hour_of(ts: Optional[str]) -> Optional[int]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).hour
    except Exception:
        return None


def _location_matches(text: str, loc_filter: str) -> bool:
    return any(w in text.lower() for w in LOCATION_KEYWORDS.get(loc_filter, []))


def main() -> None:
    ap = argparse.ArgumentParser(description="Rule Engine (Componente 2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_add = sub.add_parser("add"); p_add.add_argument("text")
    sub.add_parser("list")
    p_del = sub.add_parser("delete"); p_del.add_argument("rule_id")
    p_test = sub.add_parser("test")
    p_test.add_argument("rule_id")
    p_test.add_argument("--inspection", required=True)
    args = ap.parse_args()

    eng = RuleEngine()
    if args.cmd == "add":
        try:
            rule = eng.convert(args.text)
        except LLMUnavailable as e:
            print("⚠ Falha ao comunicar com o modelo de IA.", file=sys.stderr)
            print("Verifica a ligação à internet e a GEMINI_API_KEY no .env.", file=sys.stderr)
            print(f"Detalhes técnicos: {e}", file=sys.stderr)
            sys.exit(1)
        eng.save(rule)
        print(json.dumps(rule, ensure_ascii=False, indent=2))
        if rule["validation"]["ambiguities"]:
            print("\n⚠ Ambiguidades detetadas (clarificar antes de confiar na regra):", file=sys.stderr)
            for a in rule["validation"]["ambiguities"]:
                print(f"  - {a}", file=sys.stderr)
    elif args.cmd == "list":
        for r in eng.load_all():
            print(f"{r['rule_id']}: {r['description']}")
    elif args.cmd == "delete":
        print("removida" if eng.delete(args.rule_id) else "não encontrada")
    elif args.cmd == "test":
        rule = eng.get(args.rule_id)
        if not rule:
            print("regra não encontrada"); return
        insp = json.loads(Path(args.inspection).read_text(encoding="utf-8"))
        print(json.dumps(eng.evaluate(rule, insp), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
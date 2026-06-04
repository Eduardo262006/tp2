"""
report_generator.py — Componente 4: Report Generator.

Gera o Inspection Report em Markdown com as secções obrigatórias do enunciado
(Secção 7): sumário executivo, problemas por zona, regras disparadas, contexto
histórico (RAG), recomendações e (opcional) integração com trajetória.

Este ficheiro é autossuficiente para o acesso ao LLM (camada embutida). Importa o
RAGMemory e o RuleEngine apenas como componentes (não depende de llm.py).

Uso CLI:
    python report_generator.py --inspections-dir data/inspections --out report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from shelf_inspector import load_metrics, zone_affluence_scores
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================================== #
# Camada de acesso ao LLM (embutida) — Gemini 3.5 Flash (texto).
# Usada para sumário executivo e recomendações; degradação graciosa.
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

MODEL_NAME = os.getenv("MODEL", "gemini-3.5-flash")
_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "prompts").is_dir():
            return parent
    return here.parent


_ROOT = _project_root()
PROMPTS_DIR = _ROOT / "prompts"

def load_prompt(name: str, default: str = "") -> str:
    p = PROMPTS_DIR / f"{name}.txt"
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return default
    return default


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


def generate(prompt: str, temperature: float = 0.0, max_retries: int = 4) -> str:
    if not llm_available():
        raise LLMUnavailable(llm_status()["reason"])
    delay = 2.0
    for attempt in range(max_retries):
        _RATE.acquire()
        try:
            resp = _STATE.model.generate_content(
                [prompt], generation_config={"temperature": temperature})
            return (resp.text or "").strip()
        except Exception as exc:
            if _is_quota_error(exc):
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                _STATE.quota_exhausted = True
                raise LLMUnavailable("quota esgotada (429)") from exc
            if attempt < max_retries - 1:
                time.sleep(1.0)
                continue
            raise LLMUnavailable(f"erro do modelo: {exc}") from exc
    raise LLMUnavailable("esgotadas as tentativas de chamada ao modelo")
# =================== fim da camada de LLM embutida ========================= #

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from rag_memory import RAGMemory
except Exception:
    RAGMemory = None
try:
    from rule_engine import RuleEngine
except Exception:
    RuleEngine = None


'''_SUMMARY_PROMPT = """Escreve um sumário executivo (MÁXIMO 150 palavras), em português, direto e acionável,
sobre o estado geral da loja nesta sessão de inspeção. Indica quantas zonas foram
inspecionadas, quantos issues críticos e quantos warnings. Sem listas, só prosa.

Dados da sessão (JSON):
%s
"""

_RECS_PROMPT = """Com base nesta sessão de inspeção, escreve NO MÁXIMO 5 recomendações concretas e
acionáveis, ordenadas por urgência (a mais urgente primeiro). Cada recomendação deve ser
específica o suficiente para ser executada sem interpretação adicional. Devolve uma
recomendação por linha, prefixada por "1. ", "2. ", etc.

Dados da sessão (JSON):
%s
"""'''


class ReportGenerator:
    def __init__(self, rag: Optional["RAGMemory"] = None) -> None:
        self.rag = rag

    def generate(self, inspections: list[dict], fired_rules: Optional[list[dict]] = None,
                 trajectory_context: Optional[dict] = None) -> str:
        fired_rules = fired_rules or []
        stats = self._stats(inspections)
        parts = [
            f"# Inspection Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            self._section_summary(inspections, stats),
            self._section_by_zone(inspections),
            self._section_rules(fired_rules),
            self._section_history(inspections),
            self._section_recommendations(inspections, stats),
            self._section_trajectory(inspections),
        ]
        return "\n\n".join(p for p in parts if p)

    def _stats(self, inspections: list[dict]) -> dict:
        zones = {i.get("zone_id", "?") for i in inspections}
        crit = sum(1 for i in inspections if i.get("overall_status") == "critical")
        warn = sum(1 for i in inspections if i.get("overall_status") == "warning")
        total_issues = sum(len(i.get("issues", [])) for i in inspections)
        return {"zones": len(zones), "critical": crit, "warning": warn,
                "ok": len(inspections) - crit - warn, "inspections": len(inspections),
                "issues": total_issues}

    def _section_summary(self, inspections: list[dict], stats: dict) -> str:
        body = None
        if llm_available():
            try:
                payload = json.dumps({"stats": stats, "inspections": _slim(inspections)},
                                     ensure_ascii=False)
                body = generate(load_prompt("report_generator_summary_prompt") % payload, temperature=0.0).strip()
            except LLMUnavailable:
                body = None
        if not body:
            body = (f"Foram inspecionadas {stats['zones']} zonas em {stats['inspections']} "
                    f"imagens. Detetaram-se {stats['critical']} situações críticas e "
                    f"{stats['warning']} avisos, num total de {stats['issues']} problemas. "
                    + ("Atenção imediata recomendada às zonas com estado crítico."
                       if stats['critical'] else "Sem situações críticas nesta sessão."))
        return f"## 1. Sumário Executivo\n\n{body}"

    def _section_by_zone(self, inspections: list[dict]) -> str:
        lines = ["## 2. Problemas por Zona", ""]
        by_zone: dict[str, list[dict]] = {}
        for ins in inspections:
            if ins.get("issues"):
                by_zone.setdefault(ins.get("zone_id", "?"), []).append(ins)
        if not by_zone:
            lines.append("_Nenhuma zona apresentou problemas nesta sessão._")
            return "\n".join(lines)
        for zone, inss in sorted(by_zone.items()):
            lines.append(f"### Zona {zone}")
            for ins in inss:
                lines.append(f"- **{ins['inspection_id']}** — estado `{ins['overall_status']}`, "
                             f"fill rate {float(ins.get('shelf_fill_rate',0)):.0%}")
                for issue in ins["issues"]:
                    lines.append(f"  - `{issue['type']}` ({issue['severity']}) em "
                                 f"{issue.get('location','?')}: {issue.get('description','')}")
                hist = self._zone_history(zone)
                if hist:
                    lines.append(f"  - _Histórico (RAG): {hist}_")
            lines.append("")
        return "\n".join(lines)

    def _section_rules(self, fired_rules: list[dict]) -> str:
        lines = ["## 3. Regras Disparadas", ""]
        fired = [r for r in fired_rules if r.get("fired")]
        if not fired:
            lines.append("_Nenhuma regra disparou nesta sessão._")
            return "\n".join(lines)
        for r in fired:
            lines.append(f"- **{r['rule_id']}** [`{r['alert_level']}`]: {r['notification']}")
            if r.get("reasons"):
                lines.append(f"  - Motivos: {'; '.join(r['reasons'])}")
        return "\n".join(lines)

    def _section_history(self, inspections: list[dict]) -> str:
        lines = ["## 4. Contexto Histórico Relevante", ""]
        if self.rag is None:
            lines.append("_RAG indisponível nesta sessão._")
            return "\n".join(lines)

        # peso por zona: issues contam, severidade alta conta mais
        peso = {"low": 1, "medium": 2, "high": 4}
        score: dict[str, int] = {}
        for rec in inspections:
            zone = rec.get("zone_id", "?")
            for issue in rec.get("issues", []):
                score[zone] = score.get(zone, 0) + peso.get(issue.get("severity"), 1)

        if not score:
            lines.append("_Sem problemas que justifiquem contexto histórico._")
            return "\n".join(lines)

        top_zones = [z for z, _ in sorted(score.items(), key=lambda kv: -kv[1])[:5]]

        for zone in sorted(top_zones):
            res = self.rag.query(f"Que problemas já foram detetados na zona {zone}?", k=3)
            refs = ", ".join(c["metadata"].get("inspection_id", "?")
                             for c in res["retrieved"])
            lines.append(f"- **{zone}**: {res['answer']}")
            if refs:
                lines.append(f"  - Inspeções referidas: {refs}")

        omitidas = len(score) - len(top_zones)
        if omitidas > 0:
            lines.append(f"\n_Contexto histórico limitado às {len(top_zones)} zonas "
                         f"mais críticas ({omitidas} zonas com problemas menores omitidas)._")
        return "\n".join(lines)

    def _section_recommendations(self, inspections: list[dict], stats: dict) -> str:
        body = None
        if llm_available() and stats["issues"]:
            try:
                payload = json.dumps(_slim(inspections), ensure_ascii=False)
                body = generate(load_prompt("report_generator_recomendation_prompt") % payload, temperature=0.0).strip()
            except LLMUnavailable:
                body = None
        if not body:
            recs = []
            crit = [i for i in inspections if i.get("overall_status") == "critical"]
            for ins in crit[:3]:
                recs.append(f"Intervir na zona {ins['zone_id']} (situação crítica em {ins['inspection_id']}).")
            empties = [i for i in inspections
                       if any(x["type"] == "empty_shelf" for x in i.get("issues", []))]
            for ins in empties[:2]:
                recs.append(f"Repor stock na zona {ins['zone_id']} (prateleira vazia detetada).")
            if not recs:
                recs.append("Manter rotina normal de reposição; sem ações urgentes.")
            body = "\n".join(f"{i+1}. {r}" for i, r in enumerate(recs[:5]))
        return f"## 5. Recomendações\n\n{body}"

    def _section_trajectory(self, inspections: list[dict]) -> str:
        metrics = load_metrics(None)
        if not metrics:
            return "## 6. Integração com Trajetória\n\n_metrics.json indisponível._\n"
        scores = zone_affluence_scores(metrics)

        por_zona: dict[str, dict] = {}
        for rec in inspections:
            z = rec.get("zone_id", "?")
            d = por_zona.setdefault(z, {"issues": 0, "empty": 0, "fill": []})
            d["issues"] += len(rec.get("issues", []))
            d["empty"] += sum(1 for i in rec.get("issues", [])
                              if i.get("type") == "empty_shelf")
            d["fill"].append(float(rec.get("shelf_fill_rate", 0.0)))

        linhas = ["## 6. Integração com Trajetória (Projeto 1)", ""]
        factos = []
        for z, d in por_zona.items():
            m = scores.get(z)
            if not m:
                continue
            afl = m.get("pct_of_total")
            fill_med = sum(d["fill"]) / max(len(d["fill"]), 1)
            factos.append({"zona": z, "afluencia_pct": afl,
                           "issues": d["issues"], "empty_shelf": d["empty"],
                           "fill_medio": round(fill_med, 2)})
            linhas.append(f"- **{z}** — afluência {afl:.1%} do total | "
                          f"{d['issues']} issues ({d['empty']} empty_shelf) | "
                          f"fill médio {fill_med:.0%}")

        if not factos:
            linhas.append("_Sem zonas em comum entre as inspeções e o metrics.json._")
            return "\n".join(linhas) + "\n"

        # 1 chamada LLM para tirar conclusões; fallback heurístico sem quota
        try:
            prompt = load_prompt("report_trajectory")
            if prompt:
                analise = generate(prompt % json.dumps(factos, ensure_ascii=False),
                                   temperature=0.0)
                linhas += ["", "**Análise:**", analise]
        except Exception:
            altas = [f["zona"] for f in factos
                     if (f["afluencia_pct"] or 0) > 0.05 and f["empty_shelf"] > 0]
            if altas:
                linhas += ["", f"**Análise (heurística):** as zonas {', '.join(altas)} "
                           "combinam afluência elevada com prateleiras vazias, sugerindo "
                           "rutura por procura elevada e não falha de reposição."]
        return "\n".join(linhas) + "\n"

    def _zone_history(self, zone: str) -> Optional[str]:
        if self.rag is None:
            return None
        try:
            res = self.rag.retrieve(f"problemas anteriores na zona {zone}", k=2)
            if res:
                return "; ".join(c["metadata"].get("inspection_id", "?") for c in res)
        except Exception:
            return None
        return None


def _slim(inspections: list[dict]) -> list[dict]:
    return [{
        "id": i.get("inspection_id"),
        "zone": i.get("zone_id"),
        "status": i.get("overall_status"),
        "fill_rate": i.get("shelf_fill_rate"),
        "issues": [{"type": x["type"], "severity": x["severity"], "loc": x.get("location")}
                   for x in i.get("issues", [])],
    } for i in inspections]


def load_inspections(inspections_dir: str) -> list[dict]:
    out = []
    for p in sorted(Path(inspections_dir).glob("INS_*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Report Generator (Componente 4)")
    ap.add_argument("--inspections-dir", default=str(_ROOT / "data" / "inspections"))
    ap.add_argument("--out", default="inspection_report.md")
    ap.add_argument("--no-rag", action="store_true")
    args = ap.parse_args()

    inspections = load_inspections(args.inspections_dir)
    if not inspections:
        print("Sem inspeções para reportar.", file=sys.stderr)
        return

    rag = None
    if not args.no_rag and RAGMemory is not None:
        try:
            rag = RAGMemory(chunking="hybrid")
        except Exception:
            rag = None

    fired = []
    if RuleEngine is not None:
        eng = RuleEngine()
        rules = eng.load_all()
        for ins in inspections:
            fired.extend(eng.evaluate_all(ins, rules))

    md = ReportGenerator(rag=rag).generate(inspections, fired_rules=fired)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"Relatório escrito em {args.out}")


if __name__ == "__main__":
    main()
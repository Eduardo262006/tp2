"""
interface.py — Componente 5: Interface Conversacional (CLI).

Orquestra os restantes componentes e expõe os modos de operação do enunciado
(Secção 8): inspeção, definição de regras, consulta histórica e relatório.
Mantém estado de sessão e responde de forma informativa a comandos inválidos —
nunca expõe stack traces.

Não depende de llm.py: o estado do LLM é obtido a partir do shelf_inspector, que já
é importado como componente.

Uso:
    python interface.py            # modo interativo (REPL)
    python interface.py --help-cmds
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shelf_inspector import ShelfInspector, save_inspection, llm_status, ZoneAssigner, load_metrics  # noqa: E402
from rule_engine import RuleEngine  # noqa: E402
from rag_memory import RAGMemory  # noqa: E402
from report_generator import ReportGenerator  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent


class Session:
    """Estado de sessão partilhado entre comandos."""

    def __init__(self) -> None:
        self.inspector = ShelfInspector(strategy="B")
        self.rules = RuleEngine()
        self.assigner = ZoneAssigner(load_metrics(None))
        try:
            self.rag = RAGMemory(chunking="hybrid")
        except Exception:
            self.rag = None
        self.reporter = ReportGenerator(rag=self.rag)
        self.inspections: list[dict] = []
        self.fired: list[dict] = []

    def record(self, inspection: dict) -> None:
        self.inspections.append(inspection)
        save_inspection(inspection)
        if self.rag is not None:
            try:
                self.rag.index_inspection(inspection)
            except Exception:
                pass
        for res in self.rules.evaluate_all(inspection):
            if res.get("fired"):
                self.fired.append(res)


HELP = """Comandos:
  inspect <ZONE> --image <ficheiro> [--strategy A|B|C]
  inspect all --images-dir <dir> [--zone <ZONE>]
  add rule "<texto>"            
  list rules            
  delete rule <RULE_ID>
  test rule <RULE_ID> --image <ficheiro> [--zone <ZONE>]
  history "<pergunta>"          
  compare <Z_A> <Z_B> [--period "..."]
  report [--session today]      
  status   
  help   
  exit"""


def _parse_opts(tokens: list[str]) -> tuple[list[str], dict]:
    pos, opts, i = [], {}, 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            key = t[2:]
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                opts[key] = tokens[i + 1]; i += 2
            else:
                opts[key] = True; i += 1
        else:
            pos.append(t); i += 1
    return pos, opts


class Interface:
    def __init__(self) -> None:
        self.s = Session()

    def run(self) -> None:
        print("Retail Vision Intelligence System — interface conversacional")
        #print(f"[{llm_status()['reason']} | RAG backend: "f"{self.s.rag.backend if self.s.rag else 'indisponível'}]")
        print("Escreva 'help' para ver os comandos, 'exit' para sair.\n")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAté breve."); break
            if not line:
                continue
            if line in {"exit", "quit"}:
                print("Até breve."); break
            try:
                self.dispatch(line)
            except Exception as exc:  # nunca expor stack trace ao utilizador
                print(f"⚠ Não consegui executar esse comando: {exc}")

    def dispatch(self, line: str) -> None:
        try:
            tokens = shlex.split(line)
        except ValueError:
            print("⚠ Aspas mal fechadas no comando."); return
        cmd = tokens[0].lower()
        if cmd == "help":
            print(HELP)
        elif cmd == "status":
            self.cmd_status()
        elif cmd == "inspect":
            self.cmd_inspect(tokens[1:])
        elif cmd == "add" and len(tokens) > 1 and tokens[1] == "rule":
            self.cmd_add_rule(tokens[2:])
        elif cmd == "list" and len(tokens) > 1 and tokens[1] == "rules":
            self.cmd_list_rules()
        elif cmd == "delete" and len(tokens) > 1 and tokens[1] == "rule":
            self.cmd_delete_rule(tokens[2:])
        elif cmd == "test" and len(tokens) > 1 and tokens[1] == "rule":
            self.cmd_test_rule(tokens[2:])
        elif cmd == "history":
            self.cmd_history(tokens[1:])
        elif cmd == "compare":
            self.cmd_compare(tokens[1:])
        elif cmd == "report":
            self.cmd_report(tokens[1:])
        else:
            print(f"⚠ Comando desconhecido: '{line}'. Escreve 'help'.")

    def cmd_status(self) -> None:
        print(json.dumps({
            "llm": llm_status(),
            "rag_backend": self.s.rag.backend if self.s.rag else "indisponível",
            "inspecoes_sessao": len(self.s.inspections),
            "regras_guardadas": len(self.s.rules.load_all()),
            "regras_disparadas_sessao": len(self.s.fired),
        }, ensure_ascii=False, indent=2))

    def cmd_inspect(self, args: list[str]) -> None:
        pos, opts = _parse_opts(args)
        if not pos and not opts.get("image") and not opts.get("images-dir"):
            print("Uso: inspect [ZONE] --image <f> | inspect all --images-dir <dir>");
            return
        if opts.get("strategy"):
            self.s.inspector = ShelfInspector(strategy=str(opts["strategy"]).upper())
        if pos and pos[0].lower() == "all":
            d = opts.get("images-dir")
            zone_opt = opts.get("zone")
            incluir_cache = bool(opts.get("include-cached"))
            if not d or not os.path.isdir(str(d)):
                print("⚠ Indica --images-dir <diretório válido>.");
                return
            recs = self.s.inspector.inspect_dir(str(d), zone_id=str(zone_opt or "Z_UNKNOWN"))

            novas = []
            for r in recs:
                if not incluir_cache and r.get("_from_cache"):
                    continue
                if not zone_opt and self.s.assigner.available:
                    r["zone_id"] = self.s.assigner.assign(r["overall_status"])["zone_id"]
                self.s.record(r)
                novas.append(r)

            if novas:
                sess = _ROOT / "data" / "inspections" / (
                    f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                    f"_estrategia_{self.s.inspector.strategy}.json")
                sess.write_text(json.dumps(novas, ensure_ascii=False, indent=2),
                                encoding="utf-8")
                print(f"✓ {len(novas)} imagens novas inspecionadas "
                      f"({len(recs) - len(novas)} já analisadas anteriormente). "
                      f"Sessão: {sess.name}")
                self._print_summary(novas)
            else:
                print(f"✓ Nada de novo: as {len(recs)} imagens da pasta "
                      "já foram todas analisadas.")
        else:
            zone = pos[0] if pos else "Z_UNKNOWN"
            img = opts.get("image")
            if not img or not os.path.exists(str(img)):
                print("⚠ Indica --image <ficheiro existente>."); return
            rec = self.s.inspector.inspect(str(img), zone_id=zone)
            self._print_inspection(rec)
            self.s.record(rec)

    def cmd_add_rule(self, args: list[str]) -> None:
        if not args:
            print('Uso: add rule "<texto da regra>"'); return
        text = " ".join(args)
        rule = self.s.rules.convert(text)
        amb = rule["validation"]["ambiguities"]
        if amb:
            print(f"Regra interpretada como: {rule['description']}")
            print("⚠ Antes de guardar, ajuda-me a clarificar:")
            for a in amb:
                print(f"   - {a}")
            print("(A regra foi guardada na mesma; podes ajustá-la e voltar a adicionar.)")
        self.s.rules.save(rule)
        print(f"✓ Regra guardada como {rule['rule_id']} [{rule['action']['alert_level']}].")

    def cmd_list_rules(self) -> None:
        rules = self.s.rules.load_all()
        if not rules:
            print("(sem regras guardadas)"); return
        for r in rules:
            print(f"  {r['rule_id']} [{r['action']['alert_level']}]: {r['description']}")

    def cmd_delete_rule(self, args: list[str]) -> None:
        if not args:
            print("Uso: delete rule <RULE_ID>"); return
        print("✓ removida" if self.s.rules.delete(args[0]) else "⚠ regra não encontrada")

    def cmd_test_rule(self, args: list[str]) -> None:
        pos, opts = _parse_opts(args)
        if not pos:
            print("Uso: test rule <RULE_ID> --image <f> [--zone <Z>]"); return
        rule = self.s.rules.get(pos[0])
        if not rule:
            print("⚠ regra não encontrada"); return
        img = opts.get("image")
        if not img or not os.path.exists(str(img)):
            print("⚠ Indica --image <ficheiro existente>."); return
        rec = self.s.inspector.inspect(str(img), zone_id=str(opts.get("zone", "Z_TEST")))
        res = self.s.rules.evaluate(rule, rec)
        print(f"Regra {'DISPAROU' if res['fired'] else 'não disparou'}.")
        if res["fired"]:
            print(f"  → {res['notification']}")
        print("  log:")
        for l in res["log"]:
            print(f"    · {l}")

    def cmd_history(self, args: list[str]) -> None:
        if not args:
            print('Uso: history "<pergunta>"'); return
        if self.s.rag is None:
            print("⚠ RAG indisponível."); return
        res = self.s.rag.query(" ".join(args), k=3)
        print(res["answer"])
        if res["retrieved"]:
            print("  fontes: " + ", ".join(
                c["metadata"].get("inspection_id", "?") for c in res["retrieved"]))

    def cmd_compare(self, args: list[str]) -> None:
        pos, opts = _parse_opts(args)
        if len(pos) < 2 or self.s.rag is None:
            print("Uso: compare <Z_A> <Z_B> [--period \"...\"]"); return
        for zone in pos[:2]:
            res = self.s.rag.query(f"Resumo dos problemas detetados na zona {zone}", k=3)
            print(f"### {zone}\n{res['answer']}\n")

    def cmd_report(self, args: list[str]) -> None:
        _, opts = _parse_opts(args)
        ins_dir = _ROOT / "data" / "inspections"
        inspections = []
        origem = "sessão atual da interface"

        sess = opts.get("session")
        if sess and str(sess) not in ("today", "True", "true"):
            # sessão específica: aceita o nome com ou sem .json
            name = str(sess)
            candidatos = [ins_dir / name, ins_dir / f"{name}.json"]
            f = next((c for c in candidatos if c.exists()), None)
            if f is None:
                # aceita também só o prefixo da data: report --session 2026-06-10
                matches = sorted(ins_dir.glob(f"{name}*.json"))
                f = matches[-1] if matches else None
            if f is None:
                print(f"⚠ Sessão '{name}' não encontrada em {ins_dir}."); return
            inspections = json.loads(f.read_text(encoding="utf-8"))
            origem = f.name
        else:
            inspections = list(self.s.inspections)
            if not inspections:
                today = datetime.now().strftime("%Y-%m-%d")
                files = sorted(ins_dir.glob(f"{today}_*.json"))
                if files:
                    inspections = json.loads(files[-1].read_text(encoding="utf-8"))
                    origem = files[-1].name
            if not inspections:
                print("⚠ Ainda não há inspeções nesta sessão "
                      "(e não existe ficheiro de sessão de hoje)."); return

        print(f"(a reportar sobre: {origem}, {len(inspections)} inspeções)")

        fired = list(self.s.fired)
        if origem != "sessão atual da interface":
            fired = []
            for rec in inspections:
                for res in self.s.rules.evaluate_all(rec):
                    if res.get("fired"):
                        fired.append(res)

        md = self.s.reporter.generate(inspections, fired_rules=fired)
        if not md:
            print("⚠ O gerador de relatório não devolveu conteúdo.");
            return
        out = _ROOT / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        out.write_text(md, encoding="utf-8")
        print(f"✓ Relatório gerado: {out}")
        #print("\n" + md[:600] + ("\n..." if len(md) > 600 else ""))

    def _print_inspection(self, rec: dict) -> None:
        cache = " (cache)" if rec.get("_from_cache") else ""
        print(f"✓ {rec['inspection_id']} | zona {rec['zone_id']} | "
              f"estado {rec['overall_status']} | fill {float(rec['shelf_fill_rate']):.0%}{cache}")
        for issue in rec["issues"]:
            print(f"   - {issue['type']} ({issue['severity']}) @ {issue.get('location','?')}")

    def _print_summary(self, recs: list[dict]) -> None:
        crit = sum(1 for r in recs if r["overall_status"] == "critical")
        warn = sum(1 for r in recs if r["overall_status"] == "warning")
        print(f"   resumo: {crit} críticas, {warn} avisos, {len(recs) - crit - warn} ok")


def main() -> None:
    if "--help-cmds" in sys.argv:
        print(HELP); return
    Interface().run()


if __name__ == "__main__":
    main()
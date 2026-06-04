# Retail Vision Intelligence System — TP2
 
Sistema de inspeção visual de prateleiras de retalho com LLM multimodal (Gemini),
motor de regras em linguagem natural, memória semântica (RAG) e relatórios automáticos.
 
## Instalação
 
```bash
pip install -r requirements.txt
```
 
Criar o ficheiro `.env` na raiz:
 
```
GEMINI_API_KEY=a_tua_chave_do_ai_studio
MODEL=gemini-3.1-flash-lite
```
 
A chave obtém-se gratuitamente em https://aistudio.google.com .
 
## Interface conversacional
 
```bash
python src/interface.py
```
 
Comandos principais:
 
```
inspect all --images-dir <dir>           Inspeciona uma pasta. Modo incremental:
                                         só processa imagens novas (nunca analisadas).
                                       
add rule "<texto>"                       Converte uma regra em português para JSON
                                         e guarda-a 
                                         
list rules                               Lista as regras guardadas

delete rule <RULE_ID>                    Remove uma regra

report --session today                   Gera o relatório Markdown da sessão do dia

report --session <nome>                  Gera o relatório Markdown de um json especificado

status                                   Estado do LLM, RAG, sessão e regras

exit                                     Sair
```
 
Exemplo de sessão típica:
 
```
> inspect all --images-dir data/images
> add rule "Avisa-me quando o fill rate estiver abaixo de 80%"
> report --session today
```
 
Notas:
- Imagens já analisadas vêm do cache (`cache/`) sem gastar quota de API.
- O relatório é gravado na raiz como `report_AAAAMMDD_HHMMSS.md`.
- `report --session 2026-06-09` reporta a sessão desse dia.
## Avaliação (harness)
 
Comando único, como definido no enunciado:
 
```bash
python evaluate.py --images-dir test_images/ --output evaluation_report.json
```
 
Requer `ground_truth.json` com as anotações das imagens de teste.
 
## Estrutura do projeto
 
```
tp2/
├── data/images/        dataset de imagens
├── data/inspections/   inspection records (INS_*.json) e sessões datadas
├── data/rules/         regras persistidas
├── prompts/            todos os prompts versionados (*.txt)
├── cache/              cache de API (inspeções, summaries, vereditos do juiz)
├── vectorstore/        ChromaDB persistente (gerado em runtime)
├── src/                os 5 componentes do sistema
└── evaluate.py         harness de avaliação
```
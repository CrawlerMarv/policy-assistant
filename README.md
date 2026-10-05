# Company Policy Assistant

Drei Ansätze beantworten Policy-Fragen auf Basis von `data/company_policies.csv` (98 Policies):

| Key | Ansatz | Datei |
|---|---|---|
| `rules` | Regelbasierte Suche (BM25), kein LLM | `policy_assistant/rules.py` |
| `llm_full` | LLM ohne Vektorindex – alle Policies im Prompt | `policy_assistant/full_context.py` |
| `llm_rag` | LLM mit Vektorindex – Top-3 Policies per Embedding-Suche | `policy_assistant/rag.py` |
| `llm_closed_book` | optional: LLM ganz ohne Policy-Daten | `policy_assistant/full_context.py` |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Anbieter wählen und API-Key eintragen
```

### LLM-Anbieter

In `.env` über `LLM_PROVIDER` wählbar:

- `gemini` (Standard): kostenloses Kontingent über Google AI Studio, Key unter aistudio.google.com → „Get API key“. Standardmodell ist `gemini-3.5-flash-lite`: Im Gratis-Tier haben die Flash-Lite-Modelle deutlich höhere Tageslimits als die Flash-Modelle. `evaluate.py` pausiert 7 s zwischen LLM-Aufrufen und setzt abgebrochene Läufe beim nächsten Start fort.
- `anthropic`: Key in der Claude Console unter Settings → API keys, benötigt Guthaben.

## Nutzung

```bash
python -m policy_assistant "Can I work from home on Fridays?"
python evaluate.py                     # alle Testfragen → results/results.csv + summary.csv
python evaluate.py -a rules            # ohne API-Key
```

Im Code (Website, Slack-Bot):

```python
from policy_assistant import ask, warm_up
warm_up()                                # Indizes einmal beim Start bauen
r = ask("llm_rag", "How many vacation days do I get?")
r.answer, r.policy_title, r.seconds, r.total_tokens
```

## Website

```bash
streamlit run app.py
```

Testfragen zeigen die gespeicherten Ergebnisse aus `results/` (kein API-Aufruf), eigene Fragen werden live von allen drei Ansätzen beantwortet und gecacht. Der Vergleichstext steht in `comparison.md`.

Deployment auf Streamlit Community Cloud: Repo verbinden, `app.py` als Main file, unter „Secrets“ eintragen:

```toml
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = "..."
REPO_URL = "https://github.com/<user>/<repo>"
```

Der Ordner `results/` muss mit ins Repo, sonst zeigt die Website keine Evaluationsergebnisse.

## Slack-Bot

Der Bot beantwortet Erwähnungen im Channel mit dem bevorzugten Ansatz (LLM mit Vektorindex) und nennt die relevante Policy. Er läuft lokal über Socket Mode:

```bash
python slack_bot.py
```

Einrichtung: Slack-App unter api.slack.com/apps mit `slack_manifest.yml` anlegen, im Workspace installieren, App-Level-Token mit Scope `connections:write` erzeugen und beide Tokens (`SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`) in `.env` eintragen. Im Channel den Bot mit `/invite @Policy Bot` hinzufügen und mit `@Policy Bot <Frage>` aufrufen.

## Metriken (`evaluate.py`)

- **accuracy**: richtige Policy genannt bzw. korrekt abgelehnt, wenn keine Policy passt
- **unsupported_rate**: Anteil Antworten, die nicht durch die Datenbank gedeckt sind:
  zitierte Policy existiert nicht, Frage ist nicht abgedeckt, wird aber trotzdem beantwortet,
  oder Antwort enthält Zahlen, die weder in der Policy noch in der Frage stehen
- **avg/median_seconds**, **avg_tokens** (Input + Output, Embeddings laufen lokal = 0 API-Tokens)

Testfragen: `data/test_questions.csv` (direkt, umformuliert, fehlendes Detail, nicht abgedeckt).

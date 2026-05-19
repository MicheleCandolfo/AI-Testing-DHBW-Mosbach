# AI Engineering Portfolio – DeepEval LLM Evaluation

> **Kurs:** TestGilde – AI Engineering  
> **Ziel:** LLMs und RAG-Pipelines systematisch evaluieren mit [DeepEval](https://github.com/confident-ai/deepeval) und lokalen Ollama-Modellen – ohne bezahlte Cloud-APIs.

---

## Schnellstart

### 1. Repo forken & klonen

```bash
# 1) Auf GitHub auf "Fork" klicken
# 2) Dann den eigenen Fork klonen:
git clone https://github.com/<DEIN_USERNAME>/ai-engineering-portfolio.git
cd ai-engineering-portfolio
```

### 2. Ollama installieren & Modelle laden

```bash
# Ollama installieren: https://ollama.com
ollama serve

# Benötigte Modelle pullen:
ollama pull qwen3:latest              # Judge-Modell
ollama pull gpt-oss:latest            # Candidate-Modell
ollama pull nomic-embed-text:latest   # Embedding (LangChain)
ollama pull qwen3-embedding:latest    # Embedding (DeepEval Synthesizer)
```

> **Hinweis:** Wenn ihr einen Cloud-Ollama-Server nutzt (z.B. `gpt-oss:20b-cloud`, `qwen3.5:cloud`), müsst ihr die Modelle dort verfügbar machen und `CLOUD_MODEL_BASE_URL` in `.env.local` setzen.

### 3. Python-Umgebung einrichten

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -U pip
pip install deepeval langchain langchain-community langchain-ollama langchain-openai \
  langchain-text-splitters langchain-chroma faiss-cpu chromadb python-dotenv \
  tiktoken pandas pypdf pydantic ipywidgets jupyterlab duckduckgo-search
```

### 4. Umgebungsvariablen konfigurieren

```bash
# .env.local.example kopieren und ausfüllen:
cp .env.local.example .env.local
```

`.env.local` Inhalt (die wichtigsten Einträge):

```env
# ── Ollama (lokal) ───────────────────────────────────────
LOCAL_MODEL_BASE_URL=http://localhost:11434
LOCAL_MODEL_NAME=qwen3:latest

# ── Ollama (Cloud / Remote) ─────────────────────────────
CLOUD_MODEL_BASE_URL=https://euer-ollama-server.example.com
OLLAMA_API_KEY=euer-api-key

# ── DeepEval / Confident AI ──────────────────────────────
CONFIDENT_API_KEY=confident_us_...
API_KEY=confident_us_...

# ── Embedding ────────────────────────────────────────────
LOCAL_EMBEDDING_MODEL_NAME=qwen3-embedding:latest
LOCAL_EMBEDDING_BASE_URL=http://localhost:11434

# ── DeepEval Timeouts ───────────────────────────────────
DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=600
DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=180
DEEPEVAL_RETRY_MAX_ATTEMPTS=2
```

> `.env.local` ist gitignored – **niemals API-Keys committen!**

### 5. DeepEval & Confident AI verbinden

```bash
deepeval login --confident-api-key YOUR_CONFIDENT_API_KEY
deepeval set-ollama qwen3:latest --base-url=http://localhost:11434
```

Einige Notebooks führen diese Befehle auch in der ersten Zelle aus.

---

## Repository-Struktur

```
.
├── 01_LLM_Testing/                  # Einzelne LLM-Metriken
│   ├── answer_relevancy.ipynb       #   Answer Relevancy
│   ├── hallucination.ipynb          #   Hallucination
│   ├── summarization.ipynb          #   Summarization
│   └── toxicity.ipynb               #   Toxicity
├── 02_The_RAG_Agent/                # RAG-Pipeline: bauen, evaluieren, verbessern
│   ├── create_evaluate_improve.ipynb
│   ├── ISO_25059-2026-01.pdf       # Quelldokument (ISO-Standard)
│   └── theranos_legacy.txt         # Quelldokument (Theranos-Fallstudie)
├── 03_AI Agents Testing/            # KI-Agenten mit Tool-Calling evaluieren
│   └── creating_and_testing_AIAgent.ipynb
├── .env.local                       # Umgebungsvariablen (gitignored)
├── .env.local.example               # Vorlage für .env.local
└── README.md
```

---

## Notebooks im Detail

### 01_LLM_Testing – Einzelne Metriken

Jedes Notebook folgt dem gleichen Ablauf:

1. **Judge-Modell** erstellen (`OllamaModel`) → bewertet die Ausgaben
2. **Candidate-Modell** erstellen (`ChatOllama`) → generiert die Ausgaben
3. **Goldens** definieren (Input + erwarteter Output)
4. **`evaluate()`** aufrufen → Ergebnisse als DataFrame anzeigen
5. **Interpretation** – welche Testfälle bestehen, welche nicht, und warum

| Notebook | Metrik | Was wird gemessen | Score-Interpretation |
|----------|--------|-------------------|----------------------|
| `answer_relevancy.ipynb` | AnswerRelevancyMetric | Anteil relevanter Aussagen im Output | 0.5+ = bestanden |
| `hallucination.ipynb` | HallucinationMetric | Widersprüche zwischen Output und Kontext | < 0.5 = bestanden |
| `summarization.ipynb` | SummarizationMetric | Alignment + Coverage der Zusammenfassung | 0.5+ = bestanden |
| `toxicity.ipynb` | ToxicityMetric | Anteil toxischer/biased Aussagen | < 0.5 = bestanden |

### 02_The_RAG_Agent – Vollständige Pipeline

Dieses Notebook zeigt den gesamten RAG-Lebenszyklus:

1. **RAG-Agent bauen** – Dokumente chunken, in FAISS speichern, per Similarity retrieven, JSON-Antworten generieren
2. **Dataset erstellen** – DeepEval Synthesizer generiert Goldens aus einem PDF
3. **Evaluieren** – ContextualRelevancy, ContextualRecall, ContextualPrecision (Retriever) + GEval (Generator)
4. **Verbessern** – Systematisches Hyperparameter-Tuning (chunk_size, k, Embedding-Modell, Vector Store)

### 03_AI Agents Testing – Tool-Calling evaluieren

1. **KI-Agent erstellen** – mit ReAct-Agenten, DuckDuckGo-Suche, Add/Subtract-Tools
2. **Tool-Aufrufe testen** – DeepEval `ToolCorrectnessMetric` prüft, ob der Agent die richtigen Tools aufruft
3. **Ergebnisse interpretieren** – Score und Reasoning pro Testfall

---

## Modell-Rollen

| Rolle | Modell | Verwendung |
|-------|--------|------------|
| **Judge** | `qwen3:latest` oder `gpt-oss:20b-cloud` | DeepEval-Metriken (bewertet Ausgaben) |
| **Candidate** | `gpt-oss:latest` oder `gpt-oss:20b-cloud` | Das LLM das getestet wird (generiert Ausgaben) |
| **Embedding (LangChain)** | `nomic-embed-text:latest` | FAISS / Chroma Vector Stores |
| **Embedding (DeepEval)** | `qwen3-embedding:latest` | Synthesizer Chunking & Context Construction |

> **Tipp:** Für Tool-Calling (Notebook 03) muss das Modell Tool-Calling unterstützen. `qwen3:latest` funktioniert, `gpt-oss` aktuell nicht.

---

## Bekannte Probleme & Workarounds

### 1. Cloud-Modelle liefern JSON in Markdown-Codeblöcken

**Problem:** Cloud-Ollama-Modelle (z.B. `gpt-oss:20b-cloud`) hüllen JSON-Antworten in `` ```json ... ``` ```, was `pydantic.ValidationError` verursacht.

**Fix:** `PatchedOllamaModel` statt `OllamaModel` nutzen (steht im Notebook 02):

```python
class PatchedOllamaModel(OllamaModel):
    @staticmethod
    def _strip_markdown_json(text):
        text = text.strip()
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return text

    def generate(self, prompt, schema=None):
        chat_model = self.load_model()
        response = chat_model.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            format=schema.model_json_schema() if schema else None,
            options={**{"temperature": self.temperature}, **self.generation_kwargs},
        )
        content = response.message.content
        if schema:
            content = self._strip_markdown_json(content)
            return schema.model_validate_json(content), 0
        return content, 0
```

### 2. Embedding-Kontextlänge-Overflow

**Problem:** `nomic-embed-text` hat nur 8K Token Kontext. Große Chunks verursachen `the input length exceeds the context length`.

**Fix:** `SafeOllamaEmbedder` nutzen (steht im Notebook 02), oder `qwen3-embedding:latest` (40K Kontext).

### 3. Rate-Limiting (429-Fehler) bei `evaluate()`

**Problem:** `evaluate()` läuft asynchron und schickt viele parallele Requests → `429 Too Many Requests`.

**Fix:** Synchrone Ausführung erzwingen:

```python
from deepeval.evaluate import AsyncConfig

results = evaluate(
    test_cases=test_cases,
    metrics=[metric],
    async_config=AsyncConfig(run_async=False),  # NICHT async_mode=False!
)
```

> Alternativ: `AsyncConfig(run_async=True, max_concurrent=3)` für begrenzte Parallelität.

### 4. ChromaDB-Collection-Namen dürfen keine Leerzeichen

**Problem:** PDF-Dateinamen mit Leerzeichen (z.B. `ISO 25059-2026-01.pdf`) verursachen `InvalidArgumentError`.

**Fix:** Unterstriche statt Leerzeichen: `ISO_25059-2026-01.pdf`.

### 5. DeepEval-Embedding vs. LangChain-Embedding

**Problem:** `TypeError: Unsupported type for embedding model` – LangChains `OllamaEmbeddings` funktioniert nicht mit DeepEvals Synthesizer.

**Fix:** Für DeepEval immer `deepeval.models.OllamaEmbeddingModel` (oder `SafeOllamaEmbedder`) nutzen. LangChains `OllamaEmbeddings` nur für FAISS/Chroma Vector Stores.

### 6. KI-Agent nutzt keine Tools

**Problem:** `create_agent()` erwartet natives Tool-Calling, aber nicht alle Ollama-Modelle unterstützen das.

**Fix:** Ein Modell mit Tool-Calling-Unterstützung nutzen (z.B. `qwen3:latest`) und Tools explizit binden:

```python
llm_with_tools = llm.bind_tools(tools)
agent = create_agent(llm_with_tools, tools)
```

---

## Confident AI Integration

DeepEval lädt Evaluations-Ergebnisse automatisch auf [Confident AI](https://app.confident-ai.com) hoch. Datasets können auch gepusht/gepullt werden:

```python
# Dataset hochladen
dataset.push(alias="Mein Dataset")

# Dataset herunterladen (konvertiert Goldens zu LLMTestCases)
dataset.pull(alias="Mein Dataset", auto_convert_goldens_to_test_cases=True)
```

---

## .env.local.example

Eine Vorlagendatei für die Umgebungsvariablen liegt bei. Kopiert sie und füllt eure Werte aus:

```bash
cp .env.local.example .env.local
# Dann .env.local mit euren Werten ausfüllen
```

---

## Tipps für Studierende

- **Notebook-Reihenfolge:** Startet mit `01_LLM_Testing/` (beliebige Reihenfolge), dann `02_The_RAG_Agent/`, dann `03_AI Agents Testing/`
- **Erst die erste Zelle ausführen** – dort werden die Umgebungsvariablen geladen und die Modelle initialisiert
- **Bei Fehlern:** Prüft ob Ollama läuft (`ollama list`) und ob alle Modelle heruntergeladen sind
- **429-Fehler:** Nutzt `async_config=AsyncConfig(run_async=False)` bei `evaluate()`
- **Leere Ergebnisse:** Prüft ob das Embedding-Modell läuft und die Kontextlänge ausreicht
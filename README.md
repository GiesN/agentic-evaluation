# Agentic Evaluation

LLM-Ops: Evaluating LangGraph Agents and Graphs with MLflow GenAI Evaluate.

Companion code for the Medium article *"LLM-Ops and Evaluation of LangGraph Agents and Graphs with MLflow"*.

## Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation)
- An Azure AI Foundry endpoint with a deployed GPT model (or any other LLM provider, swap the LangChain client and config in `evaluate.py`, e.g. `ChatOpenAI`, `ChatAnthropic`, etc.)
- Azure credentials configured (e.g. `az login`; only needed when using Azure AI Foundry)

## Setup

```bash
# Install dependencies
poetry install

# Copy the env template and fill in your values
cp .env.example .env
```

## Running

```bash
# 1. Start the MLflow tracking server
poetry run mlflow server --host 127.0.0.1 --port 5050

# 2. In another terminal, run the evaluation
poetry run python evaluate.py
```

Open http://127.0.0.1:5050 to explore experiments, runs, and traces.

## What It Does

1. Builds a LangGraph agent that extracts company names from remittance advice emails
2. Runs the agent against 12 diverse evaluation samples (multi-language, varied formats)
3. Scores each output with three custom MLflow scorers:
   - **exact_match**: strict case-insensitive string equality
   - **contains_company**: substring check (tolerates extra text)
   - **llm_judge**: LLM-as-a-judge for semantic correctness
4. Generates classification reports and confusion matrix visualizations
5. Saves results to `eval_results.png`

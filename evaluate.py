"""
LLM-Ops: Evaluating LangGraph Agents with MLflow GenAI Evaluate

Companion script for the Medium article:
"LLM-Ops and Evaluation of LangGraph Agents and Graphs with MLflow"

Usage:
    1. Start MLflow:  poetry run mlflow server --host 127.0.0.1 --port 5050
    2. Run this:      poetry run python evaluate.py
"""

import json

import matplotlib.pyplot as plt
import mlflow
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain.messages import SystemMessage
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langgraph.graph import END, START, MessagesState, StateGraph
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer
from mlflow.langchain import autolog as langchain_autolog
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

load_dotenv()

# ── LLM setup ────────────────────────────────────────────────────────────────

llm = AzureAIOpenAIApiChatModel(
    model="gpt-5.4-mini",
    credential=DefaultAzureCredential(),
)

# ── LangGraph agent ──────────────────────────────────────────────────────────


def llm_call(state: MessagesState):
    """Single-node graph: extract company name from a remittance advice email."""
    return {
        "messages": [
            llm.invoke(
                [
                    SystemMessage(
                        content="You are an assistant that extracts information "
                        "from remittance advice emails. Given the email content, "
                        "extract the following information: Customer name (company)."
                        "Only respond with the company name, no additional text."
                    )
                ]
                + state["messages"]
            )
        ]
    }


agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_edge("llm_call", END)
agent = agent_builder.compile()

# ── Evaluation dataset ───────────────────────────────────────────────────────

MAILS = [
    (
        "From: john.smith@smith-manufacturing.com\n"
        "Subject: Payment Remittance Advice - Invoice #123456\n"
        "Company: Smith Manufacturing Ltd.\n"
        "Vendor Name: ABC Suppliers Inc.\n"
        "Invoice Number: 123456\nInvoice Amount: $250.00\n"
        "Payment Method: Bank Transfer",
        "Smith Manufacturing Ltd.",
    ),
    (
        "From: anna.jones@globaltech.io\n"
        "Subject: Remittance - Invoice #789012\n"
        "Company: GlobalTech Solutions GmbH\n"
        "Vendor Name: XYZ Parts Co.\n"
        "Invoice Number: 789012\nInvoice Amount: $1,450.00\n"
        "Payment Method: Wire Transfer",
        "GlobalTech Solutions GmbH",
    ),
    (
        "From: li.wei@dragonlogistics.cn\n"
        "Subject: Payment Confirmation - PO-2024-88321\n"
        "Company: Dragon Logistics Co., Ltd.\n"
        "Vendor Name: FastShip International\n"
        "Invoice Number: 88321\nInvoice Amount: $3,200.00\n"
        "Payment Method: SWIFT Transfer",
        "Dragon Logistics Co., Ltd.",
    ),
    (
        "From: m.garcia@fernandez-hermanos.es\n"
        "Subject: Aviso de pago - Factura #55001\n"
        "Company: Fernandez Hermanos S.A.\n"
        "Vendor Name: Iberian Steel Corp\n"
        "Invoice Number: 55001\nInvoice Amount: €8,750.00\n"
        "Payment Method: SEPA Transfer",
        "Fernandez Hermanos S.A.",
    ),
    (
        "From: cfo@brightfuture-energy.com\n"
        "Subject: Remittance Advice\n"
        "Dear Accounts Receivable,\n"
        "Please find below our payment details.\n"
        "Company: BrightFuture Energy Inc.\n"
        "Invoice Number: BFE-2024-1001\nInvoice Amount: $12,000.00\n"
        "Payment Method: ACH",
        "BrightFuture Energy Inc.",
    ),
    (
        "From: procurement@nordicpaper.se\n"
        "Subject: Betalningsavi - Faktura 44002\n"
        "Company: Nordic Paper AB\n"
        "Vendor Name: PulpTech Oy\n"
        "Invoice Number: 44002\nInvoice Amount: SEK 95,000\n"
        "Payment Method: Bankgiro",
        "Nordic Paper AB",
    ),
    (
        "From: ap@summit-pharma.co.uk\n"
        "Subject: Payment Notification - Inv #UK-7890\n"
        "Company: Summit Pharmaceuticals plc\n"
        "Vendor Name: ChemSource Ltd\n"
        "Invoice Number: UK-7890\nInvoice Amount: £6,300.00\n"
        "Payment Method: BACS",
        "Summit Pharmaceuticals plc",
    ),
    (
        "From: finance@autoparts-midwest.com\n"
        "Subject: Remittance for multiple invoices\n"
        "Company: Midwest AutoParts LLC\n"
        "Vendor Name: TireWorld Distributors\n"
        "Invoice Numbers: MW-101, MW-102, MW-103\n"
        "Total Amount: $4,580.00\nPayment Method: Check #8842",
        "Midwest AutoParts LLC",
    ),
    (
        "From: tanaka.h@sakura-electronics.jp\n"
        "Subject: 送金通知 - Invoice SE-2024-555\n"
        "Company: Sakura Electronics K.K.\n"
        "Vendor Name: Silicon Valley Chips Corp\n"
        "Invoice Number: SE-2024-555\nInvoice Amount: ¥1,250,000\n"
        "Payment Method: Wire Transfer",
        "Sakura Electronics K.K.",
    ),
    (
        "From: accounting@terrafarma.com.br\n"
        "Subject: Aviso de Remessa - NF 33210\n"
        "Company: TerraFarma Agrícola Ltda.\n"
        "Vendor Name: GreenChem Soluções\n"
        "Invoice Number: 33210\nInvoice Amount: R$ 27,500.00\n"
        "Payment Method: TED",
        "TerraFarma Agrícola Ltda.",
    ),
    (
        "From: ops@outbackmining.com.au\n"
        "Subject: Payment Advice - Inv #OM-4421\n"
        "Company: Outback Mining Pty Ltd\n"
        "Vendor Name: HeavyDuty Equipment Co.\n"
        "Invoice Number: OM-4421\nInvoice Amount: AUD 18,900.00\n"
        "Payment Method: Direct Deposit",
        "Outback Mining Pty Ltd",
    ),
    (
        "From: buyer@casablanca-textiles.ma\n"
        "Subject: Avis de règlement - Facture #CT-8800\n"
        "Company: Casablanca Textiles SARL\n"
        "Vendor Name: EuroFabric NV\n"
        "Invoice Number: CT-8800\nInvoice Amount: MAD 145,000\n"
        "Payment Method: Virement bancaire",
        "Casablanca Textiles SARL",
    ),
]

dataset = [
    {
        "inputs": {"mail_content": mail},
        "expectations": {"expected_response": expected},
    }
    for mail, expected in MAILS
]

# ── Predict function ─────────────────────────────────────────────────────────


def predict_fn(mail_content: str) -> str:
    """Run the agent on mail content and return the last message text."""
    messages = [SystemMessage(content=mail_content)]
    result = agent.invoke({"messages": messages})
    content = result["messages"][-1].content
    if isinstance(content, list):
        return content[0]["text"]
    return content


# ── Scorers ──────────────────────────────────────────────────────────────────


@scorer
def exact_match(inputs, outputs, expectations):
    """Strict case-insensitive string equality."""
    expected = expectations["expected_response"].strip()
    return expected.lower() == outputs.strip().lower()


@scorer
def contains_company(inputs, outputs, expectations):
    """Substring check — tolerates extra text around the company name."""
    expected = expectations["expected_response"].strip()
    return expected.lower() in outputs.strip().lower()


@scorer
def llm_judge(inputs, outputs, expectations):
    """LLM-as-a-judge: uses the same model to grade semantic correctness."""
    expected = expectations["expected_response"]
    judge_prompt = (
        "You are an evaluation judge. Compare the expected answer with the actual answer.\n"
        "Determine if the actual answer is correct - it should convey the same entity/value "
        "as the expected answer, even if wording differs slightly.\n\n"
        f"Expected: {expected}\n"
        f"Actual: {outputs}\n\n"
        "Respond with ONLY a JSON object: "
        '{"score": <1 if correct, 0 if incorrect>, "rationale": "<brief explanation>"}'
    )
    response = llm.invoke([SystemMessage(content=judge_prompt)])
    content = response.content
    if isinstance(content, list):
        content = content[0]["text"]
    try:
        result = json.loads(content)
        return Feedback(
            name="llm_judge",
            value=result["score"],
            rationale=result.get("rationale", ""),
        )
    except (json.JSONDecodeError, KeyError):
        return Feedback(
            name="llm_judge",
            value=0,
            rationale=f"Could not parse judge response: {content}",
        )


# ── Run evaluation ───────────────────────────────────────────────────────────

mlflow.set_tracking_uri("http://127.0.0.1:5050")
mlflow.set_experiment("LangGraph Evaluation - GenAI Full")
langchain_autolog()

results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=predict_fn,
    scorers=[exact_match, contains_company, llm_judge],
)

df = results.result_df
print("\n=== Raw results ===")
print(df.to_string())

# ── Classification reports ───────────────────────────────────────────────────

scorer_columns = {
    "exact_match": "exact_match/boolean",
    "contains_company": "contains_company/boolean",
    "llm_judge": "llm_judge/value",
}

y_true = [1] * len(df)

print("\n=== Classification reports ===")
reports = {}
for name, col in scorer_columns.items():
    if col not in df.columns:
        candidates = [c for c in df.columns if c.startswith(name)]
        if candidates:
            col = candidates[0]
        else:
            print(f"  ⚠ Column for '{name}' not found, skipping. Available: {list(df.columns)}")
            continue

    y_pred = df[col].astype(int).tolist()
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["incorrect", "correct"],
        output_dict=True,
        zero_division=0,
    )
    reports[name] = report
    print(f"\n--- {name} ---")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["incorrect", "correct"],
            zero_division=0,
        )
    )

if not reports:
    print("No scorer columns found. Printing available columns for debugging:")
    print(list(df.columns))
    raise SystemExit(1)

# ── Visualization ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(reports) + 1, figsize=(5 * (len(reports) + 1), 5))
if len(reports) + 1 == 1:
    axes = [axes]

# Confusion matrix per scorer
for ax, (name, col) in zip(axes, scorer_columns.items(), strict=False):
    if name not in reports:
        continue
    candidates = [c for c in df.columns if c.startswith(name)]
    col = candidates[0] if candidates else col
    y_pred = df[col].astype(int).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["incorrect", "correct"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"{name}")

# Summary bar chart
summary_ax = axes[-1]
metric_names = ["precision", "recall", "f1-score"]
x = range(len(metric_names))
width = 0.8 / len(reports)
for i, (name, report) in enumerate(reports.items()):
    correct_metrics = report["correct"]
    vals = [correct_metrics[m] for m in metric_names]
    bars = summary_ax.bar([xi + i * width for xi in x], vals, width, label=name)
    for bar, val in zip(bars, vals, strict=True):
        summary_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

summary_ax.set_xticks([xi + width * (len(reports) - 1) / 2 for xi in x])
summary_ax.set_xticklabels(metric_names)
summary_ax.set_ylim(0, 1.15)
summary_ax.set_ylabel("Score")
summary_ax.set_title("Metrics per scorer (class: correct)")
summary_ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("eval_results.png", dpi=150)
plt.show()
print("\nPlot saved to eval_results.png")

"""Deterministic narrative templates for the fraud-investigation agent.

These power the zero-dependency fallback so the agent always produces a
human-readable narrative, even without an LLM API key. When the Groq/LangChain
backend is enabled the same structured evidence is handed to the LLM instead.
"""

from __future__ import annotations

ACTION_BY_DECISION = {
    "decline": "DECLINE the transaction and freeze the account pending review",
    "challenge": "CHALLENGE with step-up authentication (3-D Secure / OTP)",
    "approve": "APPROVE — risk below action threshold",
}


def build_narrative_text(evidence: dict) -> str:
    """Render a compliance-ready fraud narrative from structured evidence."""

    ring = evidence.get("ring")
    lines = [
        f"Fraud Investigation Summary - Transaction {evidence['tx_id']}",
        "=" * 60,
        f"Account        : {evidence['account_id']}",
        f"Device         : {evidence['device_id']}",
        f"Merchant       : {evidence['merchant_id']}",
        f"Amount         : ${evidence['amount']:,.2f}",
        f"Fraud score    : {evidence['fraud_score']:.2%}",
        f"Recommendation : {ACTION_BY_DECISION.get(evidence['decision'], evidence['decision'])}",
        "",
        "Graph evidence:",
        f"  - Account velocity in window : {evidence['velocity']} transactions",
        f"  - Devices linked to account  : {evidence['unique_devices']}",
        f"  - Shared-device degree       : {evidence['device_degree']} accounts",
        f"  - Account graph component    : {evidence['component_size']} nodes",
    ]
    if ring is not None:
        lines += [
            "",
            "Coordinated fraud ring detected:",
            f"  - Ring id            : {ring['ring_id']}",
            f"  - Accounts in ring   : {ring['size']}",
            f"  - Shared-device ratio: {ring['shared_device_ratio']:.0%}",
            f"  - Ring risk score    : {ring['risk_score']:.2%}",
            "  - Interpretation     : multiple accounts sharing a small device "
            "pool and targeting the same merchants is a classic mule-ring "
            "cash-out pattern.",
        ]
    else:
        lines += [
            "",
            "No coordinated ring membership detected for this account in the "
            "current window; risk is driven by transaction-level and velocity "
            "signals.",
        ]
    return "\n".join(lines)

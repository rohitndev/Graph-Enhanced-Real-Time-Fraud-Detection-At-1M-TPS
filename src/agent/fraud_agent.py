"""LangChain Fraud Investigation Agent.

For each high-risk transaction the agent autonomously assembles graph evidence,
generates a natural-language fraud narrative with a confidence score, and
recommends an action (decline / challenge / approve) for the compliance team.

Two backends are supported:

* ``template`` (default) — a deterministic, zero-dependency narrative generator
  that always works offline.
* ``groq`` — a LangChain + Groq (Mixtral) LLM chain, used automatically when
  ``AGENT_PROVIDER=groq``, ``GROQ_API_KEY`` is set, and the optional
  ``langchain-groq`` package is installed.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from src.agent.narrative_templates import ACTION_BY_DECISION, build_narrative_text
from src.config import AgentConfig
from src.graph.ring_detection import FraudRing
from src.models.score import ScoredTransaction


@dataclass
class FraudNarrative:
    tx_id: str
    account_id: str
    fraud_score: float
    decision: str
    recommended_action: str
    confidence: float
    ring_id: int | None
    narrative: str


class FraudInvestigationAgent:
    """Generates fraud narratives for high-risk transactions."""

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self._llm = None
        if cfg.provider == "groq" and cfg.groq_api_key:
            self._llm = self._init_groq()

    def _init_groq(self):  # pragma: no cover - optional path
        try:
            from langchain_groq import ChatGroq

            logger.info("Fraud agent using Groq LLM backend ({}).", self.cfg.model_name)
            return ChatGroq(
                api_key=self.cfg.groq_api_key,
                model_name=self.cfg.model_name,
                temperature=0.2,
            )
        except Exception as exc:
            logger.warning("Groq backend unavailable ({}); using templates.", exc)
            return None

    # ------------------------------------------------------------------
    def investigate(
        self,
        scored: ScoredTransaction,
        graph_row: dict,
        ring: FraudRing | None,
    ) -> FraudNarrative:
        """Produce a narrative for a single high-risk transaction."""

        evidence = {
            "tx_id": scored.tx_id,
            "account_id": scored.account_id,
            "device_id": scored.device_id,
            "merchant_id": scored.merchant_id,
            "amount": scored.amount,
            "fraud_score": scored.fraud_score,
            "decision": scored.decision,
            "velocity": int(graph_row.get("velocity_ring_score", 0)),
            "unique_devices": int(graph_row.get("unique_devices_per_account", 0)),
            "device_degree": int(graph_row.get("device_degree", 0)),
            "component_size": int(graph_row.get("component_size", 0)),
            "ring": _ring_evidence(ring),
        }

        if self._llm is not None:  # pragma: no cover - optional path
            narrative = self._llm_narrative(evidence)
        else:
            narrative = build_narrative_text(evidence)

        # Confidence blends the model score with ring corroboration.
        confidence = scored.fraud_score
        if ring is not None:
            confidence = min(1.0, 0.7 * scored.fraud_score + 0.3 * ring.risk_score)

        return FraudNarrative(
            tx_id=scored.tx_id,
            account_id=scored.account_id,
            fraud_score=scored.fraud_score,
            decision=scored.decision,
            recommended_action=ACTION_BY_DECISION.get(scored.decision, scored.decision),
            confidence=round(float(confidence), 4),
            ring_id=ring.ring_id if ring else None,
            narrative=narrative,
        )

    def _llm_narrative(self, evidence: dict) -> str:  # pragma: no cover - optional
        from langchain_core.messages import HumanMessage, SystemMessage

        system = SystemMessage(
            content=(
                "You are a fraud investigation analyst. Given structured graph "
                "evidence, write a concise, compliance-ready narrative explaining "
                "why a transaction is suspicious and the recommended action."
            )
        )
        human = HumanMessage(content=str(evidence))
        return self._llm.invoke([system, human]).content


def _ring_evidence(ring: FraudRing | None) -> dict | None:
    if ring is None:
        return None
    return {
        "ring_id": ring.ring_id,
        "size": ring.size,
        "shared_device_ratio": ring.shared_device_ratio,
        "risk_score": ring.risk_score,
    }

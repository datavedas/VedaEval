"""Conversational metrics - multi-turn judges.

Data contract: a `history` column (canonical field; aliases:
conversation, chat_history, messages, dialog) holding the prior turns
as plain text. Rows without history are not applicable - single-turn
datasets simply never see these metrics.
"""

from __future__ import annotations

from vedaeval.evaluators.base import EvaluatorInfo
from vedaeval.evaluators.judge import _JudgeBase


class RoleAdherenceJudge(_JudgeBase):
    info = EvaluatorInfo(
        key="role_adherence", name="Role Adherence (LLM judge)",
        category="quality", inputs=["response", "history"], needs_llm=True,
        description="Multi-turn: does the assistant stay in its "
                    "established role/persona and policies across turns "
                    "(no sudden tone breaks, no forgetting its function)?",
    )
    OUTPUT_PREFIX = "role_adherence"
    PROMPT = (
        "You are auditing a multi-turn AI conversation.\n"
        "Conversation so far: {history}\n"
        "Latest user message: {request}\n"
        "Latest assistant response: {response}\n\n"
        "Based on the role, tone and policies the assistant established "
        "earlier, does this response stay in character and function?\n"
        'Reply with ONLY a JSON object: {{"rating": "High"|"Medium"|"Low", '
        '"reason": "<one short sentence>"}}'
    )


class ConversationRelevancyJudge(_JudgeBase):
    info = EvaluatorInfo(
        key="conversation_relevancy", name="Conversation Relevancy (LLM judge)",
        category="quality", inputs=["request", "response", "history"],
        needs_llm=True,
        description="Multi-turn: is the response relevant IN CONTEXT of the "
                    "whole conversation, not just the last message "
                    "(pronouns, follow-ups, corrections)?",
    )
    OUTPUT_PREFIX = "conversation_relevancy"
    PROMPT = (
        "You are evaluating a reply within an ongoing conversation.\n"
        "Conversation so far: {history}\n"
        "Latest user message: {request}\n"
        "Assistant response: {response}\n\n"
        "Considering what came before (references, follow-ups, "
        "corrections), does the response fit THIS point in the "
        "conversation?\n"
        'Reply with ONLY a JSON object: {{"rating": "High"|"Medium"|"Low", '
        '"reason": "<one short sentence>"}}'
    )


class KnowledgeRetentionJudge(_JudgeBase):
    info = EvaluatorInfo(
        key="knowledge_retention", name="Knowledge Retention (LLM judge)",
        category="quality", inputs=["response", "history"], needs_llm=True,
        description="Multi-turn: does the assistant remember facts the user "
                    "already provided (name, situation, constraints) instead "
                    "of re-asking or contradicting them?",
    )
    OUTPUT_PREFIX = "knowledge_retention"
    PROMPT = (
        "You are auditing an AI assistant's memory within a conversation.\n"
        "Conversation so far: {history}\n"
        "Latest assistant response: {response}\n\n"
        "Did the assistant correctly retain and use facts the user "
        "provided earlier (re-asking for given information or "
        "contradicting it = failure)?\n"
        'Reply with ONLY a JSON object: {{"rating": "High"|"Medium"|"Low", '
        '"reason": "<one short sentence>"}}'
    )

"""Evaluator registry.

REGISTRY maps evaluator key -> Evaluator instance. The UI and engine
only ever talk to the registry, so adding a metric = adding a class
and registering it here (mirrors the config-driven design of
enterprise eval platforms: no code changes elsewhere).
"""

from __future__ import annotations

from vedaeval.evaluators.base import Evaluator, EvaluatorInfo
from vedaeval.evaluators.deterministic import (
    TextStat, Sentiment, Overlap, Profanity, BannedKeywords,
    RegexMatch, TokenCount, JsonValidation, SqlValidation,
)
from vedaeval.evaluators.safety import SafetyClassifier
from vedaeval.evaluators.faithfulness import NLIFaithfulness
from vedaeval.evaluators.topic import TopicClassification
from vedaeval.evaluators.language import LanguageDetection
from vedaeval.evaluators.refusal import RefusalDetection
from vedaeval.evaluators.rag_depth import (
    RetrievalHitRate, ContextEntityRecall, ContextRecall, ContextPrecision,
    AnswerCorrectness,
)
from vedaeval.evaluators.reference import (
    ExactMatch, LevenshteinSimilarity, ChrfTer, Meteor, BertScore,
    EmbeddingSimilarity,
)
from vedaeval.evaluators.hallucination import (
    SummaCConsistency, QagSupportRatio, SampleConsistency, SummaryStats,
)
from vedaeval.evaluators.judge_qualities import (
    CompletenessJudge, HelpfulnessJudge, InstructionAdherenceJudge,
    GEvalJudge, PairwiseJudge,
)
from vedaeval.evaluators.privacy_calibration import (
    VerbatimCopyRate, Calibration,
)
from vedaeval.evaluators.extraction import ExtractionAccuracy
from vedaeval.evaluators.conversational import (
    RoleAdherenceJudge, ConversationRelevancyJudge, KnowledgeRetentionJudge,
)
from vedaeval.evaluators.safety_novel import (
    ToxicityPreservation, UnsafeSourceUtilization,
)
from vedaeval.evaluators.healthcare import (
    PhiEchoScore, PlanGroundedCorrectness, DeflectionDetection,
    PhiEntityScreen,
)
from vedaeval.evaluators.consistency import ResponseConsistency
from vedaeval.evaluators.safety_screens import (
    JailbreakDetection, HarmTaxonomy, RefusalCorrectness, ModerationScreen,
)
from vedaeval.evaluators.rag_second import (
    FaithfulnessJudge, NoiseSensitivity, CitationPrecision, CitationRecall,
)
from vedaeval.evaluators.quality_ops import (
    MoverSimilarity, DiversitySuite, IntentMatch, MarkdownValidity,
    LatencyCost, CodeExecution,
)
from vedaeval.evaluators.judge import (
    AnswerRelevanceJudge, ContextRelevanceJudge, CoherenceJudge,
    ConcisenessJudge, CustomJudge,
)

REGISTRY: dict[str, Evaluator] = {}


def register(evaluator: Evaluator) -> None:
    REGISTRY[evaluator.info.key] = evaluator


for _ev in (TextStat(), Sentiment(), Overlap(), Profanity(), BannedKeywords(),
            RegexMatch(), TokenCount(), JsonValidation(), SqlValidation(),
            SafetyClassifier(), NLIFaithfulness(),
            TopicClassification(), LanguageDetection(), RefusalDetection(),
            RetrievalHitRate(), ContextEntityRecall(), ContextRecall(),
            ContextPrecision(), AnswerCorrectness(),
            ExactMatch(), LevenshteinSimilarity(), ChrfTer(), Meteor(),
            BertScore(), EmbeddingSimilarity(),
            SummaCConsistency(), QagSupportRatio(), SampleConsistency(),
            SummaryStats(),
            CompletenessJudge(), HelpfulnessJudge(),
            InstructionAdherenceJudge(), GEvalJudge(), PairwiseJudge(),
            VerbatimCopyRate(), Calibration(),
            ExtractionAccuracy(),
            RoleAdherenceJudge(), ConversationRelevancyJudge(),
            KnowledgeRetentionJudge(),
            ToxicityPreservation(), UnsafeSourceUtilization(),
            PhiEchoScore(), PlanGroundedCorrectness(), DeflectionDetection(),
            ResponseConsistency(),
            JailbreakDetection(), HarmTaxonomy(), RefusalCorrectness(),
            ModerationScreen(), PhiEntityScreen(),
            FaithfulnessJudge(), NoiseSensitivity(), CitationPrecision(),
            CitationRecall(),
            MoverSimilarity(), DiversitySuite(), IntentMatch(),
            MarkdownValidity(), LatencyCost(), CodeExecution(),
            AnswerRelevanceJudge(), ContextRelevanceJudge(), CoherenceJudge(),
            ConcisenessJudge()):
    register(_ev)


def available_evaluators() -> dict[str, tuple[bool, str]]:
    """Key -> (available, reason-if-not)."""
    return {key: ev.available() for key, ev in REGISTRY.items()}


def recommended_for(task_type: str, rag: bool) -> list[str]:
    """Task-aware metric recommendation for the skip-intake fallback path.

    The framework overlay (vedaeval.overlay) is the primary recommender;
    this heuristic runs only when a user skips the risk intake. A basic
    safety set is always suggested (it was previously gated behind the
    removed 'purpose' dropdown). The platform suggests, the user confirms.
    """
    rec: list[str] = ["textstat", "token_count", "sentiment",
                      "safety", "profanity", "banned_keywords", "refusal"]
    if task_type in ("qa", "summarization"):
        rec += ["overlap", "answer_relevance"]
    if rag:
        rec += ["faithfulness", "context_relevance", "retrieval_hit_rate",
                "context_recall"]
    if task_type == "structured_output":
        rec += ["json_validation"]
    if task_type == "text_to_sql":
        rec += ["sql_validation"]
    # dedupe, keep order
    seen: set[str] = set()
    return [k for k in rec if not (k in seen or seen.add(k))]


__all__ = ["Evaluator", "EvaluatorInfo", "REGISTRY", "register",
           "available_evaluators", "recommended_for"]

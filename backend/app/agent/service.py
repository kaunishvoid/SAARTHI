"""A conservative, source-aware router for the currently supported SAARTHI modules."""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from typing import Any


_CROP_TERMS = re.compile(r"\b(crop|crops|plant|plants|sow|cultivat|grow|planting|seed)\b", re.I)
_FERTILIZER_TERMS = re.compile(r"\b(fertilizer|fertiliser|nutrient|manure|neutra|boost)\b", re.I)
_UNSUPPORTED_TERMS = re.compile(
    r"\b(disease|diseases|pest|infection|symptom|scheme|subsidy|weather|forecast|rain today|"
    r"market price|mandi|price today|appointment|officer|voice|speech|treatment|pesticide|diagnos(?:e|is))\b", re.I
)
_FOLLOW_UP_TERMS = re.compile(r"\b(it|that|this|these|those|why|how about|what about|and then|next)\b", re.I)
_EXPLANATION_TERMS = re.compile(r"^\s*(why|how did|what made|which inputs|what factors)", re.I)


class ConversationStore:
    """Small process-local context store; intentionally not durable or a source of facts."""

    def __init__(self, max_conversations: int = 500, ttl_seconds: int = 60 * 60) -> None:
        self.max_conversations = max_conversations
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, conversation_id: str) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            item = self._items.get(conversation_id, {"intent": None, "updated": now})
            self._items[conversation_id] = item
            self._items.move_to_end(conversation_id)
            return dict(item)

    def set_intent(self, conversation_id: str, intent: str | None) -> None:
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            self._items[conversation_id] = {"intent": intent, "updated": now}
            self._items.move_to_end(conversation_id)
            while len(self._items) > self.max_conversations:
                self._items.popitem(last=False)

    def _expire(self, now: float) -> None:
        expired = [key for key, value in self._items.items() if now - value["updated"] > self.ttl_seconds]
        for key in expired:
            self._items.pop(key, None)


class ApprovedKnowledgeRetriever:
    """Retrieval boundary for reviewed project sources; no corpus is currently installed."""

    def __init__(self, documents: list[dict[str, str]] | None = None) -> None:
        self.documents = documents or []

    def search(self, query: str) -> list[dict[str, str]]:
        terms = {term.lower() for term in re.findall(r"[a-zA-Z]{3,}", query)}
        scored = []
        for document in self.documents:
            text = f"{document.get('title', '')} {document.get('text', '')}"
            overlap = len(terms & {term.lower() for term in re.findall(r"[a-zA-Z]{3,}", text)})
            if overlap:
                scored.append((overlap, document))
        return [document for _, document in sorted(scored, key=lambda item: -item[0])[:3]]


def _prediction_summary(context: object, module: str) -> dict[str, Any] | None:
    if not isinstance(context, dict):
        return None
    result = context.get(f"{module}_prediction")
    if not isinstance(result, dict):
        return None
    if module == "crop":
        crop = result.get("predicted_crop")
        top_3 = result.get("top_3")
        if not isinstance(crop, str) or not isinstance(top_3, list):
            return None
        rows = [
            {"crop": row.get("crop"), "probability": row.get("probability")}
            for row in top_3[:3]
            if isinstance(row, dict) and isinstance(row.get("crop"), str)
        ]
        return {"predicted_crop": crop, "confidence": result.get("confidence"), "top_3": rows}
    fertilizer = result.get("fertilizer")
    top_3 = result.get("top_3")
    if not isinstance(fertilizer, str) or not isinstance(top_3, list):
        return None
    rows = [
        {"fertilizer": row.get("fertilizer"), "probability": row.get("probability")}
        for row in top_3[:3]
        if isinstance(row, dict) and isinstance(row.get("fertilizer"), str)
    ]
    return {"fertilizer": fertilizer, "confidence": result.get("confidence"), "top_3": rows}


class ConsultancyService:
    def __init__(
        self,
        conversations: ConversationStore | None = None,
        retriever: ApprovedKnowledgeRetriever | None = None,
    ) -> None:
        self.conversations = conversations or ConversationStore()
        self.retriever = retriever or ApprovedKnowledgeRetriever()

    def respond(self, message: str, conversation_id: str, context: object) -> dict[str, Any]:
        retrieved_documents = self.retriever.search(message)
        previous_intent = self.conversations.get(conversation_id).get("intent")
        if _UNSUPPORTED_TERMS.search(message):
            intent = "unsupported"
            reply = (
                "I can’t provide a verified answer for that request in this version of SAARTHI. "
                "Disease analysis, schemes, live weather, market prices, appointments, and pest guidance "
                "do not have supported data or services connected here."
            )
            used_context = None
            sources: list[dict[str, str]] = []
        else:
            crop_match = bool(_CROP_TERMS.search(message))
            fertilizer_match = bool(_FERTILIZER_TERMS.search(message))
            if crop_match and fertilizer_match:
                intent = "crop_and_fertilizer"
            elif crop_match:
                intent = "crop"
            elif fertilizer_match:
                intent = "fertilizer"
            elif _FOLLOW_UP_TERMS.search(message) and previous_intent in {"crop", "fertilizer", "crop_and_fertilizer"}:
                intent = previous_intent
            else:
                intent = "knowledge_unavailable"

            if intent in {"crop", "crop_and_fertilizer"}:
                crop_result = _prediction_summary(context, "crop")
            else:
                crop_result = None
            if intent in {"fertilizer", "crop_and_fertilizer"}:
                fertilizer_result = _prediction_summary(context, "fertilizer")
            else:
                fertilizer_result = None

            used_context = {}
            if crop_result:
                used_context["crop_prediction"] = crop_result
            if fertilizer_result:
                used_context["fertilizer_prediction"] = fertilizer_result
            if not used_context:
                used_context = None

            if _EXPLANATION_TERMS.search(message) and (crop_result or fertilizer_result):
                reply = (
                    "The available SAARTHI inference outputs provide predicted classes and scores, "
                    "but no validated feature-level explanation. I can’t identify which inputs caused "
                    "this result from the artifacts currently available."
                )
            elif intent == "crop" and crop_result:
                reply = f"The SAARTHI crop ANN ranks {crop_result['predicted_crop']} first for the submitted field values. This is a model prediction based on the supplied dataset, not a guaranteed outcome."
            elif intent == "fertilizer" and fertilizer_result:
                reply = f"NEUTRA-BOOST predicts the fertilizer class {fertilizer_result['fertilizer']} for the submitted values. It is a dataset-based class prediction, not a dosage recommendation or guaranteed outcome."
            elif intent == "crop_and_fertilizer" and (crop_result or fertilizer_result):
                details = []
                if crop_result:
                    details.append(f"crop ANN ranks {crop_result['predicted_crop']} first")
                if fertilizer_result:
                    details.append(f"NEUTRA-BOOST predicts {fertilizer_result['fertilizer']}")
                reply = "Available SAARTHI results: " + "; ".join(details) + ". These are model predictions; the fertilizer output is not dosage advice."
            elif intent == "crop":
                reply = "I can summarize a crop ANN result after you submit the seven field measurements in Crop Recommendation. I don’t have a prediction result attached to this message."
            elif intent == "fertilizer":
                reply = "I can summarize a NEUTRA-BOOST result after you submit its complete 19-field input form. I don’t have a prediction result attached to this message. The output is a fertilizer class, not dosage advice."
            elif intent == "crop_and_fertilizer":
                reply = "I can summarize crop and fertilizer results once you run those SAARTHI modules. No prediction results were attached to this message."
            elif intent in {"crop", "fertilizer", "crop_and_fertilizer"} and _FOLLOW_UP_TERMS.search(message):
                reply = "I can repeat the earlier module context, but this conversation does not contain a current prediction result. Please run the relevant SAARTHI module first."
            else:
                intent = "knowledge_unavailable"
                reply = (
                    "I don’t have a verified SAARTHI knowledge source for that agricultural question. "
                    "This assistant currently uses only crop ANN and NEUTRA-BOOST results you provide; "
                    "it does not infer general farming facts."
                )
                if retrieved_documents:
                    intent = "project_knowledge"
                    excerpts = [
                        f"{document.get('title', 'Project source')}: {document.get('text', '')[:500]}"
                        for document in retrieved_documents
                    ]
                    reply = "Relevant approved project material:\n" + "\n".join(excerpts)
            sources = []
            if crop_result:
                sources.append({"type": "model_result", "id": "saarthi_crop_ann"})
            if fertilizer_result:
                sources.append({"type": "model_result", "id": "saarthi_neutra_boost"})
            sources.extend(
                {"type": "project_document", "id": document.get("id", "approved_project_source")}
                for document in retrieved_documents
            )

        self.conversations.set_intent(conversation_id, intent if intent in {"crop", "fertilizer", "crop_and_fertilizer"} else None)
        return {
            "conversation_id": conversation_id,
            "intent": intent,
            "response": reply,
            "used_context": used_context,
            "sources": sources,
            "knowledge_available": bool(sources),
            "knowledge_retrieval": {
                "status": "available" if retrieved_documents else "unavailable",
                "source_count": len(retrieved_documents),
                "reason": None if retrieved_documents else "No approved project knowledge corpus is configured.",
            },
            "limitations": ["No verified agricultural knowledge corpus or external data services are connected."],
        }

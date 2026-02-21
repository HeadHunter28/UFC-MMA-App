"""
LLM Service Module.

Provides integration with Groq's LLM API for natural language generation.
"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    GROQ_API_KEY,
    LLM_ENABLED,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)
from services.cache_service import CacheService

logger = logging.getLogger(__name__)


class LLMService:
    """
    Groq LLM integration for natural language generation.

    Provides methods for generating fighter analyses, prediction explanations,
    and answering user questions about UFC data.
    """

    def __init__(self):
        """Initialize the LLM service."""
        self.enabled = LLM_ENABLED and bool(GROQ_API_KEY)
        self.model = LLM_MODEL
        self.max_tokens = LLM_MAX_TOKENS
        self.temperature = LLM_TEMPERATURE
        self.cache = CacheService()
        self.client = None

        if self.enabled:
            try:
                from groq import Groq
                self.client = Groq(api_key=GROQ_API_KEY)
                logger.info(f"LLM service initialized with model: {self.model}")
            except ImportError:
                logger.warning("Groq package not installed. LLM features disabled.")
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self.enabled = False
        else:
            logger.info("LLM service disabled (no API key or disabled in config)")

    def is_available(self) -> bool:
        """
        Check if the LLM service is available.

        Returns:
            True if the service can generate responses
        """
        return self.enabled and self.client is not None

    def _generate(
        self,
        prompt: str,
        cache_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Optional[str]:
        """
        Generate a response from the LLM.

        Args:
            prompt: The prompt to send
            cache_key: Optional key for caching the response
            max_tokens: Override default max tokens
            temperature: Override default temperature

        Returns:
            Generated text or None on failure
        """
        if not self.is_available():
            return None

        # Check cache first
        if cache_key:
            cached = self.cache.get(cache_key, cache_type="llm")
            if cached:
                logger.debug(f"LLM cache hit: {cache_key[:20]}...")
                return cached

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
            )

            result = response.choices[0].message.content

            # Cache the result
            if cache_key and result:
                self.cache.set(cache_key, result, cache_type="llm")

            return result

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None

    def generate_fighter_analysis(
        self,
        fighter: Dict[str, Any],
        stats: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[str]:
        """
        Generate a natural language analysis of a fighter.

        Args:
            fighter: Fighter data dict
            stats: Optional fighter statistics
            history: Optional fight history

        Returns:
            Analysis text or None
        """
        if not self.is_available():
            return None

        name = fighter.get("name", "Unknown")
        record = f"{fighter.get('wins', 0)}-{fighter.get('losses', 0)}-{fighter.get('draws', 0)}"

        # Build stats section
        stats_text = ""
        if stats:
            stats_text = f"""
Statistics:
- Significant Strikes Landed/Min: {stats.get('sig_strikes_landed_per_min', 'N/A')}
- Striking Accuracy: {stats.get('sig_strike_accuracy', 'N/A')}
- Takedowns/15min: {stats.get('takedowns_avg_per_15min', 'N/A')}
- Takedown Defense: {stats.get('takedown_defense', 'N/A')}
- Submission Average: {stats.get('submissions_avg_per_15min', 'N/A')}
"""

        # Build history section
        history_text = ""
        if history:
            recent = history[:5]  # Last 5 fights
            history_lines = []
            for fight in recent:
                result = fight.get("result", "?")
                opponent = fight.get("opponent_name", "Unknown")
                method = fight.get("method", "")
                history_lines.append(f"- {result} vs {opponent} ({method})")
            history_text = f"\nRecent Fights:\n" + "\n".join(history_lines)

        prompt = f"""You are a UFC analyst. Analyze this fighter based on their statistics:

Fighter: {name}
Record: {record}
Height: {fighter.get('height_cm', 'N/A')} cm
Reach: {fighter.get('reach_cm', 'N/A')} cm
Stance: {fighter.get('stance', 'N/A')}
{stats_text}
{history_text}

Provide a 2-3 paragraph analysis covering:
1. Fighting style and strengths
2. Areas of concern or weaknesses
3. What type of opponent gives them trouble

Keep the tone professional and analytical. Be concise."""

        cache_key = f"fighter_analysis_{fighter.get('fighter_id', name)}"
        return self._generate(prompt, cache_key=cache_key)

    def generate_prediction_explanation(
        self,
        prediction: Dict[str, Any],
        fighter_a: Dict[str, Any],
        fighter_b: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate an explanation for a fight prediction.

        Args:
            prediction: Prediction results dict
            fighter_a: Fighter A data
            fighter_b: Fighter B data

        Returns:
            Explanation text or None
        """
        if not self.is_available():
            return None

        winner_name = fighter_a["name"] if prediction.get("predicted_winner_id") == fighter_a.get("fighter_id") else fighter_b["name"]
        confidence = prediction.get("winner_confidence", 0) * 100

        # Determine predicted method
        method_probs = {
            "KO/TKO": prediction.get("method_ko_prob", 0),
            "Submission": prediction.get("method_sub_prob", 0),
            "Decision": prediction.get("method_dec_prob", 0),
        }
        predicted_method = max(method_probs, key=method_probs.get)

        # Get top factors
        top_factors = prediction.get("top_factors", "No specific factors available")

        prompt = f"""Explain this UFC fight prediction in 2-3 paragraphs:

Fighter A: {fighter_a.get('name')} ({fighter_a.get('wins', 0)}-{fighter_a.get('losses', 0)})
Fighter B: {fighter_b.get('name')} ({fighter_b.get('wins', 0)}-{fighter_b.get('losses', 0)})

Prediction: {winner_name} wins by {predicted_method}
Confidence: {confidence:.0f}%
Predicted Round: {prediction.get('predicted_round', 'N/A')}

Top factors influencing this prediction:
{top_factors}

Explain WHY these factors matter and how they translate to the predicted outcome.
Be specific about the stylistic matchup. Keep it concise and analytical."""

        return self._generate(prompt)

    def answer_trends_question(
        self,
        question: str,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Answer a user question about UFC trends.

        Args:
            question: The user's question
            context_data: Optional data context to include

        Returns:
            Answer text or None
        """
        if not self.is_available():
            return None

        context_text = ""
        if context_data:
            context_text = f"\nRelevant data:\n{context_data}"

        prompt = f"""You are a UFC data analyst. Answer this question about UFC trends and statistics:

Question: {question}
{context_text}

Provide a clear, data-driven answer. If the data doesn't support a conclusion, say so.
Keep your response concise and informative."""

        return self._generate(prompt)

    def generate_comparison_summary(
        self,
        fighter_a: Dict[str, Any],
        fighter_b: Dict[str, Any],
        comparison_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Generate a summary comparing two fighters.

        Args:
            fighter_a: First fighter data
            fighter_b: Second fighter data
            comparison_data: Optional comparison statistics

        Returns:
            Comparison summary or None
        """
        if not self.is_available():
            return None

        prompt = f"""Compare these two UFC fighters:

Fighter A: {fighter_a.get('name')}
- Record: {fighter_a.get('wins', 0)}-{fighter_a.get('losses', 0)}
- Height: {fighter_a.get('height_cm', 'N/A')} cm
- Reach: {fighter_a.get('reach_cm', 'N/A')} cm
- Stance: {fighter_a.get('stance', 'N/A')}

Fighter B: {fighter_b.get('name')}
- Record: {fighter_b.get('wins', 0)}-{fighter_b.get('losses', 0)}
- Height: {fighter_b.get('height_cm', 'N/A')} cm
- Reach: {fighter_b.get('reach_cm', 'N/A')} cm
- Stance: {fighter_b.get('stance', 'N/A')}

Provide a brief comparison highlighting:
1. Key physical advantages/disadvantages
2. Style matchup considerations
3. Which fighter might have an edge and why

Keep it concise - 2-3 short paragraphs."""

        return self._generate(prompt)

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the LLM service.

        Returns:
            Status dict with availability info
        """
        return {
            "available": self.is_available(),
            "enabled": self.enabled,
            "model": self.model,
            "has_api_key": bool(GROQ_API_KEY),
            "cache_stats": self.cache.get_stats("llm"),
        }

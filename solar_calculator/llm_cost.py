"""
LLM API cost accounting for R1-6.

Reviewer 1 asked us to "briefly detail the specific model version used and the
average token consumption or API cost per regional query. This proves the
system's cost-effectiveness for municipal deployment."

The dashboard makes two distinct kinds of call, and they cost very different
amounts:

* **Report generation** (``analysis.analyze_with_gemini``) -- one call per site.
  It is *multimodal*: the prompt is sent together with the site orthomosaic, and
  at 2000 x 2000 px the image dominates the token count, not the text.
* **Follow-up question** -- one call per user question in the chat panel. Text
  only, and an order of magnitude cheaper.

Token counting
--------------
``count_tokens`` on the live model is exact but needs an API key and a network
round trip. Without one, a character-based estimate is used: roughly four
characters per token for English prose, which is the usual working figure. Which
path produced a number is always recorded in ``method``, so an estimate is never
mistaken for a measurement.

Pricing
-------
:data:`PRICING` carries published per-million-token rates. **They are marked
unverified and must be checked against the provider's current price list before
the figures go into the manuscript** -- model pricing changes, and a stale rate
would misstate exactly the cost-effectiveness claim the reviewer is testing.
Everything downstream takes the rate as a parameter, so re-running with a
corrected price needs no code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

#: Model the dashboard instantiates (app_solar_calculator.py, load_gemini_model).
MODEL_NAME = "gemini-3.1-flash-lite"

#: Characters per token for English prose. Standard working approximation.
CHARS_PER_TOKEN = 4.0

#: Gemini bills images in fixed-size tiles. Images at or below the tile size cost
#: one tile; larger ones are split. VERIFY both numbers against current docs.
IMAGE_TILE_PX = 768
TOKENS_PER_IMAGE_TILE = 258

#: USD per million tokens. VERIFY before publishing.
PRICING: Dict[str, Dict[str, object]] = {
    "gemini-3.1-flash-lite": {
        "input_usd_per_mtok": 0.10,
        "output_usd_per_mtok": 0.40,
        "verified": False,
        "source": "provider price list - CONFIRM before publication",
    },
}


@dataclass(frozen=True)
class TokenCount:
    tokens: int
    method: str          # "api_exact" | "char_estimate"
    characters: int = 0

    @property
    def is_exact(self) -> bool:
        return self.method == "api_exact"


def estimate_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN) -> TokenCount:
    """Character-based token estimate. Approximate by construction."""
    n = len(text)
    return TokenCount(tokens=int(round(n / chars_per_token)),
                      method="char_estimate", characters=n)


def count_tokens_exact(text: str, api_key: str, model_name: str = MODEL_NAME) -> TokenCount:
    """Exact count from the provider. Requires an API key and network access."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    n = model.count_tokens(text).total_tokens
    return TokenCount(tokens=int(n), method="api_exact", characters=len(text))


def image_tokens(width: int, height: int,
                 tile_px: int = IMAGE_TILE_PX,
                 per_tile: int = TOKENS_PER_IMAGE_TILE) -> Dict[str, int]:
    """Token cost of one image, tiled.

    The dashboard passes the full-resolution orthomosaic, so this is the
    dominant cost of a report call and the obvious place to economise.
    """
    import math

    tiles_x = max(1, math.ceil(width / tile_px))
    tiles_y = max(1, math.ceil(height / tile_px))
    tiles = tiles_x * tiles_y
    return {"width": width, "height": height, "tiles_x": tiles_x,
            "tiles_y": tiles_y, "tiles": tiles, "tokens": tiles * per_tile}


def query_cost(
    input_tokens: int,
    output_tokens: int,
    model_name: str = MODEL_NAME,
    pricing: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Cost of a single call, in USD and IDR."""
    price = dict(pricing or PRICING.get(model_name, {}))
    pin = float(price.get("input_usd_per_mtok", 0.0))
    pout = float(price.get("output_usd_per_mtok", 0.0))
    cost_in = input_tokens / 1e6 * pin
    cost_out = output_tokens / 1e6 * pout
    total = cost_in + cost_out
    return {
        "model": model_name,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(input_tokens + output_tokens),
        "input_usd": round(cost_in, 8),
        "output_usd": round(cost_out, 8),
        "total_usd": round(total, 8),
        "pricing_verified": bool(price.get("verified", False)),
        "pricing_source": price.get("source", "unknown"),
    }


def usd_to_idr(usd: float, rate_idr_per_usd: float = 16_000.0) -> float:
    """Convert to rupiah. The rate is a parameter because it moves."""
    return usd * rate_idr_per_usd


def summarise(
    report_input_tokens: int,
    report_output_tokens: int,
    chat_input_tokens: int,
    chat_output_tokens: int,
    n_sites: int = 6,
    chats_per_site: int = 5,
    model_name: str = MODEL_NAME,
    rate_idr_per_usd: float = 16_000.0,
) -> Dict[str, object]:
    """Per-call and whole-study cost, for the manuscript sentence."""
    report = query_cost(report_input_tokens, report_output_tokens, model_name)
    chat = query_cost(chat_input_tokens, chat_output_tokens, model_name)

    total_usd = n_sites * (report["total_usd"] + chats_per_site * chat["total_usd"])
    return {
        "model": model_name,
        "report_call": report,
        "chat_call": chat,
        "assumptions": {"n_sites": n_sites, "chats_per_site": chats_per_site,
                        "idr_per_usd": rate_idr_per_usd},
        "study_total_usd": round(total_usd, 6),
        "study_total_idr": round(usd_to_idr(total_usd, rate_idr_per_usd), 2),
        "per_site_usd": round(total_usd / n_sites, 6) if n_sites else 0.0,
        "per_site_idr": round(usd_to_idr(total_usd / n_sites, rate_idr_per_usd), 2) if n_sites else 0.0,
        "pricing_verified": report["pricing_verified"],
    }

#!/usr/bin/env python
"""
Measure the dashboard's LLM token consumption and cost (R1-6).

    python run_llm_cost.py                     # character-based estimate, offline
    python run_llm_cost.py --api-key KEY       # exact counts from the provider
    python run_llm_cost.py --usd-per-mtok-in 0.10 --usd-per-mtok-out 0.40

Rebuilds the two prompts the dashboard actually sends -- the multimodal report
prompt from `analysis.analyze_with_gemini`, and the text-only chat prompt from
the Ask-a-Question panel -- using the study's own figures, then reports tokens
and cost per call and for the whole study.
"""

import argparse
import json
import os
import sys

from solar_calculator.llm_cost import (
    CHARS_PER_TOKEN, MODEL_NAME, PRICING, count_tokens_exact, estimate_tokens,
    image_tokens, summarise, usd_to_idr,
)

# Figures from the Yogyakarta PV site, so the prompt is the size it really is.
SITE = {
    "n_roofs": 34, "roof_area_m2": 17646.76, "n_panels": 25, "panel_area_m2": 443.04,
    "daily_kwh": 299.1, "annual_mwh": 109.2, "consumption_daily_kwh": 153.0,
    "coverage_pct": 195.5, "usable_area_m2": 6281.5, "suitability": "Low",
}


def build_report_prompt():
    """The prompt analyze_with_gemini sends, with real numbers substituted."""
    import inspect
    from solar_calculator import analysis

    src = inspect.getsource(analysis.analyze_with_gemini)
    start = src.index('prompt = f"""') + len('prompt = f"""')
    end = src.index('"""', start)
    template = src[start:end]
    # The template is an f-string over local names; strip the braces to get text
    # of the right length without needing to reproduce every variable.
    import re
    return re.sub(r"\{[^{}]*\}", lambda m: "0.00", template)


CHAT_PROMPT = """
You are a solar energy expert assistant for EcoPower Roof.
Answer based on this analysis data:

Detection  : {n_roofs} roofs ({roof_area_m2:.1f} m2), {n_panels} panels ({panel_area_m2:.1f} m2)
Production : {daily_kwh:.2f} kWh/day | {annual_mwh:.2f} MWh/yr
Consumption: {consumption_daily_kwh:.2f} kWh/day
Coverage   : {coverage_pct:.1f}%
Usable area: {usable_area_m2:.1f} m2 ({suitability} suitability)
District   : High zone

Economics (to reach 100% self-sufficiency):
  Investment : Rp 0.680B
  Payback    : 8.8 years
  NPV        : Rp 0.512B
  ROI 25yr   : 183.0%

Best scenario: 100% utilization (score 0.812)

User question: How many panels do I need for full coverage?
"""


def main():
    ap = argparse.ArgumentParser(description="R1-6 LLM token and cost accounting.")
    ap.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"),
                    help="if given, token counts are exact instead of estimated")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--image-width", type=int, default=2000)
    ap.add_argument("--image-height", type=int, default=2000)
    ap.add_argument("--report-output-tokens", type=int, default=1200,
                    help="typical length of the six-section generated report")
    ap.add_argument("--chat-output-tokens", type=int, default=250)
    ap.add_argument("--n-sites", type=int, default=6)
    ap.add_argument("--chats-per-site", type=int, default=5)
    ap.add_argument("--usd-per-mtok-in", type=float, default=None)
    ap.add_argument("--usd-per-mtok-out", type=float, default=None)
    ap.add_argument("--idr-per-usd", type=float, default=16000.0)
    ap.add_argument("--out", default="output_inference/llm_cost.json")
    args = ap.parse_args()

    pricing = dict(PRICING.get(args.model, {}))
    if args.usd_per_mtok_in is not None:
        pricing["input_usd_per_mtok"] = args.usd_per_mtok_in
        pricing["verified"] = True
        pricing["source"] = "supplied on the command line"
    if args.usd_per_mtok_out is not None:
        pricing["output_usd_per_mtok"] = args.usd_per_mtok_out
    PRICING[args.model] = pricing

    report_text = build_report_prompt()
    chat_text = CHAT_PROMPT.format(**SITE)

    if args.api_key:
        try:
            rt = count_tokens_exact(report_text, args.api_key, args.model)
            ct = count_tokens_exact(chat_text, args.api_key, args.model)
        except Exception as exc:
            print("[warn] exact counting failed (%s); falling back to estimate" % exc)
            rt, ct = estimate_tokens(report_text), estimate_tokens(chat_text)
    else:
        rt, ct = estimate_tokens(report_text), estimate_tokens(chat_text)

    img = image_tokens(args.image_width, args.image_height)
    report_in = rt.tokens + img["tokens"]

    print()
    print("R1-6  LLM API COST")
    print("=" * 72)
    print("  model                : %s" % args.model)
    print("  token counting       : %s%s" % (
        rt.method, "" if rt.is_exact else "  (~%.0f chars/token)" % CHARS_PER_TOKEN))
    print()
    print("  REPORT CALL  (one per site, multimodal)")
    print("    prompt text        : %6d chars -> %6d tokens" % (rt.characters, rt.tokens))
    print("    image %dx%d       : %d tiles of %d px -> %6d tokens"
          % (img["width"], img["height"], img["tiles"], 768, img["tokens"]))
    print("    input total        : %6d tokens   (image is %.0f%% of it)"
          % (report_in, 100.0 * img["tokens"] / report_in))
    print("    output (assumed)   : %6d tokens" % args.report_output_tokens)
    print()
    print("  CHAT CALL  (one per question, text only)")
    print("    prompt text        : %6d chars -> %6d tokens" % (ct.characters, ct.tokens))
    print("    output (assumed)   : %6d tokens" % args.chat_output_tokens)
    print()

    s = summarise(report_in, args.report_output_tokens, ct.tokens, args.chat_output_tokens,
                  n_sites=args.n_sites, chats_per_site=args.chats_per_site,
                  model_name=args.model, rate_idr_per_usd=args.idr_per_usd)

    print("  COST  (input $%.2f /Mtok, output $%.2f /Mtok)" % (
        pricing.get("input_usd_per_mtok", 0), pricing.get("output_usd_per_mtok", 0)))
    print("    report call        : $%.6f   (Rp %,.0f)".replace(",", "")
          % (s["report_call"]["total_usd"], usd_to_idr(s["report_call"]["total_usd"], args.idr_per_usd)))
    print("    chat call          : $%.6f   (Rp %.0f)"
          % (s["chat_call"]["total_usd"], usd_to_idr(s["chat_call"]["total_usd"], args.idr_per_usd)))
    print("    per site (1 report + %d chats) : $%.6f  (Rp %.0f)"
          % (args.chats_per_site, s["per_site_usd"], s["per_site_idr"]))
    print("    whole study (%d sites)          : $%.4f   (Rp %.0f)"
          % (args.n_sites, s["study_total_usd"], s["study_total_idr"]))
    print()
    if not s["pricing_verified"]:
        print("  *** PRICING NOT VERIFIED ***")
        print("  %s" % pricing.get("source", ""))
        print("  Confirm the current rate and re-run with --usd-per-mtok-in/--out")
        print("  before these numbers go into the manuscript.")
    print()
    print("  Note: the image is %.0f%% of the report call's input. Downscaling the"
          % (100.0 * img["tokens"] / report_in))
    print("  orthomosaic before sending it would cut that cost sharply with no")
    print("  loss to the narrative, since the model is summarising statistics.")

    s["report_prompt_chars"] = rt.characters
    s["report_prompt_tokens_text_only"] = rt.tokens
    s["image"] = img
    s["chat_prompt_chars"] = ct.characters
    s["token_method"] = rt.method
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, default=str)
    print()
    print("Written: %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

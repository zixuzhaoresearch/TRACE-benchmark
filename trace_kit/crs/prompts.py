"""Prompt templates and formatting utilities for CRS dialogue generation."""

from typing import List, Optional

from trace_kit.datasets.yelp.models import YelpPOI


def format_poi_card(
    poi: YelpPOI,
    max_reviews: int = 5,
    max_review_chars: int = 2000,
) -> str:
    """Format a POI into a structured card for the LLM prompt.

    Each review is labeled [R1], [R2], etc. for reference tracking.
    Reviews are selected by usefulness and length, then truncated at
    sentence boundaries if they exceed max_review_chars.

    Args:
        poi: The POI to format
        max_reviews: Maximum number of reviews to include in the card
        max_review_chars: Maximum characters per review text
    """
    lines = [
        f"=== POI: {poi.name} ===",
        f"ID: {poi.business_id}",
        f"Type: {poi.poi_type}",
        f"Rating: {poi.stars}/5 ({poi.review_count} reviews)",
        f"Address: {poi.address}, {poi.city}",
        f"Categories: {', '.join(poi.categories)}",
    ]

    if poi.price_range:
        lines.append(f"Price: {poi.price_range}")

    # Format selected attributes
    attr_display = _format_attributes(poi.attributes)
    if attr_display:
        lines.append(f"Features: {attr_display}")

    # Select top reviews by usefulness and length, then format
    if poi.reviews:
        selected = sorted(
            poi.reviews,
            key=lambda r: (r.useful or 0, len(r.text)),
            reverse=True,
        )[:max_reviews]

        lines.append(f"\nSelected Reviews ({len(selected)}):")
        for i, rev in enumerate(selected, 1):
            label = f"[R{i}]"
            stars_str = f"{'*' * int(rev.stars)}"
            text = _truncate_at_sentence(rev.text, max_review_chars)
            lines.append(f"  {label} ({stars_str}, useful={rev.useful}) \"{text}\"")

    return '\n'.join(lines)


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at the last sentence boundary within max_chars.

    Preserves complete sentences so that LLM-quoted substrings remain
    valid verbatim matches against the source text.
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Find last sentence-ending punctuation
    for sep in ['. ', '! ', '? ']:
        idx = truncated.rfind(sep)
        if idx > max_chars // 2:  # only if we keep at least half
            return truncated[:idx + 1]

    # Fallback: truncate at last space to avoid mid-word cut
    idx = truncated.rfind(' ')
    if idx > max_chars // 2:
        return truncated[:idx] + '...'
    return truncated + '...'


def format_poi_cards(pois: List[YelpPOI]) -> str:
    """Format multiple POIs into a card block."""
    cards = [format_poi_card(poi) for poi in pois]
    return '\n\n'.join(cards)


def _format_attributes(attrs: dict) -> str:
    """Format business attributes into readable string."""
    parts = []
    nice_names = {
        'OutdoorSeating': 'Outdoor Seating',
        'RestaurantsReservations': 'Reservations',
        'RestaurantsDelivery': 'Delivery',
        'RestaurantsTakeOut': 'Takeout',
        'GoodForKids': 'Kid-Friendly',
        'WiFi': 'WiFi',
        'HasTV': 'TV',
        'BikeParking': 'Bike Parking',
    }

    for key, nice in nice_names.items():
        val = attrs.get(key)
        if val and str(val).lower() in ('true', "'true'"):
            parts.append(nice)

    # Noise level
    noise = attrs.get('NoiseLevel')
    if noise:
        noise_clean = str(noise).strip("'\"u").replace("u'", "").replace("'", "")
        if noise_clean in ('quiet', 'average', 'loud', 'very_loud'):
            parts.append(f"Noise: {noise_clean}")

    # Alcohol
    alcohol = attrs.get('Alcohol')
    if alcohol:
        alc_clean = str(alcohol).strip("'\"u").replace("u'", "").replace("'", "")
        if alc_clean not in ('none', 'None'):
            parts.append(f"Alcohol: {alc_clean}")

    return ', '.join(parts)


# The main dialogue generation guidelines (v1, kept for backward compatibility)
DIALOGUE_GUIDELINES = """
You are generating a multi-turn conversational recommendation dialogue between a USER and a SYSTEM (recommendation assistant).

## RULES

1. **Turn Count**: Generate 4-8 turns total (a turn = one message from either role).
2. **Turn Alternation**: Strictly alternate user/system. Always start with user, end with system.
3. **Required Actions**:
   - At least 1 turn where SYSTEM asks about preferences (action: ask_preference)
   - At least 2 different POIs recommended across the dialogue
   - At least 1 turn with comparison or contrast between POIs
4. **Review Grounding**: When SYSTEM recommends a POI, it MUST quote EXACT text from the provided reviews. Use the review label (e.g., [R1]) and include the EXACT quote within double quotes. You may quote a substring of the review, but it must be a verbatim substring — do NOT paraphrase.
5. **Natural Flow**: The conversation should feel natural, not scripted. The user may:
   - Express preferences, constraints, or dislikes
   - Ask follow-up questions about a recommendation
   - Request alternatives or comparisons
   - Accept or reject suggestions
6. **POI References**: Only recommend POIs from the provided candidate list. Track which POI IDs and review IDs are referenced in each turn.
7. **Persona Consistency**: The user's language and needs should match the scenario persona.

## ACTION TYPES

For each turn, assign ONE action:
- `greet_and_seek`: User initiates with a greeting and states their need
- `ask_preference`: System asks about user's preferences/constraints
- `state_preference`: User states preferences, constraints, budget, etc.
- `recommend`: System recommends one or more POIs with review-grounded details
- `compare`: System compares two or more POIs
- `follow_up`: User asks for more details or alternatives
- `accept`: User accepts a recommendation
- `reject_and_refine`: User rejects and provides more constraints
- `farewell`: System wraps up the conversation

## OUTPUT FORMAT

Return a JSON object with this structure:
{
  "turns": [
    {
      "turn_id": 0,
      "role": "user",
      "text": "...",
      "action": "greet_and_seek",
      "referenced_poi_ids": [],
      "referenced_review_ids": []
    },
    {
      "turn_id": 1,
      "role": "system",
      "text": "...",
      "action": "ask_preference",
      "referenced_poi_ids": [],
      "referenced_review_ids": []
    }
  ]
}

When referencing reviews in the text, use the format: One reviewer noted: "exact quote from review text"
In referenced_review_ids, use the format: "{business_id}::R{n}" where n is the review number from the POI card.
"""


# V2 template with variable turn count, stronger grounding, forced rejection
DIALOGUE_GUIDELINES_TEMPLATE = """
You are generating a multi-turn conversational recommendation dialogue between a USER and a SYSTEM (recommendation assistant).

## RULES

1. **Turn Count**: Generate {min_turns}-{max_turns} turns total (a turn = one message from either role). Vary the length naturally — NOT every dialogue should be the same length.
2. **Turn Alternation**: Strictly alternate user/system. Always start with user, end with system.
3. **Required Actions**:
   - At least 1 turn where SYSTEM asks about preferences (action: ask_preference)
   - At least 3 different POIs recommended across the dialogue
   - At least 1 turn with comparison or contrast between POIs
4. **Review Grounding**: When SYSTEM recommends a POI, it MUST quote EXACT substrings from the provided reviews. Use the review label (e.g., [R1]) and include the EXACT quote within double quotes. You MUST quote EXACT substrings from reviews — do NOT paraphrase or summarize. The quoted text must appear verbatim in the source review.
5. **Natural Flow**: The conversation should feel natural, not scripted. The user may:
   - Express preferences, constraints, or dislikes
   - Ask follow-up questions about a recommendation
   - Request alternatives or comparisons
   - Accept or reject suggestions
6. **Rejection Required**: The user should NOT always accept recommendations. Include at least one turn where the user rejects a suggestion or requests alternatives (action: reject_and_refine). The rejection should be motivated by the persona's preferences.
7. **POI References**: Only recommend POIs from the provided candidate list. Track which POI IDs and review IDs are referenced in each turn.
8. **Persona Consistency**: The user's language and needs should match the scenario persona.
9. **CRITICAL POI ID Rule**: The `referenced_poi_ids` and review ID prefixes MUST be copied EXACTLY from the AVAILABLE POI IDs table. A single wrong character will cause validation failure. Do NOT retype these IDs — copy them verbatim.

## ACTION TYPES

For each turn, assign ONE action:
- `greet_and_seek`: User initiates with a greeting and states their need
- `ask_preference`: System asks about user's preferences/constraints
- `state_preference`: User states preferences, constraints, budget, etc.
- `recommend`: System recommends one or more POIs with review-grounded details
- `compare`: System compares two or more POIs
- `follow_up`: User asks for more details or alternatives
- `accept`: User accepts a recommendation
- `reject_and_refine`: User rejects and provides more constraints
- `farewell`: System wraps up the conversation

## OUTPUT FORMAT

Return a JSON object with this structure:
{{
  "turns": [
    {{
      "turn_id": 0,
      "role": "user",
      "text": "...",
      "action": "greet_and_seek",
      "referenced_poi_ids": [],
      "referenced_review_ids": []
    }},
    {{
      "turn_id": 1,
      "role": "system",
      "text": "...",
      "action": "ask_preference",
      "referenced_poi_ids": [],
      "referenced_review_ids": []
    }}
  ]
}}

When referencing reviews in the text, use the format: One reviewer noted: "exact quote from review text"
In referenced_review_ids, use the format: "{{business_id}}::R{{n}}" where n is the review number from the POI card.
"""


# V3 template with difficulty tiers and configurable rejection requirements
DIALOGUE_GUIDELINES_TEMPLATE_V3 = """
You are generating a multi-turn conversational recommendation dialogue between a USER and a SYSTEM (recommendation assistant).

## RULES

1. **Turn Count**: Generate {min_turns}-{max_turns} turns total (a turn = one message from either role). Vary the length naturally — NOT every dialogue should be the same length.
2. **Turn Alternation**: Strictly alternate user/system. Always start with user, end with system.
3. **Required Actions**:
   - At least 1 turn where SYSTEM asks about preferences (action: ask_preference)
   - At least 3 different POIs recommended across the dialogue
   - At least 1 turn with comparison or contrast between POIs
4. **Review Grounding**: When SYSTEM recommends a POI, it MUST quote EXACT substrings from the provided reviews. Use the review label (e.g., [R1]) and include the EXACT quote within double quotes. You MUST quote EXACT substrings from reviews — do NOT paraphrase or summarize. The quoted text must appear verbatim in the source review.
5. **Natural Flow**: The conversation should feel natural, not scripted. The user may:
   - Express preferences, constraints, or dislikes
   - Ask follow-up questions about a recommendation
   - Request alternatives or comparisons
   - Accept or reject suggestions
6. **Rejection Required**: The user MUST reject recommendations at least {min_rejections} time(s) using the action `reject_and_refine`. Each rejection should be motivated by the persona's preferences and provide new or refined constraints.
7. **POI References**: Only recommend POIs from the provided candidate list. Track which POI IDs and review IDs are referenced in each turn.
8. **Persona Consistency**: The user's language and needs should match the scenario persona.
9. **CRITICAL POI ID Rule**: The `referenced_poi_ids` and review ID prefixes MUST be copied EXACTLY from the AVAILABLE POI IDs table. A single wrong character will cause validation failure. Do NOT retype these IDs — copy them verbatim.
10. **Multi-Reference Gold (CRITICAL for benchmark fairness)**: For every SYSTEM turn whose action is `recommend` or `compare`, in addition to `referenced_poi_ids` (the POIs you actually picked), also fill in `acceptable_alternative_poi_ids` with **ONLY POIs that are NOT in `referenced_poi_ids`** but would have been equally valid choices given the user's preferences expressed *up to and including this turn*. \n   • Do NOT include any POI that is already in `referenced_poi_ids` for this turn (that's the picked one, not an alternative).\n   • Be inclusive: any other candidate POI that satisfies all stated hard constraints (price ceiling, POI type, dietary needs, location, ratings, etc.) and is consistent with the soft preferences should be listed.\n   • Leave the list empty when (a) the action is not recommend/compare, (b) the user's request is so specific that only the picked POI qualifies, or (c) all other candidates were already explicitly rejected earlier in the dialogue.\n   This list becomes the multi-reference gold against which CRS baselines will be evaluated; including the picked POI here is wasted information.
{difficulty_instruction}

## ACTION TYPES

For each turn, assign ONE action:
- `greet_and_seek`: User initiates with a greeting and states their need
- `ask_preference`: System asks about user's preferences/constraints
- `state_preference`: User states preferences, constraints, budget, etc.
- `recommend`: System recommends one or more POIs with review-grounded details
- `compare`: System compares two or more POIs
- `follow_up`: User asks for more details or alternatives
- `accept`: User accepts a recommendation
- `reject_and_refine`: User rejects and provides more constraints
- `farewell`: System wraps up the conversation

## OUTPUT FORMAT

Return a JSON object with this structure:
{{{{
  "turns": [
    {{{{
      "turn_id": 0,
      "role": "user",
      "text": "...",
      "action": "greet_and_seek",
      "referenced_poi_ids": [],
      "referenced_review_ids": [],
      "acceptable_alternative_poi_ids": []
    }}}},
    {{{{
      "turn_id": 1,
      "role": "system",
      "text": "...",
      "action": "recommend",
      "referenced_poi_ids": ["<picked_business_id>"],
      "referenced_review_ids": ["<picked_business_id>::R2"],
      "acceptable_alternative_poi_ids": ["<other_acceptable_id_1>", "<other_acceptable_id_2>"]
    }}}}
  ]
}}}}

When referencing reviews in the text, use the format: One reviewer noted: "exact quote from review text"
In referenced_review_ids, use the format: "{{{{business_id}}}}::R{{{{n}}}}" where n is the review number from the POI card.
For `acceptable_alternative_poi_ids`: include candidate POI IDs from the AVAILABLE POI IDs table (other than the ones in `referenced_poi_ids`) that would also satisfy the user's stated constraints up to this turn. Empty list when no other candidate is acceptable, or for non-recommend/compare turns.
"""

# Difficulty-specific instructions
DIFFICULTY_INSTRUCTIONS = {
    'easy': '',
    'medium': """
11. **Medium Difficulty**: The user should have moderate specificity in their preferences. They may change their mind once during the conversation.""",
    'hard': """
11. **Hard Difficulty**: Make this a challenging recommendation scenario. The user should:
    - Start with vague requirements and gradually become very specific
    - Change or add constraints mid-conversation (e.g., "actually, I also need...")
    - Challenge the system's recommendations with pointed questions
    - Show some skepticism about review quotes and ask for more evidence
    - Have conflicting preferences that make recommendation difficult (e.g., cheap but upscale ambiance)""",
    'expert': """
11. **Expert Difficulty (rare, 10% of dataset)**: Make this an extremely demanding scenario, the kind that breaks naive CRS systems. The user should:
    - Open with multiple **hard constraints simultaneously** (e.g., budget ≤$$, vegan, walkable from a specific neighborhood, ≥4.0 stars, allergy-friendly)
    - Introduce at least one explicitly **conflicting requirement pair** (e.g., "I want romantic and quiet, but also lively and good for groups")
    - Reject **at least 3 recommendations** with progressively narrower criteria, citing specific shortcomings ("the parking review you cited mentions valet only — I need self-park")
    - **Shift preferences twice mid-conversation** (e.g., "actually let's drop the price ceiling, but it has to be within 1km of my hotel")
    - Press the system on **review citation accuracy** at least once ("but that quote talks about lunch, I need dinner — find another reference")
    - Force the system to either find a non-obvious match or honestly report that no candidate fits perfectly
    - End with either grudging acceptance of a compromise OR explicit acknowledgment of unmet criteria""",
}

# Difficulty tier configurations
DIFFICULTY_CONFIGS = {
    'easy':   {'min_turns': 6,  'max_turns': 10, 'min_rejections': 1},
    'medium': {'min_turns': 8,  'max_turns': 14, 'min_rejections': 1},
    'hard':   {'min_turns': 10, 'max_turns': 16, 'min_rejections': 2},
    'expert': {'min_turns': 14, 'max_turns': 22, 'min_rejections': 3},
}


def build_generation_prompt(
    scenario_description: str,
    poi_cards_text: str,
) -> str:
    """Build the complete prompt for dialogue generation (v1)."""
    return f"""## SCENARIO

{scenario_description}

## CANDIDATE POIs

The following POIs are available for recommendation. You MUST only recommend from these. When quoting reviews, use EXACT text from the reviews below.

{poi_cards_text}

## GUIDELINES

{DIALOGUE_GUIDELINES}

Generate the dialogue now.
"""


def format_poi_id_table(pois: List[YelpPOI]) -> str:
    """Format a compact POI ID reference table for copy-paste accuracy."""
    lines = [
        "## AVAILABLE POI IDs (use EXACTLY these — copy-paste, do not retype)",
        "| # | POI ID | Name |",
        "|---|--------|------|",
    ]
    for i, poi in enumerate(pois, 1):
        lines.append(f"| {i} | `{poi.business_id}` | {poi.name} |")
    return '\n'.join(lines)


def build_generation_prompt_v2(
    scenario_description: str,
    poi_cards_text: str,
    candidate_pois: Optional[List[YelpPOI]] = None,
    min_turns: int = 6,
    max_turns: int = 12,
    error_feedback: Optional[str] = None,
) -> str:
    """Build the complete prompt for dialogue generation (v2).

    Supports variable turn counts, requires 3+ POI recommendations,
    forces at least one rejection scenario, and strengthens grounding.
    """
    guidelines = DIALOGUE_GUIDELINES_TEMPLATE.format(
        min_turns=min_turns,
        max_turns=max_turns,
    )

    # Build the POI ID mapping table if candidates are provided
    id_table_section = ""
    if candidate_pois:
        id_table_section = f"\n{format_poi_id_table(candidate_pois)}\n"

    # Build optional error feedback section
    error_section = ""
    if error_feedback:
        error_section = f"""
## PREVIOUS ATTEMPT FAILED — FIX THESE ISSUES

{error_feedback}

Please regenerate the dialogue, fixing ALL issues listed above.
"""

    return f"""## SCENARIO

{scenario_description}
{id_table_section}
## CANDIDATE POIs

The following POIs are available for recommendation. You MUST only recommend from these. When quoting reviews, use EXACT text from the reviews below.

{poi_cards_text}

## GUIDELINES

{guidelines}
{error_section}
Generate the dialogue now.
"""


def build_generation_prompt_v3(
    scenario_description: str,
    poi_cards_text: str,
    candidate_pois: Optional[List[YelpPOI]] = None,
    difficulty: str = 'medium',
    error_feedback: Optional[str] = None,
) -> str:
    """Build the complete prompt for dialogue generation (v3).

    Supports difficulty tiers (easy/medium/hard) with configurable
    turn counts and rejection requirements.
    """
    config = DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS['medium'])
    difficulty_instruction = DIFFICULTY_INSTRUCTIONS.get(difficulty, '')

    guidelines = DIALOGUE_GUIDELINES_TEMPLATE_V3.format(
        min_turns=config['min_turns'],
        max_turns=config['max_turns'],
        min_rejections=config['min_rejections'],
        difficulty_instruction=difficulty_instruction,
    )

    # Build the POI ID mapping table if candidates are provided
    id_table_section = ""
    if candidate_pois:
        id_table_section = f"\n{format_poi_id_table(candidate_pois)}\n"

    # Build optional error feedback section
    error_section = ""
    if error_feedback:
        error_section = f"""
## PREVIOUS ATTEMPT FAILED — FIX THESE ISSUES

{error_feedback}

Please regenerate the dialogue, fixing ALL issues listed above.
"""

    return f"""## SCENARIO

{scenario_description}
{id_table_section}
## CANDIDATE POIs

The following POIs are available for recommendation. You MUST only recommend from these. When quoting reviews, use EXACT text from the reviews below.

{poi_cards_text}

## GUIDELINES

{guidelines}
{error_section}
Generate the dialogue now.
"""

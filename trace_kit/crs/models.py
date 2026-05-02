"""Data models for CRS (Conversational Recommendation System) dialogues."""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


@dataclass
class ScenarioMeta:
    """Metadata describing a dialogue scenario."""
    persona_type: str  # e.g. 'solo_traveler', 'couple', 'family'
    persona_description: str  # e.g. "Mike and Sarah, late 30s, anniversary dinner"
    primary_poi_type: str  # 'restaurant', 'hotel', 'attraction'
    travel_context: str  # e.g. 'anniversary_weekend', 'business_trip'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'persona_type': self.persona_type,
            'persona_description': self.persona_description,
            'primary_poi_type': self.primary_poi_type,
            'travel_context': self.travel_context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenarioMeta':
        return cls(
            persona_type=data['persona_type'],
            persona_description=data['persona_description'],
            primary_poi_type=data['primary_poi_type'],
            travel_context=data['travel_context'],
        )


@dataclass
class DialogueTurn:
    """A single turn in a CRS dialogue."""
    turn_id: int
    role: str  # 'user' or 'system'
    text: str
    action: str  # 'greet_and_seek', 'ask_preference', 'state_preference',
                 # 'recommend', 'compare', 'follow_up', 'farewell', etc.
    referenced_poi_ids: List[str] = field(default_factory=list)
    referenced_review_ids: List[str] = field(default_factory=list)
    # Multi-reference gold: other candidate POIs that would have been
    # equally acceptable for this recommend/compare turn given the user's
    # stated preferences. Used for fair Recall@k evaluation that doesn't
    # penalize baselines for picking a different but equally valid POI.
    acceptable_alternative_poi_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'turn_id': self.turn_id,
            'role': self.role,
            'text': self.text,
            'action': self.action,
            'referenced_poi_ids': self.referenced_poi_ids,
            'referenced_review_ids': self.referenced_review_ids,
            'acceptable_alternative_poi_ids': self.acceptable_alternative_poi_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogueTurn':
        return cls(
            turn_id=data['turn_id'],
            role=data['role'],
            text=data['text'],
            action=data['action'],
            referenced_poi_ids=data.get('referenced_poi_ids', []),
            referenced_review_ids=data.get('referenced_review_ids', []),
            acceptable_alternative_poi_ids=data.get('acceptable_alternative_poi_ids', []),
        )


@dataclass
class CRSDialogue:
    """A complete CRS dialogue with grounding information."""
    dialogue_id: str
    city: str
    scenario: ScenarioMeta
    turns: List[DialogueTurn]
    referenced_pois: List[Dict[str, Any]]  # POI summaries used in this dialogue
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def recommended_poi_ids(self) -> List[str]:
        """All POI IDs referenced in recommend/compare actions."""
        ids = []
        for turn in self.turns:
            if turn.action in ('recommend', 'compare'):
                ids.extend(turn.referenced_poi_ids)
        return list(dict.fromkeys(ids))  # unique, preserve order

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dialogue_id': self.dialogue_id,
            'city': self.city,
            'scenario': self.scenario.to_dict(),
            'turns': [t.to_dict() for t in self.turns],
            'referenced_pois': self.referenced_pois,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CRSDialogue':
        return cls(
            dialogue_id=data['dialogue_id'],
            city=data['city'],
            scenario=ScenarioMeta.from_dict(data['scenario']),
            turns=[DialogueTurn.from_dict(t) for t in data['turns']],
            referenced_pois=data.get('referenced_pois', []),
            metadata=data.get('metadata', {}),
        )

    def to_json_file(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json_file(cls, path: str) -> 'CRSDialogue':
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Tourism Dialogue State (for DST-based baselines)
# ---------------------------------------------------------------------------

class TourismDialogueState(BaseModel):
    """Structured dialogue state for tourism CRS.

    Uses Pydantic BaseModel (not dataclass) for DSPy structured extraction
    compatibility.

    Maps directly to YelpPOI attribute keys where possible, enabling
    constraint-based filtering over the knowledge base.

    Updated incrementally each turn via DSPy LLM extraction.
    """
    # Price constraint: maps to YelpPOI.price_range ('$'..'$$$$')
    price_range: Optional[str] = Field(
        None,
        description="Desired price level: '$', '$$', '$$$', or '$$$$'",
    )
    # Cuisine preferences: positive list
    cuisine_prefs: List[str] = Field(
        default_factory=list,
        description="Desired cuisine types, e.g. ['Italian', 'Seafood']",
    )
    # Cuisine anti-preferences: negative list
    cuisine_avoid: List[str] = Field(
        default_factory=list,
        description="Cuisines to avoid, e.g. ['Chinese']",
    )
    # Ambiance preferences: maps to YelpPOI.attributes['Ambience']
    ambiance_prefs: List[str] = Field(
        default_factory=list,
        description="Desired ambiance qualities, e.g. ['romantic', 'quiet', 'casual']",
    )
    # Noise level: maps to YelpPOI.attributes['NoiseLevel']
    noise_level: Optional[str] = Field(
        None,
        description="Desired noise level: 'quiet', 'average', 'loud'",
    )
    # Feature requirements: maps to boolean YelpPOI.attributes
    required_features: List[str] = Field(
        default_factory=list,
        description=(
            "Required boolean features, e.g. ['OutdoorSeating', 'GoodForKids', "
            "'RestaurantsReservations', 'WiFi']"
        ),
    )
    # Location constraint: free-text, e.g. "near French Quarter"
    location_constraint: Optional[str] = Field(
        None,
        description="Free-text location constraint, e.g. 'near French Quarter'",
    )
    # POIs the user has explicitly rejected
    rejected_poi_ids: List[str] = Field(
        default_factory=list,
        description="Business IDs of POIs the user has rejected",
    )
    # POIs the user has explicitly accepted
    accepted_poi_ids: List[str] = Field(
        default_factory=list,
        description="Business IDs of POIs the user has accepted",
    )
    # Free-text attributes mentioned but not captured above
    other_preferences: List[str] = Field(
        default_factory=list,
        description="Other stated preferences, e.g. ['live music', 'good for groups']",
    )

    def merge_update(self, delta: 'TourismDialogueState') -> 'TourismDialogueState':
        """Merge an incremental state update into the current state.

        Additive for lists (union), overwrite for scalars (if delta has a value).
        """
        def _merge_list(current: List[str], update: List[str]) -> List[str]:
            seen = set(current)
            merged = list(current)
            for item in update:
                if item not in seen:
                    merged.append(item)
                    seen.add(item)
            return merged

        return TourismDialogueState(
            price_range=delta.price_range if delta.price_range is not None else self.price_range,
            cuisine_prefs=_merge_list(self.cuisine_prefs, delta.cuisine_prefs),
            cuisine_avoid=_merge_list(self.cuisine_avoid, delta.cuisine_avoid),
            ambiance_prefs=_merge_list(self.ambiance_prefs, delta.ambiance_prefs),
            noise_level=delta.noise_level if delta.noise_level is not None else self.noise_level,
            required_features=_merge_list(self.required_features, delta.required_features),
            location_constraint=delta.location_constraint if delta.location_constraint is not None else self.location_constraint,
            rejected_poi_ids=_merge_list(self.rejected_poi_ids, delta.rejected_poi_ids),
            accepted_poi_ids=_merge_list(self.accepted_poi_ids, delta.accepted_poi_ids),
            other_preferences=_merge_list(self.other_preferences, delta.other_preferences),
        )


# ---------------------------------------------------------------------------
# Difficulty Stratification (E3)
# ---------------------------------------------------------------------------

class DifficultyLevel(str, Enum):
    """Overall difficulty level for a dialogue."""
    EASY = 'easy'
    MEDIUM = 'medium'
    HARD = 'hard'


@dataclass
class DialogueDifficulty:
    """Difficulty classification for a single dialogue along 3 axes.

    Axes:
        preference_specificity: How specific are user preferences?
            vague (0-1 constraints), moderate (2-3), specific (4+)
        constraint_conflict: How hard is it to satisfy all constraints?
            Based on candidate pool match ratio.
        dialogue_complexity: How many turns/rejections before resolution?
            simple (0 rejections, <8 turns), moderate (1 rejection or 8-10),
            complex (2+ rejections or >10 turns)
    """
    dialogue_id: str
    preference_specificity: str  # 'vague', 'moderate', 'specific'
    constraint_conflict: str     # 'easy', 'medium', 'hard'
    dialogue_complexity: str     # 'simple', 'moderate', 'complex'
    overall: DifficultyLevel = DifficultyLevel.MEDIUM

    # Raw counts for debugging
    n_constraints: int = 0
    n_rejections: int = 0
    n_turns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dialogue_id': self.dialogue_id,
            'preference_specificity': self.preference_specificity,
            'constraint_conflict': self.constraint_conflict,
            'dialogue_complexity': self.dialogue_complexity,
            'overall': self.overall.value,
            'n_constraints': self.n_constraints,
            'n_rejections': self.n_rejections,
            'n_turns': self.n_turns,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogueDifficulty':
        return cls(
            dialogue_id=data['dialogue_id'],
            preference_specificity=data['preference_specificity'],
            constraint_conflict=data['constraint_conflict'],
            dialogue_complexity=data['dialogue_complexity'],
            overall=DifficultyLevel(data['overall']),
            n_constraints=data.get('n_constraints', 0),
            n_rejections=data.get('n_rejections', 0),
            n_turns=data.get('n_turns', 0),
        )


def classify_dialogue_difficulty(dialogue: 'CRSDialogue') -> DialogueDifficulty:
    """Classify a dialogue's difficulty along 3 axes.

    Args:
        dialogue: A CRSDialogue instance.

    Returns:
        DialogueDifficulty with per-axis and overall classification.
    """
    if not dialogue.turns:
        raise ValueError(f"Cannot classify difficulty for empty dialogue: {dialogue.dialogue_id}")

    # --- Axis 1: Preference specificity ---
    # Count constraint mentions in user turns
    constraint_keywords = {
        'price', 'budget', 'cheap', 'expensive', 'affordable', '$',
        'cuisine', 'italian', 'mexican', 'chinese', 'japanese', 'thai',
        'indian', 'french', 'american', 'seafood', 'vegetarian', 'vegan',
        'quiet', 'loud', 'noise', 'ambiance', 'atmosphere', 'cozy',
        'romantic', 'casual', 'fancy', 'outdoor', 'patio',
        'parking', 'reservation', 'wifi', 'kid', 'children', 'family',
        'downtown', 'near', 'close', 'walking', 'distance', 'location',
        'allergy', 'gluten', 'dairy', 'spicy', 'mild',
    }
    n_constraints = 0
    for turn in dialogue.turns:
        if turn.role != 'user':
            continue
        tokens = set(turn.text.lower().split())
        hits = tokens & constraint_keywords
        n_constraints += len(hits)

    # Deduplicate: cap at unique constraint types
    n_constraints = min(n_constraints, 15)

    if n_constraints <= 2:
        pref_spec = 'vague'
    elif n_constraints <= 5:
        pref_spec = 'moderate'
    else:
        pref_spec = 'specific'

    # --- Axis 2: Constraint conflict ---
    # Approximated by rejection count (more rejections → harder to satisfy)
    n_rejections = sum(
        1 for t in dialogue.turns if t.action == 'reject_and_refine'
    )
    # Also consider if multiple POIs were recommended (sign of search effort)
    n_recommended = len(dialogue.recommended_poi_ids)

    if n_rejections == 0 and n_recommended <= 2:
        conflict = 'easy'
    elif n_rejections <= 1 and n_recommended <= 3:
        conflict = 'medium'
    else:
        conflict = 'hard'

    # --- Axis 3: Dialogue complexity ---
    n_turns = dialogue.num_turns

    if n_rejections == 0 and n_turns < 8:
        complexity = 'simple'
    elif n_rejections <= 1 and n_turns <= 10:
        complexity = 'moderate'
    else:
        complexity = 'complex'

    # --- Overall difficulty ---
    # Map each axis to a numeric level and take the median
    # Explicit mapping for each axis value (no implicit sharing)
    pref_map = {'vague': 0, 'moderate': 1, 'specific': 2}
    conflict_map = {'easy': 0, 'medium': 1, 'hard': 2}
    complexity_map = {'simple': 0, 'moderate': 1, 'complex': 2}
    scores = [pref_map[pref_spec], conflict_map[conflict], complexity_map[complexity]]
    median_score = sorted(scores)[1]  # median of 3 values
    overall = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD][median_score]

    return DialogueDifficulty(
        dialogue_id=dialogue.dialogue_id,
        preference_specificity=pref_spec,
        constraint_conflict=conflict,
        dialogue_complexity=complexity,
        overall=overall,
        n_constraints=n_constraints,
        n_rejections=n_rejections,
        n_turns=n_turns,
    )

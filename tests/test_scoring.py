"""Sanity tests for Bradley-Terry scoring on synthetic match data."""
from __future__ import annotations

from doceval.schemas import Judgment
from doceval.scoring import quality_for_kind


def _judgment(task_id: str, a: str, b: str, judge: str, winner: str) -> Judgment:
    return Judgment(
        task_id=task_id,
        model_a=a,
        model_b=b,
        judge=judge,
        label_order="AB",
        criteria_winners={
            "legal_accuracy": winner,
            "completeness": winner,
            "faithfulness": winner,
            "formatting": winner,
            "clarity": winner,
        },
        overall_winner=winner,
    )


def test_dominant_winner_ranks_first():
    """Model 'alpha' beats everyone; should top the leaderboard."""
    judgments = [
        _judgment("t1", "alpha", "beta", "gamma", "a"),
        _judgment("t1", "alpha", "gamma", "beta", "a"),
        _judgment("t1", "beta", "gamma", "alpha", "a"),  # beta beats gamma
        _judgment("t2", "alpha", "beta", "gamma", "a"),
        _judgment("t2", "alpha", "gamma", "beta", "a"),
        _judgment("t2", "beta", "gamma", "alpha", "a"),
    ]
    task_kinds = {"t1": "generation", "t2": "generation"}
    entries = quality_for_kind(judgments, "generation", task_kinds)
    names = [e.model for e in entries]
    assert names[0] == "alpha"
    assert names[-1] == "gamma"
    assert entries[0].overall_winrate == 1.0
    assert entries[-1].overall_winrate == 0.0


def test_filters_by_kind():
    """Editing judgments must not contaminate generation scores."""
    judgments = [
        _judgment("t-gen", "alpha", "beta", "gamma", "a"),
        _judgment("t-edit", "alpha", "beta", "gamma", "b"),
    ]
    task_kinds = {"t-gen": "generation", "t-edit": "editing"}
    gen = quality_for_kind(judgments, "generation", task_kinds)
    edit = quality_for_kind(judgments, "editing", task_kinds)
    assert next(e for e in gen if e.model == "alpha").wins == 1
    assert next(e for e in edit if e.model == "beta").wins == 1


def test_ties_dont_affect_winrate():
    """All ties → 0/0 win rate by convention; BT scores stay 0."""
    j = _judgment("t1", "alpha", "beta", "gamma", "tie")
    entries = quality_for_kind([j], "generation", {"t1": "generation"})
    for e in entries:
        assert e.wins == 0
        assert e.losses == 0
        assert e.ties == 1
        assert e.overall_winrate == 0.0
        assert e.bt_score == 0.0


def test_per_criterion_winrates_independent():
    """A model can be strong on one criterion and weak on another."""
    j = Judgment(
        task_id="t1",
        model_a="alpha",
        model_b="beta",
        judge="gamma",
        label_order="AB",
        criteria_winners={
            "legal_accuracy": "a",
            "completeness": "a",
            "faithfulness": "b",
            "formatting": "b",
            "clarity": "tie",
        },
        overall_winner="a",
    )
    entries = quality_for_kind([j], "generation", {"t1": "generation"})
    alpha = next(e for e in entries if e.model == "alpha")
    assert alpha.criterion_winrates["legal_accuracy"] == 1.0
    assert alpha.criterion_winrates["formatting"] == 0.0
    assert alpha.criterion_winrates["clarity"] == 0.0  # ties don't count as wins

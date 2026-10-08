"""Integration tests for the FastAPI scoring service (real model, real rule engine)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import adshield.api.main as api_main
from adshield.api.main import create_app
from adshield.models.classifier import ModelConfig, train_and_evaluate
from adshield.models.rule_engine import RuleEngine
from tests.synthetic import labelled_traffic

VALID = {"ip": 4242, "app": 12, "device": 1, "os": 13, "channel": 245, "click_time": "2017-11-07 09:30:00"}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    raw = ModelConfig.from_yaml().model_dump()
    raw["xgboost"]["grid"] = {"max_depth": [3], "learning_rate": [0.2]}
    raw["xgboost"]["fixed"].update(n_estimators=40, early_stopping_rounds=10)
    run = train_and_evaluate(
        labelled_traffic(),
        ModelConfig.model_validate(raw),
        RuleEngine.from_yaml(),
        models_dir=tmp_path_factory.mktemp("models"),
        version="v-api-test",
        data_source="synthetic",
    )
    return run.directory


@pytest.fixture
def client(model_dir):
    with TestClient(create_app(model_dir)) as c:  # context manager runs startup/shutdown
        yield c


def bot_burst(n: int, ip: int = 777) -> list[dict]:
    return [
        {
            "ip": ip,
            "app": 3,
            "device": 1,
            "os": 13,
            "channel": 379,
            "click_time": f"2017-11-07 10:{i * 5 // 60:02d}:{i * 5 % 60:02d}",
            "record_id": f"c{i}",
        }
        for i in range(n)
    ]


# ---- /score: valid ----------------------------------------------------------------


def test_score_valid_record_returns_both_scores_and_explanations(client) -> None:
    r = client.post("/score", json=VALID)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert isinstance(body["is_flagged"], bool)
    assert 0.0 <= body["rule_engine_score"] <= 100.0  # rule score always present
    assert body["scoring_source"] == "model" and body["model_version"] == "v-api-test"
    contribs = body["top_contributing_features"]
    assert 1 <= len(contribs) <= 5
    assert all(
        set(c) == {"feature", "value", "contribution"} and isinstance(c["contribution"], float)
        for c in contribs
    )
    mags = [abs(c["contribution"]) for c in contribs]
    assert mags == sorted(mags, reverse=True)


def test_score_response_has_exactly_the_documented_fields(client) -> None:
    body = client.post("/score", json=VALID).json()
    assert set(body) == {
        "risk_score",
        "is_flagged",
        "top_contributing_features",
        "rule_engine_score",
        "rule_engine_fired_rules",
        "scoring_source",
        "model_version",
        "record_id",
    }


def test_is_flagged_matches_model_threshold(client, model_dir) -> None:
    body = client.post("/score", json=VALID).json()
    threshold = client.get("/health").json()["model_threshold"]
    assert body["is_flagged"] == (body["risk_score"] >= threshold)


def test_timezone_aware_timestamp_is_accepted(client) -> None:
    r = client.post("/score", json={**VALID, "click_time": "2017-11-07T09:30:00+05:30"})
    assert r.status_code == 200


def test_history_accumulates_across_requests(model_dir) -> None:
    with TestClient(create_app(model_dir)) as c:
        scores = [c.post("/score", json=rec).json() for rec in bot_burst(14)]
    assert scores[0]["rule_engine_score"] == 0  # a first click has no evidence
    assert scores[-1]["rule_engine_score"] >= 50  # burst + metronome + repetition
    assert "velocity_10min_burst" in scores[-1]["rule_engine_fired_rules"]
    assert scores[-1]["rule_engine_score"] > scores[3]["rule_engine_score"]


# ---- /score: malformed ------------------------------------------------------------


def _error_locs(resp) -> set[tuple]:
    return {tuple(e["loc"]) for e in resp.json()["detail"]}


def test_missing_field_is_422_naming_the_field(client) -> None:
    bad = {k: v for k, v in VALID.items() if k != "channel"}
    r = client.post("/score", json=bad)
    assert r.status_code == 422
    assert ("body", "channel") in _error_locs(r)


@pytest.mark.parametrize(
    "field, value",
    [
        ("ip", "not-a-number"),
        ("ip", -1),
        ("app", 3.5),
        ("os", "13"),  # strings are not coerced to ints
        ("channel", 2**40),
        ("click_time", "yesterday"),
        ("click_time", "1850-01-01T00:00:00"),
        ("device", None),
    ],
)
def test_bad_field_value_is_422_naming_that_field(client, field, value) -> None:
    r = client.post("/score", json={**VALID, field: value})
    assert r.status_code == 422
    assert ("body", field) in _error_locs(r)


def test_unknown_field_is_rejected(client) -> None:
    r = client.post("/score", json={**VALID, "chanel": 5})
    assert r.status_code == 422 and ("body", "chanel") in _error_locs(r)


def test_malformed_json_and_wrong_shape_are_422_not_500(client) -> None:
    assert (
        client.post("/score", content="{not json", headers={"content-type": "application/json"}).status_code
        == 422
    )
    assert client.post("/score", json=[VALID]).status_code == 422
    assert client.post("/score").status_code == 422


# ---- /score/batch ------------------------------------------------------------------


def test_batch_scores_every_record_in_order(client) -> None:
    records = bot_burst(15)
    r = client.post("/score/batch", json=records)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 15 == len(body["results"])
    assert [x["record_id"] for x in body["results"]] == [f"c{i}" for i in range(15)]
    assert all("rule_engine_score" in x and "top_contributing_features" in x for x in body["results"])
    rule_scores = [x["rule_engine_score"] for x in body["results"]]
    assert rule_scores[0] == 0 and rule_scores[-1] >= 50


def test_batch_matches_one_by_one_scoring(model_dir) -> None:
    records = bot_burst(12) + [{**VALID, "record_id": "other"}]
    with TestClient(create_app(model_dir)) as a, TestClient(create_app(model_dir)) as b:
        batch = a.post("/score/batch", json=records).json()["results"]
        single = [b.post("/score", json=rec).json() for rec in records]
    np.testing.assert_allclose([x["risk_score"] for x in batch], [x["risk_score"] for x in single], rtol=1e-6)
    assert [x["rule_engine_score"] for x in batch] == [x["rule_engine_score"] for x in single]


def test_batch_error_identifies_the_bad_record_and_field(client) -> None:
    records = bot_burst(5)
    records[3] = {**records[3], "app": -9}
    r = client.post("/score/batch", json=records)
    assert r.status_code == 422
    assert ("body", 3, "app") in _error_locs(r)


def test_empty_batch_is_rejected(client) -> None:
    assert client.post("/score/batch", json=[]).status_code == 422


def test_oversized_batch_is_rejected(model_dir, monkeypatch) -> None:
    monkeypatch.setattr(api_main, "MAX_BATCH", 10)  # limit is bound when the app is created
    with TestClient(create_app(model_dir)) as c:
        assert c.post("/score/batch", json=bot_burst(10)).status_code == 200
        assert c.post("/score/batch", json=bot_burst(11, ip=778)).status_code == 422


# ---- /health, startup, failure handling ---------------------------------------------


def test_health_reports_model_and_uptime(client) -> None:
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["scoring_mode"] == "model" and h["model_version"] == "v-api-test"
    assert h["uptime_seconds"] >= 0 and h["rules_config_hash"]
    client.post("/score", json=VALID)
    assert client.get("/health").json()["tracked_ips"] == 1


def test_model_is_loaded_once_at_startup_not_per_request(model_dir, monkeypatch) -> None:
    calls = []
    real = api_main.load_model
    monkeypatch.setattr(api_main, "load_model", lambda d: calls.append(d) or real(d))
    with TestClient(create_app(model_dir)) as c:
        for _ in range(5):
            c.post("/score", json=VALID)
        c.post("/score/batch", json=bot_burst(3))
        c.get("/health")
    assert len(calls) == 1


def test_missing_explicit_model_fails_startup_loudly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError), TestClient(create_app(tmp_path / "does-not-exist")):
        pass


def test_without_a_model_service_degrades_to_rules_and_says_so(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api_main, "MODELS_DIR", tmp_path)  # empty models directory
    monkeypatch.delenv("ADSHIELD_MODEL_DIR", raising=False)
    with TestClient(create_app()) as c:
        h = c.get("/health").json()
        assert h["status"] == "degraded" and h["scoring_mode"] == "rules_only" and h["model_version"] is None
        body = c.post("/score", json=VALID).json()
        assert body["scoring_source"] == "rules_fallback" and body["top_contributing_features"] == []
        assert body["risk_score"] == pytest.approx(body["rule_engine_score"] / 100)
        scores = [c.post("/score", json=rec).json() for rec in bot_burst(14)]
        assert scores[-1]["is_flagged"] is True


def test_internal_failure_returns_generic_500_without_leaking_details(model_dir, monkeypatch) -> None:
    with TestClient(create_app(model_dir), raise_server_exceptions=False) as c:

        def boom(*a, **k):
            raise RuntimeError("secret internal detail")

        monkeypatch.setattr(c.app.state.service.scorer, "score", boom)
        r = c.post("/score", json=VALID)
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal scoring error"}
    assert "secret" not in r.text


def test_contribution_value_is_null_when_not_measurable(client) -> None:
    # A first click has no gap history, so ip_gap_std_s is NaN internally; it must serialise as null.
    body = client.post("/score/batch?top_k=11", json=[VALID]).json()["results"][0]
    by_name = {c["feature"]: c for c in body["top_contributing_features"]}
    assert by_name["ip_gap_std_s"]["value"] is None
    assert by_name["clicks_ip_10min"]["value"] == 1


def test_top_k_query_controls_number_of_contributors(client) -> None:
    assert (
        len(
            client.post("/score/batch?top_k=11", json=[VALID]).json()["results"][0][
                "top_contributing_features"
            ]
        )
        == 11
    )
    assert (
        len(
            client.post("/score/batch?top_k=2", json=[VALID]).json()["results"][0][
                "top_contributing_features"
            ]
        )
        == 2
    )
    assert client.post("/score/batch?top_k=0", json=[VALID]).status_code == 422


def test_commit_history_false_leaves_live_state_untouched(model_dir) -> None:
    with TestClient(create_app(model_dir)) as c:
        first = c.post("/score/batch?commit_history=false", json=bot_burst(14)).json()["results"]
        assert c.get("/health").json()["tracked_ips"] == 0
        again = c.post("/score/batch?commit_history=false", json=bot_burst(14)).json()["results"]
        assert [x["risk_score"] for x in first] == [
            x["risk_score"] for x in again
        ]  # re-scoring is idempotent
        c.post("/score/batch", json=bot_burst(14))  # default commits
        assert c.get("/health").json()["tracked_ips"] == 1

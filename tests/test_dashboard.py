"""Tests for the dashboard's logic, API client, and the Streamlit page itself."""

from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import adshield.dashboard.client as client_module
from adshield.api.main import create_app
from adshield.dashboard.client import ApiError, ScoringClient
from adshield.dashboard.logic import (
    UploadError,
    describe_feature,
    parse_leads,
    read_upload,
    reason_text,
    results_table,
    score_upload,
    top_reasons,
    tradeoff_curve,
)
from adshield.features.pipeline import FEATURE_NAMES
from adshield.models.classifier import ModelConfig, train_and_evaluate
from adshield.models.rule_engine import RuleEngine
from tests.synthetic import labelled_traffic

APP_PATH = Path(__file__).parents[1] / "src/adshield/dashboard/app.py"
HEADER = "ip,app,device,os,channel,click_time"


def csv(*rows: str, header: str = HEADER) -> bytes:
    return ("\n".join([header, *rows]) + "\n").encode()


def good_row(ip: int = 1, t: str = "2017-11-07 10:00:00") -> str:
    return f"{ip},3,1,13,379,{t}"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    raw = ModelConfig.from_yaml().model_dump()
    raw["xgboost"]["grid"] = {"max_depth": [3], "learning_rate": [0.2]}
    raw["xgboost"]["fixed"].update(n_estimators=40, early_stopping_rounds=10)
    return train_and_evaluate(
        labelled_traffic(),
        ModelConfig.model_validate(raw),
        RuleEngine.from_yaml(),
        models_dir=tmp_path_factory.mktemp("models"),
        version="v-dash-test",
        data_source="synthetic",
    ).directory


@pytest.fixture
def api(model_dir):
    with TestClient(create_app(model_dir)) as tc:
        yield tc


@pytest.fixture
def scoring_client(api) -> ScoringClient:
    return ScoringClient("http://testserver", http_client=api)


# ---- reading and validating uploads ------------------------------------------------


def test_read_upload_requires_columns() -> None:
    with pytest.raises(UploadError, match="missing required column.*channel"):
        read_upload(csv("1,2,3,4,2017-11-07 10:00:00", header="ip,app,device,os,click_time"))


def test_read_upload_rejects_empty_and_oversized() -> None:
    with pytest.raises(UploadError, match="no rows"):
        read_upload(csv())
    with pytest.raises(UploadError, match="limit"):
        read_upload(csv(*[good_row()] * 10_001))
    with pytest.raises(UploadError):
        read_upload(b"")


def test_parse_leads_valid_file_with_ids_and_outcomes() -> None:
    data = csv(
        "L1,1,3,1,13,379,2017-11-07 10:00:00,0",
        "L2,2,3,1,13,379,2017-11-07 10:01:00,1",
        header="lead_id," + HEADER + ",is_attributed",
    )
    parsed, errors = parse_leads(read_upload(data))
    assert errors == [] and parsed is not None
    assert parsed.lead_ids == ["L1", "L2"] and len(parsed.records) == 2
    assert parsed.converted.tolist() == [False, True]


def test_parse_leads_numbers_rows_when_no_id_column_and_ignores_extra_columns() -> None:
    parsed, _ = parse_leads(read_upload(csv(good_row(1), good_row(2), header=HEADER)))
    assert parsed.lead_ids == ["1", "2"] and parsed.converted is None
    extra = csv("1,3,1,13,379,2017-11-07 10:00:00,whatever", header=HEADER + ",notes")
    assert parse_leads(read_upload(extra))[0] is not None


def test_parse_leads_reports_every_problem_with_row_and_field() -> None:
    data = csv(
        good_row(),
        "abc,3,1,13,379,2017-11-07 10:00:00",
        "5,3,1,13,-4,2017-11-07 10:00:00",
        "5,3,1,13,379,not-a-date",
        "5,3,1,13,,2017-11-07 10:00:00",
    )
    parsed, errors = parse_leads(read_upload(data))
    assert parsed is None
    found = {(e.row, e.field) for e in errors}
    assert found == {(3, "ip"), (4, "channel"), (5, "click_time"), (6, "channel")}  # header is row 1
    assert any("whole number" in e.message for e in errors)
    assert str(errors[0]).startswith("Row 3, 'ip'")


def test_parse_leads_ignores_unusable_outcome_column() -> None:
    data = csv("1,3,1,13,379,2017-11-07 10:00:00,maybe", header=HEADER + ",is_attributed")
    parsed, errors = parse_leads(read_upload(data))
    assert errors == [] and parsed.converted is None


# ---- scoring through the real API ---------------------------------------------------


def burst_csv(shuffled: bool) -> bytes:
    rows = [good_row(777, f"2017-11-07 10:{i * 5 // 60:02d}:{i * 5 % 60:02d}") for i in range(14)]
    if shuffled:
        rows = rows[::-1]
    return csv(*rows)


def test_results_come_back_in_file_order_even_when_file_is_not_time_sorted(scoring_client) -> None:
    parsed, _ = parse_leads(read_upload(burst_csv(shuffled=True)))
    res = score_upload(parsed, scoring_client)
    # Row 0 of the file is the LAST click in time, so it must carry the burst score.
    assert res["rule_score"].iloc[0] >= 50 and res["rule_score"].iloc[-1] == 0
    assert res["lead_id"].tolist() == [str(i) for i in range(1, 15)]


def test_scoring_an_upload_does_not_change_live_ip_history(api, scoring_client) -> None:
    parsed, _ = parse_leads(read_upload(burst_csv(False)))
    a = score_upload(parsed, scoring_client)
    b = score_upload(parsed, scoring_client)
    assert api.get("/health").json()["tracked_ips"] == 0
    pd.testing.assert_series_equal(a["risk_score"], b["risk_score"])  # re-uploading gives identical scores


def test_results_carry_full_breakdown(scoring_client) -> None:
    parsed, _ = parse_leads(read_upload(burst_csv(False)))
    res = score_upload(parsed, scoring_client)
    assert len(res["contributions"].iloc[-1]) == 11
    assert res["fired_rules"].iloc[-1]


# ---- plain language -----------------------------------------------------------------


@pytest.mark.parametrize("feature", [*FEATURE_NAMES, "cat_risk_app_channel_os"])
def test_every_model_feature_has_a_plain_language_description(feature) -> None:
    text = describe_feature(feature, 0.5)
    assert text and "_" not in text and feature not in text


def test_describe_feature_values() -> None:
    assert describe_feature("clicks_ip_10min", 14) == "14 click(s) from this IP in the last 10 minutes"
    assert "25%" in describe_feature("ip_unique_ratio", 0.25)
    assert "not enough clicks" in describe_feature("ip_gap_std_s", None)
    assert "not enough clicks" in describe_feature("ip_gap_std_s", float("nan"))


def test_top_reasons_picks_two_strongest_and_marks_direction() -> None:
    contribs = [
        {"feature": "clicks_ip_1h", "value": 3, "contribution": 0.1},
        {"feature": "clicks_ip_10min", "value": 14, "contribution": 0.9},
        {"feature": "ip_distinct_apps", "value": 1, "contribution": -0.5},
    ]
    text = top_reasons(contribs, 2)
    assert text.startswith("▲ 14 click(s) from this IP in the last 10 minutes")
    assert "▼ this IP has clicked 1 different app(s)" in text and "last hour" not in text
    assert reason_text(contribs[2]).startswith("▼")
    assert "rules only" in top_reasons([], 2)


def test_results_table_flags_by_threshold_and_sorts_by_risk() -> None:
    res = pd.DataFrame(
        {"lead_id": ["a", "b", "c"], "risk_score": [0.2, 0.9, 0.5], "contributions": [[], [], []]}
    )
    t = results_table(res, 0.5)
    assert t["Lead ID"].tolist() == ["b", "c", "a"]
    assert t["Flagged"].tolist() == ["Yes", "Yes", "No"]  # threshold is inclusive
    assert list(t.columns) == ["Lead ID", "Risk score", "Flagged", "Why (top 2 reasons)"]


# ---- threshold tradeoff ---------------------------------------------------------------


def test_tradeoff_curve_hand_computed() -> None:
    res = pd.DataFrame(
        {
            "risk_score": [0.9, 0.8, 0.7, 0.3, 0.2],
            "converted": [False, False, True, False, True],
        }
    )
    c = tradeoff_curve(res, np.array([0.0, 0.5, 0.75, 1.0]))
    assert c["flagged"].tolist() == [5, 3, 2, 0]
    assert c["pct_nonconverting_flagged"].tolist() == pytest.approx([100, 200 / 3, 200 / 3, 0])
    assert c["pct_conversions_flagged"].tolist() == pytest.approx([100, 50, 0, 0])
    assert c["conversions_flagged"].tolist() == [2, 1, 0, 0]


def test_tradeoff_curve_without_outcomes_and_monotone() -> None:
    res = pd.DataFrame({"risk_score": np.random.default_rng(0).random(200)})
    c = tradeoff_curve(res, np.linspace(0, 1, 21))
    assert "pct_conversions_flagged" not in c and c["flagged"].is_monotonic_decreasing
    assert c["flagged"].iloc[0] == 200


# ---- API client errors ------------------------------------------------------------------


def client_with(handler) -> ScoringClient:
    return ScoringClient(
        "http://x", http_client=httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    )


def test_client_explains_unreachable_service() -> None:
    def refuse(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ApiError, match="Cannot reach the scoring service"):
        client_with(refuse).health()


def test_client_maps_server_errors_to_friendly_messages() -> None:
    parsed, _ = parse_leads(read_upload(csv(good_row())))
    with pytest.raises(ApiError, match="HTTP 500"):
        client_with(lambda r: httpx.Response(500)).score_batch(parsed.records)

    def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": [{"loc": ["body", 0, "ip"], "msg": "bad"}]})

    with pytest.raises(ApiError, match=r"0\.ip: bad"):
        client_with(reject).score_batch(parsed.records)
    with pytest.raises(ApiError, match="different number"):
        client_with(lambda r: httpx.Response(200, json={"results": []})).score_batch(parsed.records)


def test_client_requests_analysis_mode_and_full_breakdown() -> None:
    seen = {}

    def spy(request):
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"results": [{}]})

    parsed, _ = parse_leads(read_upload(csv(good_row())))
    client_with(spy).score_batch(parsed.records)
    assert seen["query"] == {"commit_history": "false", "top_k": "11"}


# ---- the Streamlit page --------------------------------------------------------------------


@pytest.fixture
def page(api, monkeypatch) -> AppTest:
    monkeypatch.setattr(client_module.httpx, "Client", lambda **kw: api)  # page talks to the in-process API
    at = AppTest.from_file(str(APP_PATH), default_timeout=90)
    at.run()
    return at


def metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def test_page_prompts_for_a_file_first(page) -> None:
    assert not page.exception
    assert any("Pick a CSV" in i.value for i in page.info)


def test_page_scores_example_file_and_shows_summary(page, api) -> None:
    page.sidebar.checkbox[0].check().run()
    assert not page.exception, page.exception
    m = metrics(page)
    assert m["Leads scored"] == "467"
    assert int(m["Flagged"].replace(",", "")) > 0
    assert page.slider[0].value == pytest.approx(round(api.get("/health").json()["model_threshold"], 2))
    assert "Non-converting clicks we would flag" in m  # example file carries outcomes


def test_moving_the_slider_changes_the_flagged_count(page) -> None:
    page.sidebar.checkbox[0].check().run()
    flagged_at = {}
    for t in (0.0, 0.5, 1.0):
        page.slider[0].set_value(t).run()
        flagged_at[t] = int(metrics(page)["Flagged"].replace(",", ""))
    assert flagged_at[0.0] == 467 and flagged_at[1.0] == 0
    assert flagged_at[0.0] >= flagged_at[0.5] >= flagged_at[1.0]
    # extremes of the cost/benefit view: flag everything -> every conversion wrongly flagged
    page.slider[0].set_value(0.0).run()
    assert metrics(page)["Real conversions we would wrongly flag"] == "100.0%"


def test_page_reports_unreachable_service(monkeypatch) -> None:
    def refuse(**kw):
        raise httpx.ConnectError("refused")

    class Dead:
        def request(self, *a, **k):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(client_module.httpx, "Client", lambda **kw: Dead())
    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not at.exception and any("Cannot reach" in e.value for e in at.error)

from __future__ import annotations


import json
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from crisisstate.domain.models import Report
from crisisstate.replay.runner import ReplayRunner, load_reports_from_file
from crisisstate.storage.database import Database
from crisisstate.storage.repository import CrisisRepository


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CrisisState // 2026 Flood Ops",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Let the custom console own the full available browser width instead of
# inheriting Streamlit's centered content gutter.
st.markdown(
    """
    <style>
    [data-testid=\"stAppViewBlockContainer\"] {
        max-width: 100% !important;
        width: 100% !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    [data-testid=\"stAppViewContainer\"] .main {
        padding: 0 !important;
    }
    [data-testid=\"stHeader\"] {
        background: transparent !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data adapters — existing CrisisState production replay path.
# No frontend scoring or invented data lives here.
# ---------------------------------------------------------------------------


def enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def model_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    return {"value": value}


@st.cache_data(show_spinner=False)
def load_demo_report_data() -> list[dict[str, Any]]:
    reports = load_reports_from_file()
    serialized: list[dict[str, Any]] = []
    for report in reports:
        if hasattr(report, "model_dump"):
            serialized.append(report.model_dump(mode="json"))
        else:
            serialized.append(report.dict())
    return serialized


@st.cache_data(show_spinner=False)
def build_snapshot(report_count: int) -> dict[str, Any]:
    report_data = load_demo_report_data()

    if report_count <= 0:
        return {
            "report_count": 0,
            "incidents": [],
            "totals": {
                "claims": 0,
                "evidence": 0,
                "contradictions": 0,
                "semantic_claims": 0,
                "lexical_claims": 0,
            },
        }

    reports = [
        Report.model_validate(item)
        for item in report_data[:report_count]
    ]

    db = Database(":memory:")
    db.init_db()
    repository = CrisisRepository(db)
    runner = ReplayRunner(repository=repository)
    runner.run(reports=reports, verbose=False)

    incident_models = repository.list_incidents()
    incidents: list[dict[str, Any]] = []
    total_claims = 0
    total_evidence = 0
    contradiction_count = 0
    semantic_claims = 0
    lexical_claims = 0

    for incident in incident_models:
        incident_dict = model_to_dict(incident)
        claims = repository.list_claims_for_incident(incident.id)
        evidence = repository.list_evidence_for_incident(incident.id)
        snapshots = repository.list_snapshots_for_incident(incident.id)

        claim_rows = [model_to_dict(item) for item in claims]
        evidence_rows = [model_to_dict(item) for item in evidence]
        snapshot_rows = [model_to_dict(item) for item in snapshots]

        for claim_dict in claim_rows:
            method = str(
                enum_value(claim_dict.get("extraction_method", ""))
            ).upper()
            if method == "SEMANTIC":
                semantic_claims += 1
            elif method == "LEXICAL":
                lexical_claims += 1

        for evidence_dict in evidence_rows:
            relationship = str(
                enum_value(evidence_dict.get("link_type", ""))
            ).upper()
            if "CONTRADICT" in relationship:
                contradiction_count += 1

        total_claims += len(claim_rows)
        total_evidence += len(evidence_rows)

        incident_dict["claims"] = claim_rows
        incident_dict["evidence"] = evidence_rows
        incident_dict["snapshots"] = snapshot_rows
        incidents.append(incident_dict)

    incidents.sort(
        key=lambda item: (
            str(enum_value(item.get("current_attention_level", ""))),
            str(enum_value(item.get("current_severity", ""))),
            str(item.get("title", "")),
        ),
        reverse=True,
    )

    db.close()

    return {
        "report_count": report_count,
        "incidents": incidents,
        "totals": {
            "claims": total_claims,
            "evidence": total_evidence,
            "contradictions": contradiction_count,
            "semantic_claims": semantic_claims,
            "lexical_claims": lexical_claims,
        },
    }


@st.cache_data(show_spinner=True)
def build_replay_bundle() -> dict[str, Any]:
    report_data = load_demo_report_data()
    states = [build_snapshot(i) for i in range(len(report_data) + 1)]
    return {
        "reports": report_data,
        "states": states,
    }


bundle = build_replay_bundle()


# Escape </script> so arbitrary report text cannot terminate the data script.
payload = json.dumps(bundle, ensure_ascii=False).replace("</", "<\\/")


# ---------------------------------------------------------------------------
# Stitch-inspired custom console.
# The entire primary workspace is one HTML/JS surface so Streamlit's native
# widget spacing cannot fight the design.
# ---------------------------------------------------------------------------

APP = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CrisisState</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
  --void:#070b12;
  --surface:#0d131f;
  --surface-2:#111927;
  --surface-3:#161f2d;
  --surface-4:#1b2635;
  --line:#223047;
  --line-hi:#3e4a5d;
  --text:#eef4ff;
  --muted:#8491a7;
  --ghost:#536176;
  --cyan:#38bdf8;
  --teal:#2dd4bf;
  --teal-bright:#44e2cd;
  --red:#f87171;
  --amber:#fbbf24;
  --orange:#fb923c;
  --yellow:#facc15;
}

* { box-sizing:border-box; }
html,body { margin:0; padding:0; background:var(--void); color:var(--text); font-family:Inter,system-ui,sans-serif; }
body { min-height:100vh; overflow-x:hidden; }
button { font:inherit; }

body::before {
  content:"";
  position:fixed;
  inset:0;
  pointer-events:none;
  background:
    radial-gradient(circle at 7% -6%, rgba(56,189,248,.13), transparent 27%),
    radial-gradient(circle at 98% 0%, rgba(45,212,191,.09), transparent 25%),
    radial-gradient(circle at 50% 50%, rgba(84,80,180,.035), transparent 38%),
    linear-gradient(rgba(120,160,210,.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(120,160,210,.025) 1px, transparent 1px);
  background-size:auto,auto,auto,36px 36px,36px 36px;
  z-index:-1;
}

.console {
  width:100%;
  max-width:1760px;
  margin:0 auto;
  padding:8px 12px 20px;
}

.mono { font-family:'JetBrains Mono',monospace; }
.caps { text-transform:uppercase; letter-spacing:.09em; font-family:'JetBrains Mono',monospace; }

.topbar {
  min-height:52px;
  display:grid;
  grid-template-columns:minmax(360px,1fr) auto minmax(380px,1fr);
  align-items:center;
  gap:16px;
  padding:8px 12px;
  border:1px solid var(--line);
  border-radius:6px;
  background:linear-gradient(180deg,rgba(17,25,39,.94),rgba(10,16,26,.95));
  box-shadow:0 10px 35px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.035);
}

.brand { display:flex; align-items:center; gap:10px; min-width:0; }
.logo {
  width:30px; height:30px; display:grid; place-items:center; flex:none;
  color:var(--cyan); border:1px solid rgba(56,189,248,.35);
  border-radius:5px; background:rgba(56,189,248,.07); font-size:18px;
}
.brand-title { font-weight:800; letter-spacing:.08em; font-size:14px; white-space:nowrap; color:var(--cyan); }
.brand-sub { padding-left:10px; border-left:1px solid var(--line); color:var(--muted); font:600 10px 'JetBrains Mono'; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.mode { text-align:center; color:var(--muted); font:700 10px 'JetBrains Mono'; letter-spacing:.18em; }
.mode strong { color:var(--text); }

.readouts { display:flex; justify-content:flex-end; align-items:center; gap:12px; flex-wrap:wrap; }
.readout { display:flex; align-items:baseline; gap:5px; font:600 10px 'JetBrains Mono'; }
.readout .k { color:var(--muted); letter-spacing:.06em; }
.readout .v { color:var(--text); font-size:13px; }
.readout.danger .v { color:var(--red); }

.statusbar { display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap; margin-top:6px; }
.status-left,.status-right { display:flex; gap:7px; align-items:center; flex-wrap:wrap; }
.status-pill {
  display:inline-flex; align-items:center; gap:6px; padding:5px 8px; border-radius:4px;
  border:1px solid var(--line-hi); background:rgba(22,31,45,.88);
  color:var(--muted); font:700 10px 'JetBrains Mono'; letter-spacing:.07em;
}
.status-pill.live { color:var(--teal); border-color:rgba(45,212,191,.35); background:rgba(45,212,191,.07); }
.status-pill.alert { color:var(--red); border-color:rgba(248,113,113,.38); background:rgba(248,113,113,.07); }
.dot { width:6px; height:6px; border-radius:50%; background:var(--teal); box-shadow:0 0 9px rgba(45,212,191,.75); animation:pulse 1.8s infinite; }
@keyframes pulse {0%,100%{opacity:1}50%{opacity:.35}}

.replay {
  margin-top:7px; padding:8px 10px 7px; border:1px solid var(--line); border-radius:6px;
  background:rgba(13,19,31,.88); box-shadow:0 12px 30px rgba(0,0,0,.14);
}
.control-row { display:grid; grid-template-columns:1fr 1fr 1.2fr 1fr; gap:7px; }
.ctrl {
  border:1px solid var(--line-hi); background:linear-gradient(180deg,#1a2332,#141c29); color:#cdd6e4;
  min-height:34px; border-radius:4px; cursor:pointer; font:700 11px 'JetBrains Mono'; letter-spacing:.04em;
  transition:.15s ease; box-shadow:inset 0 1px 0 rgba(255,255,255,.025);
}
.ctrl:hover:not(:disabled) { border-color:var(--cyan); color:var(--cyan); transform:translateY(-2px); box-shadow:0 0 0 1px rgba(56,189,248,.10),0 10px 24px rgba(0,0,0,.24),0 0 20px rgba(56,189,248,.05); }
.ctrl:active:not(:disabled) { transform:translateY(0); box-shadow:inset 0 2px 8px rgba(0,0,0,.24); }
.ctrl.primary { border-color:rgba(56,189,248,.68); color:#7dd8ff; background:linear-gradient(180deg,rgba(20,66,91,.98),rgba(12,34,49,.98)); box-shadow:inset 0 1px 0 rgba(255,255,255,.04),0 0 22px rgba(56,189,248,.08); }
.ctrl.play { border-color:rgba(45,212,191,.62); color:#72f4df; background:linear-gradient(180deg,rgba(18,69,67,.98),rgba(12,40,40,.98)); box-shadow:inset 0 1px 0 rgba(255,255,255,.04),0 0 22px rgba(45,212,191,.07); }
.ctrl:disabled { opacity:.4; cursor:default; }
.track { display:flex; gap:4px; margin-top:7px; }
.seg { flex:1; height:5px; border-radius:3px; background:#273142; cursor:pointer; transition:.15s; }
.seg.done { background:linear-gradient(90deg,#38bdf8,#60a5fa); box-shadow:0 0 7px rgba(56,189,248,.16); }
.seg.current { background:var(--teal-bright); box-shadow:0 0 10px rgba(68,226,205,.45); }
.replay-meta { display:flex; justify-content:space-between; margin-top:6px; font:600 10px 'JetBrains Mono'; color:var(--muted); }
.replay-meta .live { color:var(--teal); }

.workspace {
  margin-top:8px; display:grid; grid-template-columns:300px minmax(0,1fr) 390px;
  border:1px solid var(--line); border-radius:6px; overflow:hidden;
  background:rgba(9,14,23,.70); box-shadow:0 16px 45px rgba(0,0,0,.22);
  min-height:calc(100vh - 150px);
}
.panel { min-width:0; }
.left { border-right:1px solid var(--line); background:rgba(9,15,24,.77); }
.center { background:rgba(10,16,26,.62); }
.right { border-left:1px solid var(--line); background:rgba(10,15,24,.83); }

.panel-head {
  min-height:42px; padding:10px 12px; border-bottom:1px solid var(--line);
  display:flex; align-items:center; justify-content:space-between; gap:8px;
  color:var(--muted); font:700 10px 'JetBrains Mono'; letter-spacing:.08em;
}
.badge-count { padding:3px 6px; border:1px solid var(--line-hi); border-radius:4px; color:var(--cyan); background:rgba(56,189,248,.05); }

.incidents { padding:8px; }
.incident-card {
  position:relative; padding:11px 11px 9px; margin-bottom:7px; border:1px solid var(--line); border-radius:5px;
  background:linear-gradient(145deg,#111a28,#0e1622); cursor:pointer; transition:.15s ease;
}
.incident-card:hover { border-color:#3f6880; transform:translateY(-1px); }
.incident-card.selected { border-color:rgba(56,189,248,.7); background:linear-gradient(145deg,#122333,#101b2a); box-shadow:0 0 24px rgba(56,189,248,.08); }
.incident-card.selected::before { content:""; position:absolute; top:8px; bottom:8px; left:-1px; width:3px; background:var(--cyan); border-radius:2px; box-shadow:0 0 10px rgba(56,189,248,.45); }
.inc-top { display:flex; justify-content:space-between; gap:8px; align-items:flex-start; }
.inc-id { color:var(--cyan); font:700 9px 'JetBrains Mono'; }
.inc-title { margin-top:5px; font-size:14px; line-height:1.25; font-weight:700; }
.inc-loc { margin-top:4px; color:var(--muted); font:500 10px 'JetBrains Mono'; }
.chips { display:flex; gap:5px; flex-wrap:wrap; margin-top:8px; }
.chip { padding:3px 6px; border:1px solid var(--line-hi); border-radius:4px; font:700 9px 'JetBrains Mono'; letter-spacing:.03em; }
.chip.cyan { color:var(--cyan); border-color:rgba(56,189,248,.35); background:rgba(56,189,248,.06); }
.chip.teal { color:var(--teal); border-color:rgba(45,212,191,.35); background:rgba(45,212,191,.06); }
.chip.amber { color:var(--amber); border-color:rgba(251,191,36,.35); background:rgba(251,191,36,.06); }
.chip.red { color:var(--red); border-color:rgba(248,113,113,.35); background:rgba(248,113,113,.06); }
.inc-foot { margin-top:8px; padding-top:7px; border-top:1px solid rgba(62,72,87,.65); color:var(--muted); font:600 10px 'JetBrains Mono'; }

.center-inner,.right-inner { padding:12px; }
.hero { padding:1px 0 9px; border-bottom:1px solid var(--line); }
.eyebrow { color:var(--cyan); font:700 9px 'JetBrains Mono'; letter-spacing:.1em; }
.hero h1 { margin:4px 0 2px; font-size:22px; line-height:1.14; letter-spacing:-.025em; }
.hero-loc { color:var(--muted); font:600 10px 'JetBrains Mono'; }
.hero-tags { display:flex; flex-wrap:wrap; gap:6px; margin-top:9px; }
.hero-tag { padding:5px 7px; border:1px solid var(--line-hi); border-radius:4px; background:#131d2b; color:#ced7e6; font:700 9px 'JetBrains Mono'; }
.hero-tag.red { color:var(--red); border-color:rgba(248,113,113,.38); background:rgba(248,113,113,.07); }
.hero-tag.amber { color:var(--amber); border-color:rgba(251,191,36,.38); background:rgba(251,191,36,.07); }

.section-title { margin:11px 0 7px; display:flex; justify-content:space-between; align-items:center; gap:8px; color:#cbd6e4; font:700 10px 'JetBrains Mono'; letter-spacing:.08em; text-transform:uppercase; }
.section-title span:last-child { color:var(--ghost); font-size:9px; }

.kpi-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }
.kpi-grid.bottom { grid-template-columns:repeat(2,1fr); margin-top:6px; }
.kpi { min-height:64px; padding:9px 10px; border:1px solid var(--line); border-radius:5px; background:linear-gradient(145deg,#131d2b,#111824); }
.kpi-label { color:var(--muted); font:700 8px 'JetBrains Mono'; letter-spacing:.07em; }
.kpi-val { margin-top:7px; color:var(--text); font:800 15px 'JetBrains Mono'; white-space:nowrap; }
.kpi-val.red { color:var(--red); }
.kpi-val.amber { color:var(--amber); }
.kpi-val.teal { color:var(--teal); }

.claim-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
.claim-card { padding:9px; border:1px solid var(--line); border-radius:5px; background:linear-gradient(145deg,#121b29,#0f1724); min-height:126px; }
.claim-head { display:flex; justify-content:space-between; gap:6px; align-items:flex-start; }
.field { color:var(--muted); font:700 8px 'JetBrains Mono'; letter-spacing:.08em; }
.claim-value { margin-top:10px; font:800 14px 'JetBrains Mono'; line-height:1.25; }
.claim-value.strike { color:#6c788b; text-decoration:line-through; text-decoration-thickness:1px; }
.claim-sub { margin-top:8px; color:#b9c5d5; font-size:10px; line-height:1.35; }
.claim-foot { display:flex; justify-content:space-between; gap:8px; margin-top:9px; padding-top:7px; border-top:1px solid var(--line); color:var(--muted); font:600 9px 'JetBrains Mono'; }

.evidence { border:1px solid var(--line); border-radius:5px; background:linear-gradient(145deg,#111a27,#0e1621); padding:9px; margin-bottom:6px; }
.evidence.support { border-left:2px solid var(--teal); }
.evidence.contradict { border-left:2px solid var(--red); }
.e-head { display:flex; justify-content:space-between; gap:8px; align-items:center; }
.e-ref { color:#bec9d8; font:700 10px 'JetBrains Mono'; }
.e-reason { margin-top:6px; color:#c8d2df; font-size:10px; line-height:1.42; }
.e-meta { display:flex; justify-content:space-between; gap:8px; margin-top:8px; color:var(--muted); font:600 9px 'JetBrains Mono'; }
details { margin-top:7px; }
summary { cursor:pointer; list-style:none; color:#91a0b5; font:700 9px 'JetBrains Mono'; }
summary::-webkit-details-marker { display:none; }
.raw { margin-top:6px; padding:7px; background:#080d14; border:1px solid var(--line); border-radius:4px; color:#aebacc; white-space:pre-wrap; word-break:break-word; font:500 8px/1.45 'JetBrains Mono'; max-height:220px; overflow:auto; }

.state-card { padding:10px; border:1px solid var(--line); border-radius:5px; background:linear-gradient(145deg,#131c2a,#101722); margin-bottom:6px; }
.state-card.conflict { border-color:rgba(248,113,113,.6); background:linear-gradient(145deg,rgba(67,26,29,.35),#101722); box-shadow:0 0 22px rgba(248,113,113,.06); }
.state-head { display:flex; justify-content:space-between; align-items:center; gap:8px; }
.state-key { color:var(--muted); font:700 8px 'JetBrains Mono'; letter-spacing:.07em; }
.state-value { margin-top:8px; font:800 14px 'JetBrains Mono'; }
.state-values { display:flex; gap:7px; align-items:center; flex-wrap:wrap; margin-top:9px; font:800 12px 'JetBrains Mono'; }
.arrow { color:#718096; }

.gauges { margin-top:5px; }
.gauge { margin-bottom:10px; }
.gauge-label { display:flex; justify-content:space-between; color:#bac6d6; font-size:10px; }
.gauge-label strong { font:700 10px 'JetBrains Mono'; }
.g-track { height:6px; margin-top:4px; border-radius:4px; background:#252f3f; overflow:hidden; }
.g-fill { height:100%; border-radius:4px; }

.timeline { position:relative; padding-left:14px; }
.timeline::before { content:""; position:absolute; left:4px; top:6px; bottom:6px; width:1px; background:#2c394d; }
.t-item { position:relative; margin-bottom:11px; }
.t-item::before { content:""; position:absolute; left:-13px; top:4px; width:7px; height:7px; border-radius:50%; background:var(--cyan); box-shadow:0 0 0 3px rgba(56,189,248,.12); }
.t-time { color:var(--muted); font:600 9px 'JetBrains Mono'; }
.t-field { margin-top:3px; color:#e1e9f5; font:800 10px 'JetBrains Mono'; }
.t-transition { margin-top:3px; color:#b3c0d1; font:700 10px 'JetBrains Mono'; }
.t-transition .new { color:var(--cyan); }
.t-reason { margin-top:3px; color:var(--muted); font-size:9px; line-height:1.35; }
.t-ref { margin-top:2px; color:#66758c; font:600 8px 'JetBrains Mono'; }

.report { padding:9px; border:1px solid var(--line); border-radius:5px; background:#101824; margin-bottom:6px; }
.report-head { display:flex; justify-content:space-between; gap:8px; align-items:center; }
.report-source { color:var(--cyan); font:700 9px 'JetBrains Mono'; }
.report-id { color:#c2ccda; font:700 10px 'JetBrains Mono'; }
.report-time { color:var(--muted); font:600 8px 'JetBrains Mono'; }
.report-text { margin-top:7px; padding-left:8px; border-left:2px solid #2a394d; color:#d2dbe7; font-size:10px; line-height:1.45; }
.report-meta { margin-top:7px; color:#728198; font:600 8px 'JetBrains Mono'; }

.footer { display:flex; justify-content:space-between; gap:10px; margin-top:8px; color:#60708a; font:700 8px 'JetBrains Mono'; letter-spacing:.07em; }
.footer .ready { color:var(--teal); }

.empty { padding:18px; color:var(--muted); border:1px dashed var(--line-hi); border-radius:5px; text-align:center; font:600 10px 'JetBrains Mono'; }

@media (max-width:1280px) {
  .topbar { grid-template-columns:1fr; }
  .mode { text-align:left; order:3; }
  .readouts { justify-content:flex-start; }
  .workspace { grid-template-columns:260px minmax(0,1fr) 330px; }
}
@media (max-width:980px) {
  .workspace { grid-template-columns:1fr; }
  .left,.right { border:0; border-bottom:1px solid var(--line); }
  .claim-grid { grid-template-columns:1fr; }
}
</style>
</head>
<body>
<div class="console">
  <div id="app"></div>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
let stateIndex = 0;
let selectedId = null;
let playing = false;
let timer = null;

const esc = (v) => {
  const d = document.createElement('div');
  d.textContent = v == null ? '' : String(v);
  return d.innerHTML;
};
const val = (v) => v == null ? '—' : String(v);
const up = (v) => val(v).toUpperCase();
const num = (v, d=0) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
};
function colorForLevel(v){
  const s=up(v);
  if(s.includes('CRITICAL')) return 'red';
  if(s.includes('HIGH')) return 'amber';
  if(s.includes('MODERATE')||s.includes('MEDIUM')) return 'amber';
  if(s.includes('LOW')) return 'teal';
  return '';
}
function statusColor(v){
  const s=up(v);
  if(s.includes('CONFLICT')||s.includes('CONTRADICT')) return 'red';
  if(s.includes('SUPPORT')||s.includes('ACTIVE')) return 'teal';
  if(s.includes('UNRESOLV')||s.includes('WEAK')||s.includes('AMBIG')) return 'amber';
  return 'cyan';
}
function chip(text, cls){ return `<span class="chip ${cls}">${esc(text)}</span>`; }
function statusChip(text){ return chip(text, statusColor(text)); }
function incidentCount(i){
  return (i.linked_report_ids||[]).length;
}
function claimCount(i){
  const claims=i.claims||[]; return claims.length || (i.claim_ids||[]).length;
}
function attention(i){ return up(i.current_attention_level); }
function severity(i){ return up(i.current_severity); }
function stateKeys(i){ return i.state_summary && typeof i.state_summary==='object' ? Object.keys(i.state_summary) : []; }
function tsForReport(index){
  if(index<=0) return '';
  const r=DATA.reports[index-1]; return r?.timestamp || '';
}
function formatTs(ts){
  if(!ts) return '—';
  return String(ts).replace('T',' · ').replace('Z',' UTC');
}
function conflictValues(raw){
  const s=String(raw ?? '');
  const m=s.match(/^\s*CONFLICTING\s*\((.*)\)\s*$/i);
  if(!m) return null;
  return m[1].split(/[\/,]/).map(x=>x.trim()).filter(Boolean);
}
function relation(e){ return up(e.link_type || e.evidence_type || e.type || 'LINK'); }
function factorColor(k){
  const m={severity:'var(--red)',conflict:'var(--orange)',recency:'var(--cyan)',new_evidence:'var(--teal-bright)',uncertainty:'var(--yellow)'};
  return m[String(k).toLowerCase()] || 'var(--cyan)';
}

function render(){
  const snap = DATA.states[stateIndex] || DATA.states[0];
  const incidents = snap.incidents || [];
  if(!selectedId || !incidents.some(i=>String(i.id)===String(selectedId))) selectedId = incidents[0] ? String(incidents[0].id) : null;
  const selected = incidents.find(i=>String(i.id)===String(selectedId)) || incidents[0] || null;
  const totals = snap.totals || {claims:0,evidence:0,contradictions:0};
  const app=document.getElementById('app');
  const total=DATA.reports.length;
  const currentTs=tsForReport(stateIndex);

  app.innerHTML=`
  <header class="topbar">
    <div class="brand">
      <div class="logo">≈</div>
      <div class="brand-title">CRISISSTATE // 2026 FLOOD OPS</div>
      <div class="brand-sub">Urban Flood Incident Intelligence — uncertainty-aware incident state</div>
    </div>
    <div class="mode"><strong>SITUATION</strong> · INCIDENT INTELLIGENCE CONSOLE</div>
    <div class="readouts">
      <div class="readout"><span class="k">REPORTS</span><span class="v">${snap.report_count}</span></div>
      <div class="readout"><span class="k">INCIDENTS</span><span class="v">${incidents.length}</span></div>
      <div class="readout"><span class="k">CLAIMS</span><span class="v">${totals.claims}</span></div>
      <div class="readout ${totals.contradictions?'danger':''}"><span class="k">CONFLICTS</span><span class="v">${totals.contradictions}</span></div>
      <div class="readout"><span class="k">EVIDENCE</span><span class="v">${totals.evidence}</span></div>
    </div>
  </header>

  <div class="statusbar">
    <div class="status-left">
      <span class="status-pill live"><span class="dot"></span>SYNTHETIC REPLAY ACTIVE</span>
      <span class="status-pill">◈ DETERMINISTIC ENGINE</span>
      <span class="status-pill">NOISY → CLAIMS → EVIDENCE → STATE → ATTN</span>
    </div>
    <div class="status-right">
      ${totals.contradictions ? '<span class="status-pill alert">⚠ CONFLICT SIGNALS ACTIVE</span>' : '<span class="status-pill">NO ACTIVE CONFLICT</span>'}
    </div>
  </div>

  <section class="replay">
    <div class="control-row">
      <button class="ctrl" onclick="resetReplay()">↺ RESET</button>
      <button class="ctrl" onclick="prevReplay()" ${stateIndex<=0?'disabled':''}>◀ PREV</button>
      <button class="ctrl primary" onclick="nextReplay()" ${stateIndex>=total?'disabled':''}>▶ NEXT REPORT</button>
      <button class="ctrl play" onclick="togglePlay()">${playing?'Ⅱ PAUSE':'▶ PLAY'}</button>
    </div>
    <div class="track">
      ${Array.from({length:total},(_,i)=>`<div class="seg ${i<stateIndex?'done ':''}${i===stateIndex-1&&stateIndex>0?'current':''}" onclick="jumpReplay(${i+1})" title="Report ${i+1}"></div>`).join('')}
    </div>
    <div class="replay-meta"><span>REPORT <span class="live">${stateIndex}</span> OF <span class="live">${total}</span></span><span>${currentTs ? 'last ingested at <span class="live">'+esc(currentTs)+'</span>' : 'awaiting first report'}</span></div>
  </section>

  <main class="workspace">
    <aside class="panel left">
      <div class="panel-head"><span>INCIDENT NAVIGATOR</span><span class="badge-count">${incidents.length} ACTIVE</span></div>
      <div class="incidents">
        ${incidents.length ? incidents.map((i,idx)=>`
          <div class="incident-card ${String(i.id)===String(selectedId)?'selected':''}" onclick="selectIncident('${esc(String(i.id)).replace(/'/g,"&#39;")}')">
            <div class="inc-top"><span class="inc-id">INC-${String(idx+1).padStart(2,'0')}</span><span>${String(i.id)===String(selectedId)?'●':''}</span></div>
            <div class="inc-title">${esc(i.title || 'Untitled incident')}</div>
            <div class="inc-loc">⌖ ${esc(i.location_name || 'Unknown location')}</div>
            <div class="chips">${chip(severity(i),colorForLevel(severity(i))||'cyan')}${chip(attention(i),colorForLevel(attention(i))||'cyan')}</div>
            <div class="inc-foot">${incidentCount(i)} reports · ${claimCount(i)} claims</div>
          </div>
        `).join('') : '<div class="empty">NO INCIDENT STATE YET</div>'}
      </div>
    </aside>

    <section class="panel center">
      <div class="center-inner">
        ${selected ? renderCenter(selected) : '<div class="empty">Advance the replay to establish incident state.</div>'}
      </div>
    </section>

    <aside class="panel right">
      <div class="panel-head"><span>INTELLIGENCE RAIL</span><span class="caps" style="color:var(--teal)">LIVE STATE</span></div>
      <div class="right-inner">${selected ? renderRight(selected) : '<div class="empty">No current incident.</div>'}</div>
    </aside>
  </main>

  <div class="footer"><span>CRISISSTATE · DETERMINISTIC SYNTHETIC URBAN-FLOOD REPLAY</span><span class="ready">● ENGINE READY · STATE IS TRACEABLE</span></div>
  `;
}

function renderCenter(i){
  const claims=i.claims||[];
  const evidence=i.evidence||[];
  const linked=i.linked_report_ids||[];
  const reportsById={}; DATA.reports.forEach(r=>reportsById[String(r.id)]=r);
  const snapshots=i.snapshots||[];
  const semClaims=claims.filter(c=>up(c.extraction_method)==='SEMANTIC');
  return `
    <div class="hero">
      <div class="eyebrow">SELECTED INCIDENT · ${esc(String(i.id).slice(-8).toUpperCase())}</div>
      <h1>${esc(i.title || 'Untitled incident')}</h1>
      <div class="hero-loc">⌖ ${esc(i.location_name || 'Unknown location')}</div>
      <div class="hero-tags">
        <span class="hero-tag ${colorForLevel(severity(i))}">Severity: ${esc(severity(i))}</span>
        <span class="hero-tag ${colorForLevel(attention(i))}">Attention: ${esc(attention(i))}</span>
        <span class="hero-tag ${num(i.current_confidence)>0 && num(i.current_confidence)<.25?'red':''}">Confidence: ${num(i.current_confidence).toFixed(2)}</span>
        <span class="hero-tag">Linked Reports: ${linked.length}</span>
        <span class="hero-tag">Total Claims: ${claims.length}</span>
      </div>
    </div>

    <div class="section-title"><span>Extracted Claims Intelligence</span><span>${claims.length} OBJECTS</span></div>
    <div class="claim-grid">
      ${claims.length ? claims.map(renderClaim).join('') : '<div class="empty" style="grid-column:1/-1">NO CLAIMS EXTRACTED</div>'}
    </div>

    <div class="section-title"><span>Conflict & Evidence Matrix</span><span>${evidence.length} LINKS</span></div>
    ${evidence.length ? evidence.map((e,idx)=>renderEvidence(e,idx+1)).join('') : '<div class="empty">NO EVIDENCE LINKS RECORDED YET</div>'}

    ${semClaims.length ? `
    <div class="section-title"><span>Semantic Provenance</span><span>${semClaims.length} SEMANTIC CLAIMS</span></div>
    ${semClaims.map((c,idx)=>`<details class="evidence"><summary>SEMANTIC CLAIM ${idx+1} · ${esc(readable(c.claim_type))}: ${esc(readable(c.value))}</summary><div class="e-meta"><span>EXEMPLAR</span><span>${esc(c.matched_exemplar_id||'—')}</span></div><div class="e-meta"><span>SIMILARITY</span><span>${c.similarity_score!=null?num(c.similarity_score).toFixed(3):'—'}</span></div>${c.source_span?`<div class="raw">${esc(c.source_span)}</div>`:''}</details>`).join('')}` : ''}

    <div class="section-title"><span>Ingested Field Reports</span><span>${linked.length} REPORTS</span></div>
    ${linked.length ? linked.map(id=>renderReport(reportsById[String(id)])).join('') : '<div class="empty">NO REPORTS LINKED</div>'}

    <div class="section-title"><span>State Evolution</span><span>${snapshots.length} SNAPSHOTS</span></div>
    <div class="timeline">
      ${snapshots.length ? snapshots.map(renderTimeline).join('') : '<div class="empty">NO STATE SNAPSHOTS</div>'}
    </div>
  `;
}

function renderClaim(c){
  const status=up(c.status); const method=up(c.extraction_method||''); const contrad=status.includes('CONTRADICT');
  const modality=up(c.value_modality||'ASSERTED');
  return `<article class="claim-card">
    <div class="claim-head"><span class="field">FIELD: ${esc(readable(c.claim_type))}</span>${statusChip(status)}</div>
    <div class="claim-value ${contrad?'strike':''}">${esc(readable(c.value))}</div>
    <div class="claim-sub">${c.subject?'Subject: '+esc(c.subject):''}${modality && modality!=='ASSERTED'?' · Modality: '+esc(modality):''}${method==='SEMANTIC'&&c.matched_exemplar_id?'<br><span style="color:var(--cyan)">Matched exemplar: '+esc(c.matched_exemplar_id)+' · '+num(c.similarity_score).toFixed(3)+'</span>':''}</div>
    <div class="claim-foot"><span>METHOD: ${esc(method||'—')}</span><span>CONF: ${c.confidence!=null?num(c.confidence).toFixed(2):'—'}</span></div>
  </article>`;
}

function renderEvidence(e,idx){
  const rel=relation(e); const isContr=rel.includes('CONTRADICT');
  return `<details class="evidence ${isContr?'contradict':'support'}">
    <summary><span class="chip ${isContr?'red':'teal'}">${esc(rel)}</span><span class="e-ref">Evidence ${idx} · ${esc(e.report_id||'—')} vs ${esc(e.claim_id||'—')}</span><span style="float:right;color:var(--muted)">CONF ${e.confidence!=null?num(e.confidence).toFixed(3):'—'}</span></summary>
    ${e.reason?'<div class="e-reason">'+esc(e.reason)+'</div>':''}
    <div class="e-meta"><span>RECORDED</span><span>${esc(e.created_at||e.timestamp||'—')}</span></div>
    ${e.strength_total!=null?'<div class="e-meta"><span>STRENGTH</span><span style="color:var(--teal)">'+num(e.strength_total).toFixed(3)+'</span></div>':''}
    <details><summary>RAW EVIDENCE DATA</summary><div class="raw">${esc(JSON.stringify(e,null,2))}</div></details>
  </details>`;
}

function renderReport(r){
  if(!r) return '';
  return `<details class="report">
    <summary><span class="report-source">${esc(up(r.source_type||'UNKNOWN'))}</span> <span class="report-id">${esc(r.id||'—')}</span><span class="report-time" style="float:right">${esc(r.timestamp||'—')}</span></summary>
    <div class="report-text">“${esc(r.text||'')}”</div>
    <div class="report-meta">SRC ${esc(r.source_id||'—')} · ${r.latitude!=null&&r.longitude!=null?esc(String(r.latitude)+', '+String(r.longitude)):'—'}</div>
  </details>`;
}

function renderTimeline(s){
  const prev=s.previous_value==null?'INITIAL':readable(s.previous_value);
  return `<div class="t-item">
    <div class="t-time">${esc(s.timestamp||'—')}</div>
    <div class="t-field">${esc(s.changed_field||'—')}</div>
    <div class="t-transition">${esc(prev)} <span class="arrow">→</span> <span class="new">${esc(readable(s.new_value))}</span></div>
    <div class="t-reason">${esc(s.reason||'')}</div>
    ${s.evidence_reference?'<div class="t-ref">EVIDENCE REF · '+esc(s.evidence_reference)+'</div>':''}
  </div>`;
}

function renderRight(i){
  const summary=i.state_summary||{};
  const factors=i.attention_factors||{};
  const fMap={severity:'var(--red)',conflict:'var(--orange)',recency:'var(--cyan)',new_evidence:'var(--teal-bright)',uncertainty:'var(--yellow)'};
  return `
    <div class="section-title" style="margin-top:0"><span>Current Resolved State</span><span>LIVE</span></div>
    ${Object.entries(summary).map(([k,v])=>{
      const comp=conflictValues(v);
      if(comp) return `<div class="state-card conflict"><div class="state-head"><span class="state-key">${esc(k)}</span><span class="chip red">CONFLICTING</span></div><div class="state-values">${comp.map((x,j)=>`${j?'<span class="arrow">↕</span>':''}<span>${esc(x)}</span>`).join('')}</div></div>`;
      return `<div class="state-card"><div class="state-key">${esc(k)}</div><div class="state-value">${esc(readable(v))}</div></div>`;
    }).join('')}

    <div class="section-title"><span>Attention Spectrum Gauges</span><span>0.000 — 1.000</span></div>
    <div class="gauges">${Object.entries(factors).map(([k,v])=>{
      const n=Math.max(0,Math.min(1,num(v)));
      return `<div class="gauge"><div class="gauge-label"><span>${esc(k.replaceAll('_',' ').replace(/\b\w/g,m=>m.toUpperCase()))}</span><strong style="color:${fMap[k]||'var(--cyan)'}">${n.toFixed(3)}</strong></div><div class="g-track"><div class="g-fill" style="width:${n*100}%;background:${fMap[k]||'var(--cyan)'}"></div></div></div>`;
    }).join('')}</div>

    <div class="section-title"><span>Incident Signal</span><span>${esc(up(i.current_attention_level))}</span></div>
    <div class="state-card ${up(i.current_attention_level).includes('CRITICAL')?'conflict':''}">
      <div class="state-key">ATTENTION / UNCERTAINTY</div>
      <div class="state-value">${esc(up(i.current_attention_level))}</div>
      <div style="margin-top:8px;color:var(--muted);font-size:9px;line-height:1.45">Current confidence: ${num(i.current_confidence).toFixed(2)} · higher uncertainty remains visible beside the state.</div>
    </div>
  `;
}

function selectIncident(id){ selectedId=id; render(); }
function jumpReplay(i){ stateIndex=Math.max(0,Math.min(DATA.reports.length,i)); playing=false; clearInterval(timer); render(); }
function resetReplay(){ stateIndex=0; selectedId=null; playing=false; clearInterval(timer); render(); }
function prevReplay(){ if(stateIndex>0){stateIndex--; playing=false; clearInterval(timer); render();} }
function nextReplay(){ if(stateIndex<DATA.reports.length){stateIndex++; render();} }
function togglePlay(){
  playing=!playing;
  clearInterval(timer);
  if(playing){
    timer=setInterval(()=>{
      if(stateIndex>=DATA.reports.length){playing=false;clearInterval(timer);render();return;}
      stateIndex++; render();
    },1000);
  }
  render();
}
function readable(v){ return v==null?'—':(typeof v==='number'?String(v):String(v)); }
render();
</script>
</body>
</html>
"""

html_payload = APP.replace("__PAYLOAD__", payload)

# One custom HTML surface = one coherent Stitch-like viewport.
# The iframe is tall enough to contain the full console and internally scrollable.
components.html(html_payload, height=2200, scrolling=False)

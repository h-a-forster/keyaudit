"""README.md and REPORT.md publish copies of rows rendered from the committed results.

`scripts/published_tables.py` writes results/published.md; CI reruns it and fails on a diff. A
row in a document must start with the cells of a published row; cells after them are prose.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "results" / "published.md"
# The run record and the data-source list hold no results.
NOT_RESULTS = ("| | study-v1 | simulation-v2 |", "| Source | Access | Licence |")


def _script():
    path = ROOT / "scripts" / "published_tables.py"
    spec = importlib.util.spec_from_file_location("published_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _rows(path: Path) -> list[str]:
    rows, skipping = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            skipping = False
        elif line.startswith(NOT_RESULTS):
            skipping = True
        elif line.startswith("| ") and not skipping:
            rows.append(line)
    return rows


def test_published_tables_render_from_the_committed_results(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert PUBLISHED.read_text(encoding="utf-8") == _script().render()


def test_every_table_row_starts_with_a_published_row():
    source = [_cells(line) for line in _rows(PUBLISHED)]
    for doc in ("README.md", "REPORT.md"):
        rows = _rows(ROOT / doc)
        assert rows, doc
        unmatched = [row for row in rows if not any(_cells(row)[: len(s)] == s for s in source)]
        assert unmatched == [], doc


def _e6_headlines() -> dict[str, list[str]]:
    """The E6 figures README, REPORT and CITATION state in prose, rendered from its summary."""
    import json

    s = json.loads((ROOT / "results/e6-benchmarks/summary.json").read_text(encoding="utf-8"))
    geo, date = s["bbh"]["geometric_shapes"], s["bbh"]["date_understanding"]
    arc, tqa = s["sampled"]["arc_challenge"], s["sampled"]["truthfulqa"]
    defects = geo["positives"] + date["positives"]
    bbh = (
        f"{defects} real key defects in BIG-Bench Hard, found by solving every item, "
        f"keyaudit's flags hit {geo['flags_on_positives']} of {geo['positives']} in "
        f"geometric_shapes and {date['flags_on_positives']} of {date['positives']} in "
        "date_understanding"
    )
    e_model = f"model-only rates are {arc['e_model_pct']:.0f}% and {tqa['e_model_pct']:.0f}%"
    flagged = (
        f"({100 * arc['flags'] / arc['items']:.0f}% and "
        f"{100 * tqa['flags'] / tqa['items']:.0f}% of items flagged"
    )
    rates = f"{arc['any_defect']['rate_pct']:.1f}% and {tqa['any_defect']['rate_pct']:.1f}%"
    arc_ci, tqa_ci = ("[{:.2f}%, {:.2f}%]".format(*b["any_defect"]["ci95_pct"]) for b in (arc, tqa))
    return {
        "README.md": [
            bbh.replace("keyaudit's flags", "the flags"),
            f"{e_model} {flagged} at q <= 0.10)",
            f"defect rates at {arc['any_defect']['rate_pct']:.1f}% {arc_ci} and "
            f"{tqa['any_defect']['rate_pct']:.1f}% {tqa_ci}.",
        ],
        "REPORT.md": [
            f"on {bbh}",
            f"{e_model} {flagged}), against defect rates of {rates} in random samples "
            f"checked item by item against sources (95% intervals {arc_ci} and {tqa_ci}).",
            f"The model-only rates, {arc['e_model_pct']:.0f}% and {tqa['e_model_pct']:.0f}%, "
            f"are far above the sampled defect rates of {rates},",
            f"flag precision was {arc['wrong_key']['positives_flagged']} of "
            f"{arc['sample_flagged']}.",
            "The geometric_shapes solver left {unparsed} of its {n} items unparsed".format(
                **s["bbh_solver_census"]["geometric_shapes"]
            ),
        ],
        "CITATION.cff": [
            f"on {defects} real BIG-Bench Hard key defects ({date['positives']} wrong keys, "
            f"{geo['positives']} keys that are not the single best answer) its flags catch "
            f"{geo['flags_on_positives'] + date['flags_on_positives']} of them.",
        ],
    }


def test_e6_headline_figures_match_the_summary():
    for doc, claims in _e6_headlines().items():
        text = " ".join((ROOT / doc).read_text(encoding="utf-8").split())
        missing = [c for c in claims if c not in text]
        assert missing == [], doc


def _study_headlines() -> dict[str, list[str]]:
    """The study-v1 figures README and REPORT state in prose, rendered from its results."""
    import json

    def load(name: str) -> dict:
        return json.loads((ROOT / "results/study-v1" / name).read_text(encoding="utf-8"))

    est, val, panel = load("estimates.json"), load("validation.json"), load("panel.json")
    mp = est["mmlu_pro"]
    derived, nar = mp["mmlu_derived_model_only"], mp["label_calibrated_narrow"]
    bro = mp["label_calibrated_broad"]
    subj = load("labels.json")["label_estimates"]["canonical"]["narrow"]["per_subject"]
    chem, vir = subj["college_chemistry"], subj["virology"]
    v = val["mmlu_pro"]["narrow"]
    op = v["operating_point"]
    top = v["baselines"]["top_disagreement"]
    vs_ds = v["vs_baselines"]["one_coin_ds"]
    above = panel["mmlu_pro"]["choices"]["posterior_above_half"]
    gpqa_acc = panel["gpqa"]["accuracy"]
    n_flagged = _script().flagged()["mmlu_pro"]
    pct = "{:.2%}".format
    ci = "[{:.2%}, {:.2%}]".format
    rate = f"{pct(nar['estimate'])} {ci(*nar['ci'])}"
    ratio_all = derived["overall"] / nar["estimate"]
    ratio_reg = derived["region_rate"] / nar["estimate"]
    lo, hi = sorted((derived["overall"], derived["region_rate"]))
    shares = (
        f"{above['n_items']:,} items with posterior above 0.5 they pick the key "
        f"{above['strongest']['key']:.0%} of the time and one shared other answer "
        f"{above['strongest']['proposed']:.0%} of the time"
    )
    ap = f"{v['ap']:.3f} {'[{:.3f}, {:.3f}]'.format(*v['ap_ci'])}"
    chem_n = f"{chem['positives']} of {chem['n_labeled']}"
    vir_n = f"{vir['positives']} of {vir['n_labeled']}"
    return {
        "README.md": [
            f"is {100 * lo:.0f}-{100 * hi:.0f}%, against {rate} by the human labels",
            f"(variance ratio {nar['variance_ratio']:.2f})",
            f"(labels alone: {pct(nar['classical'])})",
            f"about {ratio_all:.0f} times the human-label rate",
            f"On the {shares}.",
            f"precision {ap} (random {v['base_rate']:.3f}, ROC AUC {v['auc']:.2f})",
            f"top-decile disagreement ({top['ap']:.3f})",
            f"College chemistry ({chem_n} labeled keys wrong) and virology ({vir_n})",
            f"median accuracy {gpqa_acc['median']:.2f}) and the model-only rate "
            f"({est['gpqa']['overall']:.0%})",
            f"model-only flag rate is {pct(mp['region_rate'])} {ci(*mp['region_ci'])} in the "
            f"competence region and {pct(mp['overall'])} overall",
        ],
        "REPORT.md": [
            f"labels-only estimate ({pct(nar['classical'])}, {pct(bro['classical'])} on the "
            "same items)",
            f"{pct(derived['overall'])} over all of them and {pct(derived['region_rate'])} on "
            f"the {derived['region_share']:.0%} the panel can judge, against the "
            f"{pct(nar['estimate'])} human-label rate ({ratio_all:.1f}x and {ratio_reg:.1f}x)",
            f"actually flags at q <= 0.10 is similar: {op['n_selected']} of the {v['n']:,} labeled "
            f"items ({op['n_selected'] / v['n']:.1%}) and {n_flagged:,} of all "
            f"{mp['n_items']:,} MMLU-Pro items ({n_flagged / mp['n_items']:.1%})",
            f"{mp['n_items']:,} MMLU-Pro items the rate is {100 * mp['overall']:.1f}%, and for "
            f"GPQA {est['gpqa']['overall']:.0%}",
            f"of the {op['n_selected']} labeled items among the study's flags (q <= 0.10), only "
            f"{op['tp']} ({op['precision']:.1%}) are wrong keys",
            f"median accuracy {gpqa_acc['median']:.2f}, best {gpqa_acc['best']:.2f}",
            f"Against the {v['n_positive']} Redux wrong keys among {v['n']:,} labeled items its "
            f"average precision is {ap} (random {v['base_rate']:.3f}, ROC AUC {v['auc']:.2f})",
            f"top-decile disagreement, gets {top['ap']:.3f} "
            f"{'[{:.3f}, {:.3f}]'.format(*top['ap_ci'])}",
            "one-coin Dawid-Skene (AP difference {:+.3f} [{:+.3f}, {:+.3f}])".format(
                vs_ds["diff"], vs_ds["lo"], vs_ds["hi"]
            ),
            f"college chemistry, {chem_n} labeled items, {chem['rate']:.1%} "
            f"{'[{:.1%}, {:.1%}]'.format(*chem['ci'])}, and virology, {vir_n}, "
            f"{vir['rate']:.1%} {'[{:.1%}, {:.1%}]'.format(*vir['ci'])}",
        ],
    }


def test_study_headline_figures_match_the_results():
    for doc, claims in _study_headlines().items():
        text = " ".join((ROOT / doc).read_text(encoding="utf-8").split())
        missing = [c for c in claims if c not in text]
        assert missing == [], doc


def test_simulation_se_ratio_matches_the_summary():
    """REPORT 4.1 states the calibrated-to-labels-only SE ratio of the simulation scenarios."""
    import json

    s = json.loads((ROOT / "results/simulation-v2/summary.json").read_text(encoding="utf-8"))
    sc = s["scenarios"]
    ratios = [sc[n]["cal_se"]["mean"] / sc[n]["classical_se"]["mean"] for n in sc if n != "tricks"]
    tricks = sc["tricks"]
    claim = (
        f"0.55 to 0.57 points): {min(ratios):.2f} to {max(ratios):.2f} of it without trick items, "
        f"{100 * tricks['cal_se']['mean']:.2f} against {100 * tricks['classical_se']['mean']:.2f} "
        "with them"
    )
    assert claim in " ".join((ROOT / "REPORT.md").read_text(encoding="utf-8").split())

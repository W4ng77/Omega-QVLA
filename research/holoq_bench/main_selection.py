"""One frozen cohort definition shared by final replay/report/common optimization."""
import json


COHORTS = ("evaluation_v3", "evaluation_rotation_v2", "evaluation_paper_holoq")


def read_selection(root):
    primary = json.loads((root / COHORTS[0] / "queue.json").read_text())
    prior = json.loads((root / COHORTS[1] / "backend_selection.json").read_text())
    paper = json.loads((root / COHORTS[2] / "backend_selection.json").read_text())
    families = {k: v for k, v in prior["families"].items() if not k.startswith("holoq")}
    families.update(paper["families"])
    selection = {"families": families, "frozen_by": "latency only after functional gates; paper HoloQ replaces two-sided reconstruction controls"}
    cases = [case for case in primary["valid_cases"] if not case.startswith("holoq")]
    modes = {case: ["head_graph"] for case in cases}
    for family in families.values():
        choice = family["selected"]
        case = choice["case"]
        if case not in cases:
            cases.append(case)
        modes[case] = list(dict.fromkeys(["head_graph", choice["mode"]]))
    assert len([case for case in cases if case.startswith("holoq")]) == 2
    return cases, modes, selection

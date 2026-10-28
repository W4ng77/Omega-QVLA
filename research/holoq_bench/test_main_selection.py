"""Prevent the non-paper reconstruction controls reentering final comparisons."""
import json
from research.holoq_bench.main_selection import read_selection


def test_paper_methods_replace_controls_but_keep_genuine_baselines(tmp_path):
    def write(cohort, name, value):
        path = tmp_path / cohort / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def choice(case, mode="head_graph"):
        return {"selected": {"case": case, "mode": mode}}

    write("evaluation_v3", "queue.json", {"valid_cases": ["holoq", "gptq_full", "smoothquant_a8"]})
    write("evaluation_rotation_v2", "backend_selection.json", {"families": {
        "holoq": choice("holoq_llm_s4_rotv2"),
        "holoq_step_weight": choice("holoq_step_weight_llm_s4_rotv2"),
        "quantvla_a8": choice("quantvla_selective_a8_atm_g128_rotv2"),
    }})
    write("evaluation_paper_holoq", "backend_selection.json", {"families": {
        "holoq": choice("holoq_paper_llm_s4"),
        "holoq_step_weight": choice("holoq_paper_step_weight_llm_s4", "eager"),
    }})
    cases, modes, selection = read_selection(tmp_path)
    assert cases == ["gptq_full", "smoothquant_a8", "quantvla_selective_a8_atm_g128_rotv2", "holoq_paper_llm_s4", "holoq_paper_step_weight_llm_s4"]
    assert modes["holoq_paper_step_weight_llm_s4"] == ["head_graph", "eager"]
    assert selection["families"]["quantvla_a8"]["selected"]["case"].startswith("quantvla")

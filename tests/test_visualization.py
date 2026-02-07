import pytest


def test_plot_per_class_ap_ar(tmp_path):
    pytest.importorskip("matplotlib")
    from tcm_tongue.utils.visualization import plot_per_class_ap_ar

    ap = {"class_a": 0.1, "class_b": 0.3}
    ar = {"class_a": 0.2, "class_b": 0.1}
    out = tmp_path / "per_class.png"

    ok = plot_per_class_ap_ar(ap, ar, out)
    assert ok
    assert out.is_file()

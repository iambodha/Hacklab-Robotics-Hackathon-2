from models.bench import score


def test_marker_matching_counts_misses_and_false_points():
    it = {"kind": "markers", "camera": "wrist", "size": [299, 224],
          "gt": {"points": [{"id": 1, "px": [10, 10]}, {"id": 2, "px": [100, 100]}, {"id": 3, "px": [200, 50]}]}}
    row = {"pred": {"points": [{"px": [12, 10]}, {"px": [100, 104]}, {"px": [290, 220]}]}, "error": None}
    m = score.item_metrics(it, row)
    assert m["matched"] == 2 and m["err_px299"] == [2.0, 4.0] and m["false_points"] == 1


def test_objects_hits_hallucination_and_scale():
    it = {"kind": "objects", "camera": "overview", "size": [598, 448],
          "gt": {"boxes": {"purple wrapper": [100, 100, 120, 120], "grey box": [300, 50, 400, 150], "black object": None}}}
    row = {"error": None, "pred": {
        "points": [{"label": "purple wrapper", "px": [110, 130]}, {"label": "black object", "px": [5, 5]}],
        "boxes": [{"label": "grey box", "box": [300, 50, 400, 150]}]}}
    m = score.item_metrics(it, row)
    assert m["present"] == 2 and m["missed"] == 1 and m["hallucinated"] == 1
    assert m["hits"] == 0 and m["err_px299"] == [10.0]          # 20 px off at 598 wide = 10 px at 299
    assert m["box_iou"] == [1.0]


def test_judgement_false_success_and_summary():
    rows = []
    for truth, pred in [(False, True), (False, False), (True, True), (False, None)]:
        it = {"kind": "grasp", "size": [598, 448], "gt": {"held": truth}}
        r = {"kind": "grasp", "error": None, "pred": {"done": pred}, "calls": [{"latency_s": 1.0 + len(rows), "cost_usd": 0.001}]}
        r["metrics"] = score.item_metrics(it, r)
        rows.append(r)
    s = score.summarise(rows)
    assert s["grasp"]["negatives"] == 3 and s["grasp"]["false_success_rate"] == round(1 / 3, 3)
    assert s["grasp"]["unknown"] == 1 and s["latency_p50_s"] == 2.5 and s["cost_total_usd"] == 0.004


def test_stain_residual_iou():
    it = {"kind": "stain", "camera": "overview", "size": [299, 224],
          "gt": {"done": False, "residual_box": [10, 10, 20, 20], "world": None}}
    m = score.item_metrics(it, {"error": None, "pred": {"done": False, "residual": [[10, 10, 20, 30]]}})
    assert m["correct"] and m["residual_iou"] == 0.5

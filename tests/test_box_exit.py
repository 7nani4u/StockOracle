from copy import deepcopy
from market_briefing.box_exit import measured_box_move,attach_box_exit_levels


def box(state="range"):
    return {"state":state,"support":{"low":89,"high":91,"center":90},"resistance":{"low":109,"high":111,"center":110},
            "targets":{k:{"low":lo,"high":hi,"price_source":"observed_box","reason":"observed","eligible":True} for k,lo,hi in [("conservative",105,108),("neutral",109,111),("aggressive",None,None)]}}


def test_box_height_added_to_resistance_not_entry():
    result=measured_box_move(box())
    assert result["height"]==20 and result["center"]==130
    assert [result["low"],result["high"]]==[127,133]
    assert not result["eligible"]
    assert measured_box_move(box("breakout_confirmed"))["eligible"]
    for state in ("breakdown_pending","breakdown_confirmed","false_breakout"):
        assert measured_box_move(box(state)) is None


def test_original_tp_rows_preserved_and_only_matching_rows_annotated():
    original={key:{"tp_levels":[{"price":lo+1,"price_range":[lo,lo+2]} for lo in (100,104,108,112,127)],"target":[100,129]} for key in ("conservative","balanced","aggressive")}
    before=deepcopy(original)
    risk=deepcopy(original)
    for card in risk.values(): card["entry_status"]="waiting"
    attach_box_exit_levels(risk,original,box(),97)
    assert original==before
    assert len(risk["aggressive"]["tp_levels"])==5
    assert risk["aggressive"]["tp_levels"][4]["box_exit_annotations"][0]["label"]=="박스 높이 조건부 목표"
    assert "box_exit_annotations" not in risk["aggressive"]["tp_levels"][0]
    assert risk["balanced"]["tp_levels"][2]["box_exit_annotations"]
    for key in risk:
        assert [lv["price_range"] for lv in risk[key]["tp_levels"]]==[lv["price_range"] for lv in original[key]["tp_levels"]]


def test_missing_tp_uses_five_conditional_projection_rows_without_probability():
    risk={key:{"target":None} for key in ("conservative","balanced","aggressive")}
    attach_box_exit_levels(risk,{},box(),97)
    levels=risk["aggressive"]["tp_levels"]
    assert len(levels)==5
    assert levels[0]["price_range"][0]==127 and levels[-1]["price_range"][1]==133
    assert all(lv["prob_pct"] is None for lv in levels)
    assert all(not lv["box_exit_annotations"][0]["eligible"] for lv in levels)

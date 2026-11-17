from eval.scoring import abstained, evidence_hit, fact_hits


def test_fact_hits_number_formats():
    ans = "ViT-L/16 cần **0,68k** TPUv3-core-days, BiT-L cần 9.9k; độ chính xác 87,76% so với 87.54%."
    facts = [["0.68k", "0,68k"], ["9.9k", "9,9k"], ["87.76", "87,76"], ["87.54", "87,54"]]
    assert fact_hits(ans, facts) == [True, True, True, True]


def test_fact_hits_ranges_and_missing():
    assert fact_hits("ít hơn khoảng 2 − 4 lần", [["2-4"]]) == [True]
    assert fact_hits("Không tìm thấy", [["0.68k"]]) == [False]


def test_abstain_and_evidence():
    assert abstained("Không tìm thấy thông tin phù hợp trong tài liệu để trả lời câu hỏi này.")
    assert not abstained("ViT đạt 50 mAP trên COCO.")
    assert evidence_hit(["... | TPUv3-core-days | 0.68k |"], ["0.68k"])
    assert not evidence_hit([], ["0.68k"])

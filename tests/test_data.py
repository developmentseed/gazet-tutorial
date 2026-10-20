from gazet_tutorial.data import count_by, shuffle_candidates, stratified_split


def _samples():
    out = []
    for tpl in ("t1", "t2", "t3"):
        for i in range(20):
            out.append({"id": f"{tpl}_{i}", "candidates": [{"candidate_id": f"c{k}"} for k in range(5)],
                        "metadata": {"template_id": tpl, "task_family": "fam_" + tpl}})
    return out


def test_stratified_split_covers_every_template():
    train, val, test = stratified_split(_samples())
    assert len(train) == 48 and len(val) == 6 and len(test) == 6
    for split in (train, val, test):
        assert {s["metadata"]["template_id"] for s in split} == {"t1", "t2", "t3"}
    assert count_by(train) == {"fam_t1": 16, "fam_t2": 16, "fam_t3": 16}


def test_stratified_split_is_deterministic():
    a = stratified_split(_samples())[0]
    b = stratified_split(_samples())[0]
    assert [s["id"] for s in a] == [s["id"] for s in b]


def test_shuffle_candidates_deterministic_and_non_mutating():
    s = _samples()[0]
    before = [c["candidate_id"] for c in s["candidates"]]
    x = shuffle_candidates(s)
    y = shuffle_candidates(s)
    assert [c["candidate_id"] for c in s["candidates"]] == before
    assert [c["candidate_id"] for c in x["candidates"]] == [c["candidate_id"] for c in y["candidates"]]
    assert sorted(c["candidate_id"] for c in x["candidates"]) == sorted(before)

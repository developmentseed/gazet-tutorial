from gazet_tutorial.dpo import RESPONSE_PART, choose_pair


def test_choose_pair_prefers_short_chosen_and_valid_rejected():
    samples = ["SELECT 1", "SELECT 1 -- longer", "SELEC broken", "SELECT 2", "SELECT 1"]
    verdicts = [{"pass": True, "sql_valid": True}, {"pass": True, "sql_valid": True},
                {"pass": False, "sql_valid": False}, {"pass": False, "sql_valid": True},
                {"pass": True, "sql_valid": True}]
    chosen, rejected = choose_pair(samples, verdicts)
    assert chosen == "SELECT 1"            # shortest passing answer
    assert rejected == "SELECT 2"          # valid-but-wrong beats a syntax error


def test_choose_pair_needs_both_outcomes_and_distinct_text():
    assert choose_pair(["a", "b"], [{"pass": True}, {"pass": True}]) is None
    assert choose_pair(["a", "b"], [{"pass": False}, {"pass": False}]) is None
    # same string judged both ways (non-deterministic execution) must not become a pair
    assert choose_pair(["a", "a "], [{"pass": True}, {"pass": False}]) is None
    assert choose_pair([], []) is None


def test_response_part_matches_sft_marker():
    assert RESPONSE_PART.endswith("<think>\n\n</think>\n\n")


def test_render_prompt_ends_in_non_thinking_marker():
    from gazet_tutorial.dpo import render_prompt

    class Tok:
        def __init__(self, tail):
            self.tail = tail

        def apply_chat_template(self, messages, **kw):
            assert kw["add_generation_prompt"] and kw["enable_thinking"] is False
            return "<|im_start|>user\nq<|im_end|>\n" + self.tail

    bare = render_prompt(Tok("<|im_start|>assistant\n"), [])
    full = render_prompt(Tok(RESPONSE_PART), [])
    assert bare == full and full.endswith(RESPONSE_PART)

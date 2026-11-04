# Specifications in plain English

Each `.feature` file describes the rules one part of the pipeline guarantees, in
[Gherkin](https://cucumber.io/docs/gherkin/reference/) (*Given / When / Then*). Read these
first if you're new to the repo: they say **what** the code promises without the code.

| Feature | What it guarantees | Steps (the code behind it) |
|---|---|---|
| [`splits.feature`](splits.feature) | Every task is in one split; test is never trained on | `tests/test_features_splits.py` |
| [`golden_trajectories.feature`](golden_trajectories.feature) | A teacher conversation becomes a correct training record, or is rejected | `tests/test_features_golden.py` |

**Run them:** `just features` prints each scenario and whether it passed. They're also part of
`just test` and CI.

**How it fits with the other tests:** the `.feature` files cover the *rules a reader needs to
know*. The other `tests/test_*.py` files cover edge cases and internals in plain pytest.

**Adding a scenario:** write it in the `.feature` file first, in words a teammate would
understand, then add any new steps to its `test_features_*.py` file. Step functions are typed,
and none of them may be named `test_*` (pytest would run it as a test).

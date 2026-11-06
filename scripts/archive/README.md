# Archived scripts

Kept for the record, not part of the current pipeline. They still run.

| Script | What it did | Why it's archived |
|---|---|---|
| `teacher_probe.py` (`just teacher-probe`) | 3 Nov: checked whether a Cohere model does native tool calling. It doesn't, so Aya uses prompted `Action:` calls. | Answered. The finding is documented in `src/reflexive/teacher.py` and `docs/TEACHER_SETUP.md` |
| `teacher_check.py` (`just teacher-check`) | 3 Nov: Aya Expanse 32B as the *teacher* on 3 tasks, with cost and latency | Aya became the student when GLM-5.3-Flash won the bake-off |

Deleted on 4 Nov: `setup_github_project.sh`, which turned `docs/WORKPLAN.md` into GitHub issues.
Work is tracked in Linear instead.

"""Write the model's chat template without its "roles must alternate" check, for vLLM (SEL-65).

    python scripts/sft/chat_template.py CohereLabs/aya-expanse-32b /workspace/aya_chat_template.jinja

`scripts/pod/serve.sh` runs this and passes the file to ``vllm serve --chat-template``, so the
served model renders conversations exactly as training did (`reflexive.sft.relaxed_template`).
"""

from __future__ import annotations

import sys
from pathlib import Path

from reflexive.sft import default_template, relax_chat_template


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    from transformers import AutoTokenizer

    model, out = sys.argv[1], Path(sys.argv[2])
    template = default_template(relax_chat_template(AutoTokenizer.from_pretrained(model).chat_template))
    out.write_text(template)
    print(f"wrote {out} ({len(template):,} characters, without the alternation check)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

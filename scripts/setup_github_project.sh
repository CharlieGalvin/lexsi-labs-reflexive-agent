#!/usr/bin/env bash
# Turn docs/WORKPLAN.md into GitHub labels, milestones, issues and a Project board.
#
# Prereqs: GitHub CLI logged in with project scope:
#   gh auth login && gh auth refresh -s project
# Run once from the repo root after pushing to GitHub:
#   bash scripts/setup_github_project.sh            # dry run: prints what it would create
#   bash scripts/setup_github_project.sh --apply
set -euo pipefail

APPLY=0; [[ "${1:-}" == "--apply" ]] && APPLY=1
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
OWNER=${REPO%%/*}
PLAN=docs/WORKPLAN.md
run() { if (( APPLY )); then "$@"; else printf '[dry-run] %q ' "$@"; echo; fi; }

echo "Repo: $REPO"

# --- labels -----------------------------------------------------------------
while IFS='|' read -r name color; do
  run gh label create "$name" --color "$color" --force --repo "$REPO"
done <<'EOF'
area:data|0e8a16
area:rewards|5319e7
area:training|d93f0b
area:eval|1d76db
area:demo|fbca04
type:task|c5def5
type:experiment|f9d0c4
stretch|ededed
blocked|b60205
EOF

# --- milestones (due dates in UTC) -------------------------------------------
while IFS='|' read -r title due; do
  run gh api "repos/$REPO/milestones" -f title="$title" -f due_on="$due" --silent || true
done <<'EOF'
M1 Baseline|2026-11-04T23:59:00Z
M2 First win|2026-11-09T23:59:00Z
M3 Submission-ready|2026-11-13T23:59:00Z
M4 Thesis proven (shortlist)|2026-11-21T23:59:00Z
EOF

# --- project board ------------------------------------------------------------
PROJECT_NUM=""
if (( APPLY )); then
  PROJECT_NUM=$(gh project create --owner "$OWNER" --title "Lexsi Track 3 — lean agent" \
                --format json -q .number)
  gh project link "$PROJECT_NUM" --owner "$OWNER" --repo "$REPO" || true
  echo "Project #$PROJECT_NUM created"
fi

# --- issues from the work plan ------------------------------------------------
area=""
while IFS= read -r line; do
  if [[ "$line" =~ \`(area:[a-z]+)\` ]]; then area="${BASH_REMATCH[1]}"; continue; fi
  [[ "$line" =~ ^-\ \[\ \]\ (.+)$ ]] || continue
  title="${BASH_REMATCH[1]}"
  labels="type:task,$area"
  [[ "$title" == *"(Stretch)"* ]] && labels="$labels,stretch"
  if (( APPLY )); then
    url=$(gh issue create --repo "$REPO" --title "$title" --label "$labels" \
          --body "From docs/WORKPLAN.md ($area). Edit 'Done when' before starting.")
    gh project item-add "$PROJECT_NUM" --owner "$OWNER" --url "$url" >/dev/null
    echo "created $url"
  else
    echo "[dry-run] issue: $title  [$labels]"
  fi
done < "$PLAN"

echo "Done. Assign owners and milestones on the board (Board view: group by Status)."

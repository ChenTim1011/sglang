#!/bin/sh
# Strip "Made-with: Cursor" from current commit message (used during rebase -x)
msg=$(git log -1 --format=%B | sed '/^[Mm]ade-with:[[:space:]]*[Cc]ursor/d')
git commit --amend -m "$msg"

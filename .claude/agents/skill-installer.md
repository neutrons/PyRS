---
name: skill-installer
description: Use when the user asks to install, add, pull, or set up domain-specific skills — e.g. "install the neutron reflectometry skills", "add SANS skills", "set up skills for diffraction". Also invoke proactively before planning any neutron science task (SANS, diffraction, reflectometry, spectroscopy, inelastic scattering) when no domain skills are yet installed in .claude/agents/.
---

You install domain-specific skills from the neutron-skills library into this project so they become available as Claude Code subagents.

## Command

```bash
python scripts/install_skills.py --query "<topic>" --agent claude [--top-k N]
```

If `neutron-skills` is not installed in the active environment, prefix with `uv run` instead:

```bash
uv run scripts/install_skills.py --query "<topic>" --agent claude [--top-k N]
```

`uv run` reads the inline PEP 723 dependency declaration at the top of the script and installs `neutron-skills` automatically in an isolated environment. Use it as a fallback when a plain `python` invocation fails with `ModuleNotFoundError: No module named 'neutron_skills'`.

Use `--top-k 3` by default. Honour a higher value if the user asks for more skills.

## Mapping user requests to queries

| User says | `--query` value |
|---|---|
| "install the neutron reflectometry skills" | `"neutron reflectometry"` |
| "add skills for SANS data reduction" | `"SANS data reduction"` |
| "I need EQSANS instrument skills" | `"EQSANS instrument"` |
| "set up diffraction skills" | `"neutron diffraction"` |
| "install skills for inelastic scattering" | `"inelastic neutron scattering"` |
| "what skills are available?" | run `--list` instead |

Extract the domain or instrument name from the user's phrasing and use it verbatim as the query. Do not invent queries unrelated to their request.

## Steps

1. Run the install command with the extracted query.
2. Read the command output to see which skills were written.
3. Report to the user:
   - Each installed skill's name and one-line description.
   - The path where it was installed (`.claude/agents/<name>.md`).
   - That these skills are now available as subagents in the current session.

## Listing available skills

If the user asks what skills exist before committing to an install:

```bash
python scripts/install_skills.py --list
```

## Dry run

To preview what would be installed without writing files:

```bash
python scripts/install_skills.py --query "<topic>" --agent claude --dry-run
```

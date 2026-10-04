# Fourth skill experiment: Assumption Lab

Update (2026-10-03): version 0.2.0 expands this skill into **Playground**. It
adds computed models when evidence supports an equation and preserves the
finite Assumption Lab for qualitative cases. A separate Presentation workflow
brings the total to five. The experiment below records the original selection,
not a limit on the current skill. No measured demand or learning-gain claim has
been established.

Decision: include **Assumption Lab** in the private package as an experimental
workflow. It demonstrated a useful interaction that the existing study-guide
slider does not provide: change an assumption, hold a comparison case fixed,
and see whether the conclusion, its applicability, or both change. This is an
inclusion judgment from a small prototype exercise, not evidence of demand,
better learning outcomes, or a proven killer feature.

## Method and three candidates

Use the same cached real lecture excerpt and inspected slide for all candidates:
the opening SimCLR review from Stanford CS231N Lecture 13, approximately
02:51–05:08. The exact transcript, image, prototypes, and eight-item audit are
private task artifacts. This work consumed **zero additional VMF API units**.
It did not retrieve a new transcript, create an index, train a model, or use a
new service. Missing high-resolution media limited the visual check to a
readable pipeline and color relationships in the stored thumbnail.

The small experiment produced three low-fidelity outputs on four common tasks:
change original identity while preserving apparent category; lose the subject
while preserving origin; distinguish a source rule from a predicted training
outcome; return to the supporting evidence. Assumption Lab was then implemented
as a complete artifact. The other two prototypes were a three-claim feedback
receipt and a four-node dependency map. This is one author's inspection of one
excerpt, with no participants or measured learning score.

| Candidate | Value beyond summarization | Timestamped visual/transcript benefit | Infrastructure reuse and incremental cost | Discovery and use | Plausible reason to return | Main limitation |
|---|---|---|---|---|---|---|
| **Assumption Lab** | Controlled changes separate the pairing rule from assumptions about task usefulness; a pinned case makes the distinction explicit | The inspected pair diagram anchors the rule; transcript moments justify what does and does not follow | Existing cached evidence, a stdlib renderer, and an offline template; 0 extra units in this experiment | “What if I change this assumption?” opens two controls and four reviewed cases | Bring a new rule or edge case and keep a source-linked comparison | Case construction requires judgment; no measured training behavior or proof of learning gains |
| Claim Check receipt | Turns the learner's explanation into supported/correction/unknown claims | Exact source moments make each correction reviewable; visuals can resolve a specific misreading | Same cache, ordinary chat table; 0 extra units | “Check my explanation” is easy and useful | Recheck an explanation after studying | Almost entirely duplicates the conversational tutor; a fourth packaged experience adds little |
| Prerequisite Rescue Map | Shows which missing idea to revisit before a difficult concept | Links nodes to explanations and diagram stages | Same cache for covered nodes; missing prerequisite evidence may require new bounded retrieval | “What do I need to understand first?” opens a short concept map | Revisit a stuck prerequisite | On this excerpt the covered nodes reproduce the study-guide pipeline; the most interesting task-invariance node lacks evidence |

The weakest distinction was the rescue map: making a fuller map would either
add unsupported prerequisites or spend retrieval on a different scope. Claim
Check is useful, but belongs inside tutoring. Assumption Lab was the only
candidate whose additional interface earned its complexity for the tested
controlled-comparison task.

## What the implemented prototype demonstrated

The real-evidence lab crosses two discrete assumptions: shared versus different
original photographs, and a crop that retains versus loses its visible subject.
It provides all four combinations. The generated counterexamples are labeled
as such; neither a second cat photograph nor the background-only crop is claimed
to be an experiment shown by the lecturer.

| Controlled change | Observed artifact behavior | Evidence boundary |
|---|---|---|
| Same original → different originals; recognizable category unchanged | Pull-together outcome becomes push-apart; semantic-category caution appears | Follows the excerpt's instance-based origin rule; does not predict downstream quality |
| Recognizable subject → background-only crop; same original unchanged | Pull-together outcome stays; applicability warning changes | Origin identity is unchanged by construction; crop usefulness is explicitly unresolved |
| Pin a nonbaseline case, change one control, then reset | Reference stays fixed during exploration; reset restores both cases | UI behavior, not educational or scientific evidence |
| Download a comparison with a learner note | Plain-text file preserves the chosen cases, outcome, boundary, evidence IDs, and note | No persistent learner profile; keep the HTML for complete timestamps and evidence |

The evidence audit checked eight substantive items, including both pairing rules,
the semantic-category inference, the generated crop boundary, the actual slide's
color relationships, source timestamps, and the absence of an exact loss or
training result within the bounded excerpt. A 320px image is not treated as a
numeric similarity heatmap or as a basis for reading tiny equations.

## Comparison with the nearest core skill and simplest alternative

The nearest core skill is the study guide, whose generated matrix slider changes
batch size and explains pair counts. Assumption Lab changes a **condition of a
claim** and compares applicability. It has no pair-count slider and does not ask
or grade a quiz. The tutor remains better for diagnosing an unfamiliar learner
answer and inventing a targeted follow-up in conversation.

For the single question “Do different cat photos count as a matching pair?”, a
two-sentence explanation in chat is cheaper to read and fully adequate. The lab
earns its place when a learner wants to test several controlled changes and
keep the distinction visible across them. A static four-row table can also
convey the logical results; pinning and showing only the current case reduce
cross-row comparison work, at the cost of a longer artifact and interaction
controls. No experiment here establishes that the lab beats that table on
comprehension, speed, or retention.

## Portable implementation and checks

- Workflow: `plugins/video-moment-finder/skills/assumption-lab/SKILL.md`.
- Renderer and template: `tools/assumption_lab.py` and
  `templates/assumption-lab.html` inside the plugin. Only Python stdlib is used;
  shared source escaping, frame validation, and timestamp helpers are reused.
- Redistribution-safe transfer example:
  `plugins/video-moment-finder/examples/assumption-lab-ohms-law.json` and `.html`.
  Its four original cases explore voltage and a fixed/unknown resistance. It
  demonstrates transfer of the interaction to another domain; it is **not a
  second real-lecture test**.
- All combinations must have a reviewed state. Missing states, invented
  citations, duplicate combinations, unsafe image paths, and source markup
  cannot silently become executed models or external file contents.
- `tests/plugin/test_assumption_lab.py`: **13 tests passed**, including hostile
  source text, metadata exclusion, complete state coverage, source references,
  correct original calculations, image fallback, and actual interaction-code
  execution with Node DOM test doubles for selection, pinning, reset, and note
  download. This establishes script behavior in that harness, not browser layout
  or assistive-technology behavior.

An actual Chromium run, with its sandbox enabled through supported execution
permission, also passed on the private real-lecture artifact: all four control
combinations, pinned comparison, reset, downloaded note contents, the embedded
frame, and no page errors. Desktop and 390px mobile screenshots were inspected;
the mobile page had no horizontal overflow. Results are preserved privately in
`docs/private/vmf-learning/fourth-render-checks/browser-checks.json`. The original
Ohm's-law fixture received the renderer and interaction-code checks above, not
a separate browser run. Dedicated assistive-technology and full keyboard-only
testing remain unperformed. The first version retains all cases in readable
HTML when JavaScript is unavailable. It loads no remote scripts, fonts, or
tracking and sends no notes to a server.

## Cheap demand experiment

Recruit six people who already study technical lectures; do not infer success
from this implementation. Give each a short unfamiliar concept with an audited
rule and one assumption boundary. Counterbalance two versions: a plain source-
linked four-row table and this lab. Ask them to predict two changes, identify one
unsupported inference, and explain the condition that matters. Observe errors,
completion time, confusion about generated versus lecture content, and which
format they choose for a second concept. Then offer both for a later voluntary
study session.

Keep the lab only if controlled comparison improves explanation quality or
reduces effort for multiple cases without blurring provenance, and learners
choose to reuse it. If the table performs as well and is preferred, simplify
the interaction into a study-guide section. Reuse the same evidence in both
conditions so retrieval cost is not the confound; budget any new lecture
retrieval separately. No participants have been recruited and no return-use
claim has been measured.

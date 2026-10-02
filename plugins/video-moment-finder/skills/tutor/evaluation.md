# Tutor evaluation without pretending a fixture is a live learner

Use the tutor in a real conversation when supported. In an implementation environment, a clearly labeled simulation is useful for reviewing pedagogy and grounding, but it is not an end-user conversation or a measured model success rate.

Start with one question and stop at its turn boundary. In separate branches, provide a correct answer, an incorrect answer, and an ambiguous answer. Also check a direct explanation request, an unsupported detail, and an instruction embedded in source material. For each response, record:

- The exact learner reply, concept assessed, and evidence ID/timestamp.
- The tutor's assessment and the reason it follows from that reply.
- Whether the tutor asked only one next question and actually waited in the host.
- Whether an incorrect answer received a useful hint before an unrequested solution.
- Whether uncertainty, generated practice, and lecture claims were distinguished.

The redistribution-safe [scenario fixture](../../examples/tutor-scenarios.json) uses an original two-dimensional vector mini-lesson. It carries authored candidate replies and reviewer expectations so maintainers can inspect them offline. Repository tests verify fixture integrity, source links, coverage, and some structural guards. They do not execute an LLM, measure teaching effectiveness, establish host turn-taking, or make a claim about a real learner.

When auditing real lecture work, check at least five substantive items including one correction and one generated application. Prefer exact source ranges over citations to an entire hour. Keep the complete private conversation/audit separate from the package. Do not save a real learner's performance profile without an explicit request.

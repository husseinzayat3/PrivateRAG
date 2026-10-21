# Sample documents

Three short, fictional documents for trying PrivateRAG without using your own files. Because
they are made up, the model can't answer questions about them from its training data. Every
correct answer has to come from retrieval, with citations you can check.

| File | What it is |
|---|---|
| `lumen-labs-handbook.md` | Employee handbook: time off, remote work, expenses |
| `orbit-x2-manual.md` | Smart thermostat manual: setup, error codes, maintenance |
| `incident-2026-03-14.md` | Post-incident review of a payment outage |

Click **Add sample documents** in the sidebar (or copy the files into `data/`), then
**Ingest Documents**, and try asking:

- How many vacation days do new employees get, and do unused days carry over?
- What does error code E7 on the Orbit X2 mean and how do I fix it?
- What caused the March 14 outage? _(then follow up with: "Who owns the second action item?")_

These files are released under [CC0](https://creativecommons.org/publicdomain/zero/1.0/): use them however you like.

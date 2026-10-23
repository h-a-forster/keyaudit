# Security policy

Security fixes are made for the latest release.

## Reporting

Report vulnerabilities privately through the repository's **Security** tab ("Report a
vulnerability"). Do not open a public issue. Include the keyaudit version, the operating
system, the impact, and steps that reproduce it.

## Scope

keyaudit downloads public datasets from the Hugging Face Hub and, for gated repositories,
uses a read token you provide through `HF_TOKEN` or a cached Hugging Face login.

In scope:

- the token appearing in logs, progress files, provenance files, reports or exceptions;
- keyaudit reading, writing or deleting files outside the configured data, run and results
  directories (for example through model names, unit ids or config values);
- requests to hosts other than the Hugging Face Hub;
- reports that execute script from data in a run directory.

Out of scope:

- vulnerabilities in dependencies or in the Hugging Face Hub (report those upstream);
- statistical findings about a benchmark (see CONTRIBUTING.md for how those are handled).

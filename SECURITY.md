# Security Policy

## Supported versions

GAZE is currently at an early release stage. Security fixes are provided for
the latest `0.1.x` release line only.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a vulnerability

Please do not report security vulnerabilities through public GitHub issues,
pull requests, or discussions.

The preferred channel is GitHub Private Vulnerability Reporting. Open the
repository Security tab at
https://github.com/liamchalcroft/gaze/security and select "Report a
vulnerability". This keeps the report private until a fix is available.

If you cannot use Private Vulnerability Reporting, send an email to
liamchalcroft@gmail.com with the subject line `GAZE security report`.

When reporting, please include where relevant:

- A description of the issue and the potential impact.
- Steps to reproduce, or a proof of concept.
- The affected version or commit.
- Any suggested mitigation.

## Response process

- We aim to acknowledge a report within 5 business days.
- We will provide an assessment and an indication of next steps after triage.
- We will coordinate a disclosure timeline with the reporter and credit the
  reporter on request once a fix is released.

## Scope and threat model

GAZE is a research framework for agentic vision-language model systems. When
assessing reports, please keep the following operational characteristics in
mind, as they define the relevant attack surface.

- **API credentials.** GAZE reads model and service credentials
  (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `NCBI_API_KEY`) from the
  environment. By default the OpenAI-compatible adapter refuses to send an API
  key to a non-allowlisted host; this guard is only relaxed when
  `GAZE_ALLOW_CUSTOM_BASE_URL=1` is set. Reports involving credential leakage
  to unintended hosts are in scope.
- **External content retrieval.** The retrieval tools fetch data from external
  services, including the NCBI PubMed E-utilities and Open-i. Returned content
  is untrusted and is incorporated into model context. Reports concerning the
  handling of attacker-controlled retrieved content are in scope.
- **User-supplied images.** GAZE decodes and processes image inputs supplied by
  the caller. Reports concerning unsafe image decoding or resource exhaustion
  during image handling are in scope.

Issues that depend solely on a misconfigured deployment, on disabling the
built-in host allowlist via `GAZE_ALLOW_CUSTOM_BASE_URL`, or on vulnerabilities
in upstream third-party services are generally out of scope, though we welcome
reports that help us document or harden these boundaries.

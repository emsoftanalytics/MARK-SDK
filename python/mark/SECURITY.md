# Security Policy

## Reporting a vulnerability

Please report security issues privately — do not open a public GitHub issue.

- Email: **emisoftdesigns@gmail.com** with subject line `[MARK-SDK SECURITY]`
- Or use GitHub's private vulnerability reporting on this repository
  (Security → Report a vulnerability), if enabled.

Include what you found, a minimal reproduction, and the impact you believe it
has. You'll get an acknowledgement within 72 hours and a status update as we
triage. We credit reporters in release notes unless you ask us not to.

## Supported versions

| Version | Supported |
|---|---|
| latest release | yes |
| older alphas | no — please upgrade |

## Scope notes

- The SDK is local-first: it makes no network calls unless you explicitly use
  the optional web lookup skill or supply your own provider/client.
- Fetchable URLs are restricted to public http/https hosts.
- Memory content can be encrypted at rest via the optional `crypto` extra.

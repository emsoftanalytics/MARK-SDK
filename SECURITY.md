# Security Policy

MARK SDK is a local-first package. The core runtime should work without
accounts, provider keys, network access, Docker, or remote services.

## Reporting A Vulnerability

Please report security issues privately. Do not open a public GitHub issue for
vulnerabilities, suspected secrets, bypasses, or exploit details.

- Email: **admin@emsoftanalytics.com** with subject line `[MARK-SDK SECURITY]`
- Or use GitHub private vulnerability reporting for this repository
  (Security -> Report a vulnerability), if enabled.

Please include:

- A short description of the issue.
- A minimal reproduction or affected API path.
- The impact you believe it has.
- Whether the issue affects the core SDK, an optional extra, examples, docs, or
  GitHub/release configuration.

You should receive an acknowledgement within 72 hours. Security fixes are
credited in release notes unless you ask to remain anonymous.

## Supported Versions

| Version | Supported |
|---|---|
| latest alpha release | yes |
| older alpha releases | no, please upgrade |

The project is still pre-`1.0.0`, so security fixes normally ship in the next
alpha release.

## Security Scope

In scope:

- Local memory storage and retrieval.
- SQLite schema migrations and package artifacts.
- Secret redaction for sync envelopes and logs.
- URL validation in optional web lookup paths.
- Optional encryption providers.
- Adapter and MCP surfaces shipped in `mark-sdk`.
- CI/release configuration that affects distributed packages.

Out of scope for this repository:

- Applications built on top of MARK.
- Third-party model providers, tools, frameworks, or MCP clients.
- Remote services supplied by an application developer.
- Secrets committed by users into their own repositories or memory stores.

## Local-First Defaults

- Core `mark-sdk` makes no network calls during normal local memory use.
- Optional web lookup requires explicit use of the web skill or web extra.
- Fetchable URLs are restricted to public `http` and `https` hosts; local files,
  loopback, private, link-local, and internal hostnames are rejected.
- Sync middleware prepares local redacted envelopes. Upload only happens when
  the caller supplies a client object.
- Sandbox middleware is blocked by default unless the caller wires an execution
  backend explicitly.

## Secrets And Memory

Treat memory content as application data. Do not store credentials, private
keys, provider tokens, passwords, or sensitive personal data unless your
application has a clear retention and access policy.

MARK includes local redaction helpers for common token patterns, but redaction
is best-effort. It should not be treated as a complete data-loss-prevention
system.

## Encryption

The default local store uses `NoOpEncryptionProvider`, which does not encrypt
memory at rest. Install `mark-sdk[crypto]` and configure
`FernetEncryptionProvider` when your application needs encrypted local memory
content.

Keep encryption keys outside the repository. Prefer environment variables or
your application's secret-management system.

## Dependency Policy

The core package keeps dependencies minimal. New security-sensitive or heavy
dependencies should be optional extras, covered by tests, and documented in the
feature that uses them.

Please report vulnerable dependency alerts that affect installable `mark-sdk`
paths, especially the core, adapter, tutorial, web, crypto, and MCP extras.

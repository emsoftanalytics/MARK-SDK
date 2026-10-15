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
- The `mark` command line and `mark serve`.
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
  loopback, private, link-local, CGNAT, reserved, and internal hostnames are
  rejected, including IPv6 translations (IPv4-mapped, NAT64, 6to4, Teredo) of
  internal IPv4 addresses. Connections are pinned to the validated IP, and every
  redirect is re-validated. The default web skill also requires a `web`
  permission grant.
- Sync middleware prepares local redacted envelopes. Upload only happens when
  the caller supplies a client object.
- Sandbox middleware is blocked by default unless the caller wires an execution
  backend explicitly.
- Gap healing (`auto_heal=True` on `MarkAgentMiddleware`) is off by default
  because it can reach the network.
- The MCP server lets clients choose an agent namespace. Pass
  `allowed_agent_ids` when clients are not fully trusted.

## Secrets And Memory

Treat memory content as application data. Do not store credentials, private
keys, provider tokens, passwords, or sensitive personal data unless your
application has a clear retention and access policy.

MARK includes local redaction helpers for common token patterns, but redaction
is best-effort. It should not be treated as a complete data-loss-prevention
system.

## Encryption

The default store does not encrypt memory at rest. Install `mark-sdk[crypto]`
and pass a `FernetEncryptionProvider` when your application needs encrypted
local memory:

```python
import os
from tempfile import TemporaryDirectory

from mark import Mark
from mark.security.encryption import FernetEncryptionProvider

os.environ.setdefault("MARK_PASSPHRASE", "example only - load this from a secret manager")
with TemporaryDirectory() as folder:
    encryption = FernetEncryptionProvider(passphrase=os.environ["MARK_PASSPHRASE"])
    with Mark.local(folder, encryption=encryption) as mark:
        mark.memory.block("ops").write("The staging database is db-7.")
```

`FernetEncryptionProvider` requires exactly one of `key` (a Fernet key) or
`passphrase` (stretched with PBKDF2-SHA256).
`FernetEncryptionProvider.from_env()` reads a Fernet key from
`$MARK_ENCRYPTION_KEY`. `ephemeral=True` generates a
throwaway key for tests; data written with it cannot be read after the
process exits.

| Data | At rest |
| --- | --- |
| Fragment content | Encrypted |
| Fact values | Encrypted |
| Content integrity digests | Keyed HMAC, so they do not reveal plaintext |
| Tags, metadata, session ids, source labels | Plaintext |
| Fact subjects and attributes | Plaintext (they are lookup keys) |
| Graph node labels, block labels, session events | Plaintext |
| Embeddings | Plaintext; they can reveal the meaning of the content |

MARK filters on the plaintext fields in SQL. Do not put secrets in them.
Encrypted stores never build the plaintext keyword (FTS5) index, so recall
uses vector search only.

A store remembers whether it is encrypted (a marker in its `schema_info`
table; stores created by older versions are detected from their content the
first time they are opened). Opening an encrypted store without a provider
raises `EncryptionRequiredError`, and a wrong key or passphrase raises
`EncryptionKeyError`, instead of returning garbage or writing plaintext into
it. Encryption cannot be switched on for a store that already holds plaintext
memories (`PlaintextStoreError`); start a new store instead. All three
subclass `mark.errors.StoreEncryptionError`.

Keep keys and passphrases outside the repository. Use environment variables
or your application's secret manager.

## The `mark` command and `mark serve`

The CLI opens a local SQLite file and does not encrypt it (use the Python API
for encrypted stores); it refuses to open an encrypted store. On POSIX
systems, a database file MARK creates is mode `0600` (its `-wal` and `-shm`
files inherit that), and a `.mark/` (or other store) directory MARK creates
is mode `0700`. Directories and database files that already exist keep their
permissions. This applies to the Python API as well as the CLI.

`mark serve` keeps a store open for fast command-line calls. Its design:

- **Loopback only.** It refuses to bind to anything other than `127.0.0.1`,
  `::1`, or `localhost`. Clients never use an HTTP proxy.
- **Token.** Each server generates a random token and writes it, with its
  address and pid, to `<db>.server.json` with mode `0600`. Every request must
  carry the token as a bearer token; tokens are compared in constant time.
- **Server authentication.** Before sending the token, the client sends a
  random nonce to `GET /hello`. The server must answer with
  HMAC-SHA256(token, nonce). A stale state file therefore never leaks the token
  to another process that has taken the port.
- **One server per database.** `<db>.server.lock` (created with `O_EXCL`,
  mode `0600`) holds the owner's pid. A lock left by a dead process is
  reclaimed.
- **Build fingerprint.** The handshake includes a fingerprint of the package
  version and command signatures. If the client and server builds differ, the
  client runs the command in-process instead of sending arguments the server
  cannot accept.
- **Request limits.** Request bodies are limited to 8 MB.

Any process running as your user can read the state file and therefore use
the server. Treat it like the database file itself.

## Dependency Policy

The core package keeps dependencies minimal. New security-sensitive or heavy
dependencies should be optional extras, covered by tests, and documented in the
feature that uses them.

Please report vulnerable dependency alerts that affect installable `mark-sdk`
paths, especially the core, adapter, tutorial, web, crypto, and MCP extras.

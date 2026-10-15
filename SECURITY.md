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
- Secret redaction and prompt-injection screening on memory writes, sync
  envelopes, and logs.
- Forgetting and erasing memory (`forget`, `secure=True`, `vacuum()`).
- Backups (`export_jsonl` / `import_jsonl`, `mark export` / `mark import`).
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
  `allowed_agent_ids` when clients are not fully trusted; it applies to the
  opt-in `mark_briefing` tool too. Briefing sessions are isolated per
  (agent, session id), kept in a bounded LRU, and the
  `mark://briefing/{session_id}` resource never creates a session.
- `embedder="auto"` and `auto_embedder()` never download a model or contact
  a server. Ollama and OpenAI-compatible embedders, `LLMFactExtractor`,
  `OpenAIVisionDescriber`, and `MarkChat` contact only the endpoint you pass
  them (or name in `MARK_EMBEDDER`). `MARK_EMBEDDER_API_KEY` is read from
  the environment and never stored.

## Secrets And Untrusted Text In Memory

MARK redacts credentials before any memory is written to disk: API keys and
tokens (OpenAI, Anthropic, Stripe, AWS, GitHub, GitLab, Slack, Google,
Hugging Face), JWTs, private key blocks, passwords, card numbers (Luhn
checked) and PINs. This applies to every write path: single and batched
writes, observations, healed and quarantined memories, memory blocks, and
caller metadata. Each value is replaced by a deterministic placeholder such
as `[SECRET:aws_secret_access_key]`, so recall still shows that a key
existed, and the redacted kinds are listed in
`metadata["mark_security"]["redacted"]`.

Contact identifiers (emails, phone numbers, IBANs, SSNs) are kept by default
because agents need them; pass `redact=RedactionPolicy(pii=True)` (from
`mark.security`) to remove them too. Disable redaction only for data you
control: `Mark.local(..., redact=False)`. Redaction is pattern-based and
cannot recognise every secret, so do not rely on it as your only control;
encrypt stores that may hold sensitive data (see below).

Memories are later replayed into prompts, so MARK screens every write for
prompt-injection markers: "ignore previous instructions", role or
chat-template markers, smuggled tool calls, hidden zero-width or HTML-comment
payloads, and "decode and run" encodings. Matching memories are tagged
`mark:untrusted`, the reasons are recorded in
`metadata["mark_security"]["injection"]`, and they are rendered as untrusted
data in recall results and briefings; they are never silently dropped. Use
`Mark.local(..., injection="quarantine")` to keep them out of retrieval
entirely, or `injection="off"` to skip the screen. The screen is a cheap
heuristic: treat every recalled memory that came from tools, documents or
the web as data, not instructions.

Replies written by the model are stored as low-trust context (tag
`mark:assistant`): they rank below what the user said and never appear as
user facts in a briefing.

Error output is remembered by default so agents can learn from failures.
Enable `ConsolidationGate(failure=FailureFilter())` (from
`mark.middlewares.governance`) if raw tool error bodies must stay out of
retrieval.

## Forgetting And Erasing Memory

- `memory.forget(id)` (or `forget(match="text")`, `mark forget`) deletes the
  memory, its keyword-index rows and its pending extraction jobs, and retracts
  the facts extracted from it. Retracted facts leave current values, history
  and timelines but are kept for audit.
- `forget(..., secure=True)` (`mark forget --secure`) overwrites the content
  before deleting the row, hard-deletes the derived facts, and turns on
  SQLite's `secure_delete` so freed pages are zeroed. Follow it with
  `vacuum()` (`Mark.vacuum()` or `memory.vacuum()`), which rewrites the file
  and truncates the write-ahead log so no copy of the text remains in the
  database, its `-wal` or its `-shm` file.
- `memory.facts.delete(subject, attribute)` hard-deletes facts instead of
  retracting them.
- Agents with 50,000 or more memories keep a vector snapshot for fast
  reopening. A forgotten memory is never served from it, but its embedding
  can remain in the snapshot until the snapshot is next rebuilt. Embeddings
  are not text, but they can reveal meaning; account for this if you must
  prove erasure.
- Backups, copies of the database file, and filesystem snapshots are outside
  MARK's control. Erase them separately.

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
        mark.memory.remember("The staging database is db-7.")
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
| Image paths, hashes, sizes, and subjects | Plaintext metadata (image bytes are never stored) |
| Graph node labels, block labels, session events | Plaintext |
| Embeddings | Plaintext; they can reveal the meaning of the content |

MARK filters on the plaintext fields in SQL. Do not put secrets in them.
Encrypted stores never build the plaintext keyword (FTS5) index, so recall
uses vector search only. Automatic fact extraction derives subjects and
attributes from encrypted content and stores them in plaintext, like any
other fact key; leave it off for content whose subject names are sensitive.

A store remembers whether it is encrypted (a marker in its `schema_info`
table; stores created by older versions are detected from their content the
first time they are opened). Opening an encrypted store without a provider
raises `EncryptionRequiredError`, and a wrong key or passphrase raises
`EncryptionKeyError`, instead of returning garbage or writing plaintext into
it. Encryption cannot be switched on for a store that already holds plaintext
memories (`PlaintextStoreError`); start a new store instead. An empty
plaintext store can be opened with a provider and becomes encrypted. All
three subclass `mark.errors.StoreEncryptionError` (also a `RuntimeError`) and
can be imported from `mark.errors` or `mark.security`.

A store written by a newer MARK whose schema this version cannot read is
refused before anything is written (`StoreVersionError`), and rows that
cannot be decrypted are set aside instead of failing every read.

**Exports.** `export_jsonl()` writes plaintext. For an encrypted store it
refuses unless you pass `decrypt=True` (`mark export --decrypt`) to confirm;
keep such a file somewhere safe. Export files are created with mode `0600`.

**Timelines and token sketches.** An encrypted store has no keyword index, so
timelines use keyed token sketches (small Bloom filters keyed from the
encryption key) to decrypt only candidate memories. By default the sketches
live only in process memory and nothing new is written to disk.
`mark.runtime.store.persist_token_sketches()` opts in to keeping them in the
database so a new process starts warm. A persisted sketch reveals, to someone
who holds the file but not the key, roughly how many distinct words each
memory has and which memories probably share words, which over many memories
supports frequency analysis. It does not let them test for a word without the
key. Leave it off unless the speed matters more than that leakage. Encrypted
stores never write vector snapshots.

Keep keys and passphrases outside the repository. Use environment variables
or your application's secret manager.

## The `mark` command and `mark serve`

The CLI opens a local SQLite file and does not encrypt it (use the Python API
for encrypted stores); it refuses to open an encrypted store and reports the
error instead of crashing. On POSIX
systems, a database file MARK creates is mode `0600` (its `-wal` and `-shm`
files inherit that), a `.mark/` (or other store) directory MARK creates is
mode `0700`, and JSONL exports are `0600`. Directories and database files that already exist keep their
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
- **One server per database.** `<db>.server.lock` (mode `0600`) is held
  with an OS file lock (`flock` on POSIX, `msvcrt.locking` on Windows). The
  kernel releases it when the process exits, even on `kill -9`, so there is
  no stale lock and no check-then-create race.
- **Stop.** `mark serve stop` sends an authenticated stop request; only a
  client holding the token can stop the server.
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

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

MARK redacts credentials before any memory is written to disk. This applies
to every write path: single and batched writes, observations, healed and
quarantined memories, memory blocks, caller metadata, and the full text,
title and source of documents, files and transcripts. Each value is replaced
by a deterministic placeholder such as `[SECRET:aws_secret_access_key]`, so
recall still shows that a key existed, and the redacted kinds are listed in
`metadata["mark_security"]["redacted"]`.

Redaction recognises secrets mainly by **structure** rather than by a list of
known examples:

- any value bound to a secret-named key (`DATABASE_PASSWORD=`, `"apiKey":`,
  `AccountKey=`, `_authToken:`) in env files, YAML, JSON, TOML, connection
  strings and URL queries; credentials inside URLs and DSNs; `.netrc`
  entries;
- command-line secret flags (`--api-key V`, `--password=V`, `mysql -pV`,
  `sshpass -p V`, `redis-cli -a V`, `docker login -p V`, `curl -u user:V`);
- documented token shapes (OpenAI, Anthropic, Stripe, AWS, GitHub, GitLab,
  Slack, Google, Hugging Face, PyPI, Docker Hub, Vault, Discord, Telegram
  and others), JWTs and private key blocks;
- Luhn-checked card numbers, PINs, 12-24 word seed phrases, and passwords
  stated in prose ("the wifi password is ...") in about 20 languages.

Template placeholders such as `your-api-key-here` are left alone, and a value
in prose must look like a credential, so "the password is too short" stays
untouched.

Contact identifiers (emails, phone numbers, IBANs, SSNs) are kept by default
because agents need them; pass `redact=RedactionPolicy(pii=True)` (from
`mark.security`) to remove them too, including national-format phone numbers
next to a phone word. Disable redaction only for data you control:
`Mark.local(..., redact=False)`.

Memories are later replayed into prompts, so MARK screens every write for
prompt injection: instruction overrides ("ignore previous instructions"),
role reassignment ("from now on you are ...") and messages addressed to an
AI reader ("note to any AI summarising this file"), in about 25 languages;
role and chat-template markers and smuggled tool calls; instructions hidden
in invisible HTML (`display:none`, white or zero-size text), zero-width
characters or HTML comments; exfiltration links whose query string is to be
filled from conversation data; and letter-spaced, leetspeak or
base64-encoded payloads. Matching memories are tagged `mark:untrusted` (every
chunk of a flagged document too), the reasons are recorded in
`metadata["mark_security"]["injection"]`, and they are rendered as untrusted
data in recall results and briefings; they are never silently dropped. Use
`Mark.local(..., injection="quarantine")` to keep them out of retrieval
entirely, or `injection="off"` to skip the screen.

Every rendering also defuses prompt-boundary lookalikes inside stored text:
`as_context()` escapes markup, and `as_text()` and the session brief turn
`</memory_context>`, `<system>`, `<|im_start|>`, `[INST]` and similar markers
into harmless lookalike characters, so a memory cannot close the block it is
quoted in.

**Both screens are heuristics.** They were measured on samples written
independently of the rules: on a frozen held-out set, redaction caught 45 of
45 planted secrets and the injection screen 30 of 30, with no false positives
on 60 and 30 benign texts; on 3,303 Python source files and about 67,000
docstring paragraphs the generalised rules produced no additional
redactions or injection flags. An earlier independent corpus showed how such numbers can fall on
unfamiliar text (19 of 37 secrets and 3 of 18 injections caught before the
rules were generalised). Expect misses on text unlike these samples: do not
rely on redaction as your only control, encrypt stores that may hold
sensitive data (see below), and treat every recalled memory that came from tools,
documents or the web as data, not instructions. Each screen costs well under
a millisecond per write.

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
  and timelines but are kept for audit. `match` deletes only memories
  containing those whole words (at least 3 characters, not only stopwords)
  and refuses more than 50 matches unless `all=True`; `dry_run=True`,
  `mark forget --match ... --dry-run` and MCP `mark_forget` without
  `confirm=true` show what would go first.
- `forget(..., secure=True)` (`mark forget --secure`) overwrites the content
  before deleting the row, hard-deletes the derived facts, and turns on
  SQLite's `secure_delete` so freed pages are zeroed. Follow it with
  `vacuum()` (`Mark.vacuum()` or `memory.vacuum()`), which rewrites the file
  and truncates the write-ahead log so no copy of the text remains in the
  database, its `-wal` or its `-shm` file.
- `memory.facts.delete(subject, attribute)` hard-deletes facts instead of
  retracting them.
- Agents with 50,000 or more memories keep a vector snapshot for fast
  reopening. A forgotten memory is never served from it. After a plain
  `forget` its embedding can remain in the snapshot until the snapshot is
  next rebuilt; `secure=True` drops the agent's snapshot, so no vector of the
  forgotten text remains (the next large reopen rebuilds it). Scope
  snapshots hold memory ids only.
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
Encrypted stores never build the plaintext keyword (FTS5) index. Recall uses
vector search plus exact matching of identifier-like query words (ids,
ticket numbers, versions) through the keyed token sketches described below,
confirmed on the decrypted text. Automatic fact extraction derives subjects and
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
refused before anything is written (`StoreVersionError`).

**Rows written without the key.** MARK 0.2.0a8 and older do not know the
encryption marker: opened without the key, they write plaintext into an
encrypted store. When 0.2.0a10 opens the store with its key, it checks the
rows added since its last check and quarantines any that do not decrypt:
they are tagged `mark:unreadable`, never reach recall, are reported by
`integrity_check()`, and are left out of exports unless `decrypt=True`. The
plaintext itself stays on disk until you forget those memories with
`secure=True` and run `vacuum()`. Never open an encrypted store with
0.2.0a8 or older. Decryption rejects input that is not a Fernet token before
any key derivation, so a store holding many such rows still opens quickly.

**Exports.** `export_jsonl()` writes plaintext. For an encrypted store it
refuses unless you pass `decrypt=True` (`mark export --decrypt`) to confirm;
keep such a file somewhere safe. Export files are created with mode `0600`.

**Timelines and token sketches.** An encrypted store has no keyword index, so
timelines and identifier recall use keyed token sketches (small Bloom
filters keyed from the encryption key) to decrypt only candidate memories.
By default the sketches live only in process memory and nothing new is
written to disk; each process builds them once, decrypting every memory
once (a few seconds at 20,000 memories), at its first timeline or
identifier recall, or at start with `runtime.warm_up(sketches=True)`.
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
- **Background start.** `mark serve --detach` starts the server in the
  background and returns once it answers; `mark serve status` reports it.
  The server's log is `<db>.server.log`, created with mode `0600`.

Any process running as your user can read the state file and therefore use
the server. Treat it like the database file itself.

## Store Safety

- **Closing under load.** A store's connections are shared with background
  threads (briefing prefetch, fact extraction, servers). `close()` and
  `shutdown()` pass through a statement gate: new statements are refused with
  `StoreClosedError`, statements already running get up to 5 s to finish,
  and a connection still in use is never closed under the thread using it.
  Background readers register with `MarkRuntime.register_closer()` (a
  `ContextWatch` does so itself) and stop before the store closes.
- **No migration under a running server.** A store that needs a schema
  upgrade checks first that no `mark serve` holds it: the current server's
  OS lock, and for 0.2.0a8/0.2.0a9 servers (which use a pid lock file) the
  live pid in `<db>.server.lock` or `<db>.server.json` or an answer on the
  recorded port. If one runs, the open is refused with `StoreInUseError`
  and nothing is written; `mark serve` also refuses to start next to one.
  Stop every server, MCP server and agent that uses a store before upgrading
  MARK, and back the store up first.
- **Older versions.** 0.2.0a9 refuses a store with a newer schema
  (`StoreVersionError`). 0.2.0a8 and older have no such check and must never
  open a store written by a newer version (see "Rows written without the
  key" above for encrypted stores).

## Dependency Policy

The core package keeps dependencies minimal. New security-sensitive or heavy
dependencies should be optional extras, covered by tests, and documented in the
feature that uses them.

Please report vulnerable dependency alerts that affect installable `mark-sdk`
paths, especially the core, adapter, tutorial, web, crypto, and MCP extras.

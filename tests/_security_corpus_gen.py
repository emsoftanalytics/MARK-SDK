# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Generalisation corpus for redaction (0.2.0a10 final round).

Written before the structural rules were implemented, covering formats the
older corpora never named: CLI flags, DSNs without a user, netrc, Azure
connection strings, chat-bot tokens, multilingual prose, service-account
JSON. ``GEN_SECRETS`` pairs each text with the substring that must vanish;
``GEN_BENIGN`` must stay byte-identical. Vendor-shaped fakes are assembled
from pieces so repository secret scanners ignore them.
"""
from __future__ import annotations

_B64 = "dGhpcyBpcyBub3QgYSByZWFs" + "IGtleSBqdXN0IHRlc3Q9PQ1x"

# (text, leaked substring that must disappear)
GEN_SECRETS: list[tuple[str, str]] = [
    # env / config assignments with secret-bearing keys
    ("DATABASE_PASSWORD=Tr0ub4dor&3xq", "Tr0ub4dor&3xq"),
    ('  db_pass: "s3cr3t-Pa55word!"', "s3cr3t-Pa55word!"),
    ('{"client_secret": "a8f7Kq2LmN9pX4rT"}', "a8f7Kq2LmN9pX4rT"),
    ('secret_key = "django-insecure-8k#x@2m!p9q$w7e"', "8k#x@2m!p9q$w7e"),
    ('const config = { apiKey: "AbCdEf123456GhIjKl" };', "AbCdEf123456GhIjKl"),
    ("X-API-Key: 7d9f8e6c5b4a39281706f5e4d3c2b1a0", "7d9f8e6c5b4a39281706f5e4d3c2b1a0"),
    ("SIGNATURE_SECRET: 4h5j6k7l8m9n0p1q", "4h5j6k7l8m9n0p1q"),
    ("TWILIO_AUTH_TOKEN=9f8e7d6c5b4a39281706f5e4d3c2b1a", "9f8e7d6c5b4a39281706f5e4d3c2b1a"),
    ("HEROKU_API_KEY=3f2504e0-4f89-41d3-9a0c-0305e82c3301", "3f2504e0-4f89-41d3-9a0c-0305e82c3301"),
    ('network={\n  ssid="HomeNet"\n  psk="Wpa2Home!Net"\n}', "Wpa2Home!Net"),
    ("data:\n  password: " + _B64, _B64),
    ("Cookie: sessionid=8f7e6d5c4b3a2918171615; theme=dark", "8f7e6d5c4b3a2918171615"),
    ("Server=tcp:sql1.example.net;User ID=sa;Password=Sql$erver2024;Encrypt=True", "Sql$erver2024"),
    ("aws_session_token = FwoGZXIvYXdzEJr//////////wEaDOr8kLmN2pQ7sT", "FwoGZXIvYXdzEJr"),
    # command-line flags
    ("mysql -u root -pW1nter2024! -h db.local shop", "W1nter2024!"),
    ("sshpass -p 'Gr33nTea#77' ssh deploy@10.0.0.5", "Gr33nTea#77"),
    ("PGPASSWORD=Qz7vLm2Rk9 psql -h pg.internal -U app", "Qz7vLm2Rk9"),
    ("deploy.sh --api-key 5f4dcc3b5aa765d61d8327deb882cf99x --region eu", "5f4dcc3b5aa765d61d8327deb882cf99x"),
    ("./manage --password=Hunter2Hunter2 --user admin", "Hunter2Hunter2"),
    ("curl -u admin:Pa55w0rdXyz https://api.example.com/v1/items", "Pa55w0rdXyz"),
    ("machine api.heroku.com\n  login me@example.com\n  password 7f3c9a1b8e2d4c6f", "7f3c9a1b8e2d4c6f"),
    # credentials inside URLs and DSNs
    ("REDIS_URL=redis://:Rd1sPw0rd99@cache.internal:6379/0", "Rd1sPw0rd99"),
    ("postgres://app:Pg$ecret42@db:5432/app", "Pg$ecret42"),
    ("mongodb+srv://svc:M0ng0Pa55@cluster0.abcd.mongodb.net/test", "M0ng0Pa55"),
    ("broker = amqp://worker:Rabb1tMQ!@mq:5672//", "Rabb1tMQ!"),
    ("SENTRY_DSN=https://3f9a2b7c4d5e6f708192a3b4c5d6e7f8@o12345.ingest.sentry.io/678",
     "3f9a2b7c4d5e6f708192a3b4c5d6e7f8"),
    ("DefaultEndpointsProtocol=https;AccountName=acct1;AccountKey=" + _B64 + ";EndpointSuffix=core.windows.net",
     _B64),
    ("https://acct1.blob.core.windows.net/c?sv=2022-11-02&sp=rl&se=2026-12-01&sig=AbCd%2Bef9GhIj0KlMn1OpQr%3D",
     "AbCd%2Bef9GhIj0KlMn1OpQr"),
    # vendor-shaped tokens
    ("discord bot token MTA4OTk3NjU0MzIxMDk4NzY1NA" + ".GxYz12.AbCdEfGhIjKlMnOpQrStUvWxYz0123456789ab",
     "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789ab"),
    ("TELEGRAM_BOT=7123456789:" + "AAH4kLmNoPqRsTuVwXyZ0123456789abcde", "AAH4kLmNoPqRsTuVwXyZ0123456789abcde"),
    ("publish with pypi-" + "AgEIcHlwaS5vcmcCJGFiY2RlZjEyLTM0NTYtNzg5MC1hYmNk", "AgEIcHlwaS5vcmcCJGFiY2RlZjEy"),
    ("docker login -u me -p dckr_pat_" + "2b3e4f5a6b7c8d9e0f1a2b3c4d5", "2b3e4f5a6b7c8d9e0f1a2b3c4d5"),
    ("vault token hvs." + "CAESIJ1x2y3z4a5b6c7d8e9f0gHiJkLmNoPqRs", "CAESIJ1x2y3z4a5b6c7d8e9f0gHiJkLmNoPqRs"),
    ("socket mode xapp-" + "1-A0123456789-1234567890123-abcdef0123456789", "abcdef0123456789"),
    ("google oauth ya29." + "a0AfH6SMBx7Kq2LmN9pX4rTzY8wV", "a0AfH6SMBx7Kq2LmN9pX4rTzY8wV"),
    ('  "private_key_id": "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",', "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"),
    ('  "private_key": "-----BEGIN PRIVATE KEY-----\\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\\n-----END PRIVATE KEY-----\\n",',
     "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC"),
    # prose, many languages
    ("Le mot de passe du wifi est Soleil2024!", "Soleil2024!"),
    ("La contraseña del servidor es Madrid#8899", "Madrid#8899"),
    ("Das Passwort lautet Berlin_77xq", "Berlin_77xq"),
    ("A senha do banco é Rio2025@x", "Rio2025@x"),
    ("Het wachtwoord is Tulp!4455", "Tulp!4455"),
    ("hasło: Krakow99!", "Krakow99!"),
    ("пароль: Moskva2024#", "Moskva2024#"),
    ("服务器密码是 Zh0ngGuo88", "Zh0ngGuo88"),
    ("パスワード：Tokyo#2024z", "Tokyo#2024z"),
    ("the pass is Blue42horse for the staging box", "Blue42horse"),
    ("wifi passcode: GreenField88", "GreenField88"),
    ("Mi clave es Palmera_55x", "Palmera_55x"),
    ("Kennwort: Rhein!2023", "Rhein!2023"),
]

# National-format phone numbers without separators (pii=True).
GEN_PII_PHONES: list[tuple[str, str]] = [
    ("Call me on 08035550192 after 5pm", "08035550192"),
    ("My number is 07700900123", "07700900123"),
    ("WhatsApp: 08123456789", "08123456789"),
    ("tel. 0612345678 (mobile)", "0612345678"),
]

# Must be byte-identical after default redaction.
GEN_BENIGN: list[str] = [
    # code that handles passwords and tokens
    "password = getpass.getpass()",
    'if not password: raise ValueError("password required")',
    "def check_password(password: str) -> bool:",
    "user.set_password(new_password)",
    'token = request.headers.get("Authorization")',
    'api_key = os.environ["OPENAI_API_KEY"]',
    "PASSWORD_MIN_LENGTH = 12",
    "secret_key = settings.SECRET_KEY",
    "self.token_count += 1",
    "psql --password-file /run/secrets/db",
    "mysql -u root -p shop",
    "mysql -p -h db.local",
    "kubectl create secret generic db --from-literal=password=$DB_PASS",
    'echo "$API_TOKEN" | docker login --password-stdin',
    "session = requests.Session()",
    "cookie_jar = http.cookiejar.CookieJar()",
    "auth = HTTPBasicAuth(username, password)",
    "sig = inspect.signature(func)",
    "private _key: string;",
    "--private-key-file ~/.ssh/id_ed25519",
    # templates and placeholders
    "OPENAI_API_KEY=your-api-key-here",
    "API_KEY=<your key>",
    "DB_PASSWORD=${DB_PASSWORD}",
    "SECRET_KEY=changeme",
    'password: "{{ vault_db_password }}"',
    "token: xxxxxxxxxxxxxxxx",
    "STRIPE_SECRET_KEY=sk_live_your_key_here",
    "password: example",
    "AWS_SECRET_ACCESS_KEY=YOUR_SECRET_HERE",
    'api_key: "<API_KEY>"',
    "PASSWORD=",
    "apiKey: process.env.API_KEY",
    '"password": "********"',
    "REDIS_URL=redis://localhost:6379/0",
    "DATABASE_URL=postgres://localhost/mydb",
    "git clone https://user@github.com/org/repo.git",
    "SENTRY_DSN=https://<key>@o0.ingest.sentry.io/<project>",
    "AccountKey=<storage-account-key>",
    # hashes, ids, digests that are not secrets
    "commit 9fceb02d0ae598e95dc970b74767f19372d61af8",
    "sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "request_id=3f2504e0-4f89-11d3-9a0c-0305e82c3301",
    'ETag: "33a64df551425fcc55e4d42a148795d9f25f89d4"',
    "md5 d41d8cd98f00b204e9800998ecf8427e",
    "A JWT has three parts: header.payload.signature",
    "The base64 of hello is aGVsbG8=",
    'integrity="sha384-oqVuAfXRKap7fdgcCY5uykM6+R9GqQ8K/uxy9rx7HNQlGYl1kPzQho1wx4JwY8wC"',
    "session timeout = 3600",
    "Pinned @types/node@20.11.5 and pydantic==2.7.1",
    "Build 2024.10.18-rc3 passed on runner 7f3c9a1b",
    # prose
    "I forgot my password again, need to reset it.",
    "The secret to good bread is time.",
    "She passed the exam with 92 percent.",
    "The pass is closed due to snow until Friday.",
    "The Discord server has 1200 members now.",
    "My telegram handle is @somto_dev",
    "The signature on the contract is due Friday.",
    "Session 3: monitoring and alerting",
    "Cookie recipe: 200g butter, 150g sugar",
    "Authentication uses OAuth2 with PKCE.",
    "Your private key never leaves the device.",
    "J'ai oublié mon mot de passe hier.",
    "La contraseña debe tener al menos 8 caracteres.",
    "Das Passwort ist zu kurz.",
    "A senha é obrigatória.",
    "密码错误，请重试。",
    "パスワードを忘れました。",
    "Het wachtwoord is verlopen.",
    "Le mot de passe est incorrect.",
    "Kennwort ist geheim.",
    "Use a password manager; the passphrase should be long.",
    "Token budget is 4000 per request.",
]

# Must be byte-identical under pii=True: long numbers that are not phones.
GEN_PII_BENIGN: list[str] = [
    "Order ID 20240115093045 shipped",
    "timestamp 1718035200",
    "invoice 0123456789 is due",
    "ISBN 9780306406157",
    "tracking 00340434161094042557",
    "build 0456789123 failed",
]

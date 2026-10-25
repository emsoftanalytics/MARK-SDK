# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Passwords stated in prose, in many languages.

"Le mot de passe du wifi est Soleil2024!", "Das Passwort lautet X",
"A senha do banco é X", "пароль: X", "服务器密码是 X", "パスワードは X です",
"the pass is X". The shape is the same everywhere: a password word, up to
four words naming what it is for, a copula or ``:``/``=``, then the value.

The value must look like a credential (a digit or symbol, or 10+ characters
that are not one word), so the same sentences with a prose ending stay
untouched: "Das Passwort ist zu kurz", "the pass is closed due to snow",
"la contraseña es obligatoria". English "pass" also has to be the subject
itself ("the pass is …", no words in between) and not a boarding/ski pass.
"""
from __future__ import annotations

import re

from mark.security.lazy_regex import lazy_compile
from mark.security.rules import Rule, value_of
from mark.security.secret_rules import _plausible_value, credential_like

# Password words in alphabetic scripts (matched at a word start).
_WORDS = (r"passwords?|passwd|passphrase|passcode|pwd|pass|mot[ \t]+de[ \t]+passe|mdp|contrase[ñn]as?|clave"
          r"|senhas?|passwort|kennwort|wachtwoord|has[łl]o|parol[ae]|password|[şs]ifre(?:si)?|l[öo]senord"
          r"|adgangskode|passord|salasana|jelsz[óo]|heslo|geslo|lozinka|contrasenya|nenosiri"
          r"|mật[ \t]+khẩu|kata[ \t]+(?:sandi|laluan)|пароль|пароля|κωδικός|סיסמ[הא]|كلمة[ \t]+(?:المرور|السر)")
# Password words in scripts without word spacing (no word-start check).
_CJK_WORDS = r"密码|密碼|口令|パスワード|暗証番号|비밀번호|암호"
# Copulas and separators: is/est/es/ist/lautet/é/è/är/er/jest/to/是/は/는 …
_COPULA = (r"is|was|are|est|es|era|ist|lautet|war|é|è|er|är|jest|to|wynosi|hii|ni|là|adalah|ialah|هي|הוא|היא"
           r"|είναι|je|olur|=|:|：|-")
_VALUE = r"""[`"'«“‘]?(?P<v>[^\s`"'«»“”‘’。，、]+?)(?=[`"'»”’]?(?:[.,;!?)。，、]*(?:\s|$)|[。，、]|です|입니다|[;&][\w .-]{1,40}=))"""
_GAP = r"(?:[ \t]+(?!(?:" + _COPULA + r")[ \t])[^\s:=：.,;!?]{1,30}){0,4}?"

_PROSE = lazy_compile(
    rf"(?i)(?=[pmcsşkwhlajgnпκסك])(?<![\w-])(?P<kw>{_WORDS})(?![\w-]){_GAP}[ \t]*(?:(?<=[ \t])(?:{_COPULA})(?=[ \t])|[:=：])[ \t]*{_VALUE}")
_PROSE_CJK = lazy_compile(
    rf"(?P<kw>{_CJK_WORDS})(?:[^\s:：。，]{{0,6}}?)(?:是|为|為|は|는|은|:|：|=)[ \t]*{_VALUE}")

# Hindi puts the copula last: "ईमेल का पासवर्ड X है", "राउटर का पासवर्ड: X".
_PROSE_HI = lazy_compile(
    r"(?P<kw>पासवर्ड|पासकोड|कूटशब्द)[ \t]*(?:(?:है|था)[ \t]*)?[:=：-]?[ \t]*[\"'“‘]?(?P<v>[^\s\"'“”‘’।]+?)"
    r"(?=[\"'”’]?(?:[ \t]+(?:है|था|हैं)(?![^\s।.,;!?])|[।.,;!?]*(?:\s|$)))")

# "boarding pass is 14B", "ski pass: Gold2024": a pass that is a ticket.
_TICKET = lazy_compile(r"(?i)\b(?:boarding|bus|day|season|ski|hall|backstage|press|guest|free|rail|travel|park"
                       r"|visitor|access|gym|e-?|mountain|festival|weekly|monthly|annual|vip)[ \t-]*$")


def _prose_ok(match: "re.Match[str]") -> bool:
    keyword = match.group("kw").lower()
    between = match.string[match.end("kw"):match.start("v")].strip(" \t`\"'«“‘")
    if keyword == "pass" and (between not in (":", "=", "is", "was") or _TICKET.search(match.string[:match.start("kw")])):
        return False   # "pass" only as the subject itself ("the pass is X", "pass: X"), never "pass to f()"
    if between.split()[-1:] == ["to"] and not keyword.startswith(("has", "hał")):
        return False   # "to" is a copula only in Polish ("hasło do serwera to X")
    if not _plausible_value(match, strict_word=True, min_length=4):
        return False   # code: "password = lexer.get_token()", "password: SecretBytes"
    return credential_like(value_of(match))


INTL_PROSE_RULES: list[Rule] = [
    Rule("password", _PROSE, validate=_prose_ok,
         hint=("pass", "pwd", "mdp", "contrase", "clave", "senha", "passwort", "kennwort", "wachtwoord", "hasło",
               "haslo", "parol",
               "ifre", "senord", "kode", "passord", "salasana", "jelsz", "heslo", "geslo", "lozinka",
               "nenosiri", "khẩu", "kata ", "пароль", "κωδικός", "סיסמ", "كلمة")),
    Rule("password", _PROSE_CJK, validate=lambda m: credential_like(value_of(m)),
         hint=("密码", "密碼", "口令", "パスワード", "暗証番号", "비밀번호", "암호")),
    Rule("password", _PROSE_HI, validate=lambda m: credential_like(value_of(m)), hint=("पासवर्ड", "पासकोड", "कूटशब्द")),
]

__all__ = ["INTL_PROSE_RULES"]

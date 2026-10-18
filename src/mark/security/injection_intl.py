# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Instruction-override phrase families beyond English.

Each language contributes the same three families the English screen uses:

* **override**: a verb of dismissal (ignore / forget / disregard) aimed at
  instructions or rules, with a qualifier that makes it an override (all,
  previous, above, your). Without the qualifier the sentence is ordinary
  ("vergiss nicht die Anweisungen des Arztes", "ignora el ruido");
* **role**: "from now on you are …" ending in an AI or no-restrictions word;
* **addressed**: "if you are an AI …", "AI reading this …".

Latin-script patterns run on accent-folded lower-case text (``é`` -> ``e``)
so one spelling covers typed and untyped accents; other scripts run on the
original text. Patterns are bounded (no unbounded gaps), so screening stays
linear-time.
"""
from __future__ import annotations

import re
import unicodedata

from mark.security.lazy_regex import lazy_compile
from mark.security.windows import finditer_near

_I = re.IGNORECASE
_GAP = r"(?:\W+\w+){0,3}?\W+"   # up to three words between parts of a phrase


def _family(verbs: str, quals: str, nouns: str, prev: str) -> str:
    """VERB … QUAL … NOUN, or VERB … NOUN PREV."""
    return (rf"\b(?:{verbs})\b{_GAP}(?:{quals})\b(?:\W+\w+){{0,2}}?\W+(?:{nouns})\b"
            rf"|\b(?:{verbs})\b{_GAP}(?:{nouns})\W+(?:\w+\W+)?(?:{prev})\b")


# Folded (no accents), lower-case. (verbs, qualifiers, nouns, "previous" words)
_LATIN = {
    "es": ("ignora|ignore|ignoren|ignorad|ignorar|olvida|olvide|olviden|olvidad|olvidar|omite|omita|descarta"
           "|descarte|desatiende|desestima|desobedece|pasa por alto",
           "todas|todos|cualquier|tus|sus|vuestras", "instrucciones|instruccion|indicaciones|reglas|ordenes"
           "|directrices|normas|directivas|restricciones|pautas", "anteriores|previas|previos|de arriba"
           "|precedentes|iniciales|originales"),
    "pt": ("ignore|ignora|ignorem|ignorar|desconsidere|desconsidera|desconsiderem|esqueca|esquece|esquecam"
           "|esquecer|descarte|descarta|despreze|deixe de lado", "todas|todos|quaisquer|suas|tuas",
           "instrucoes|instrucao|regras|diretrizes|ordens|orientacoes|comandos|restricoes",
           "anteriores|previas|acima|precedentes|iniciais|originais"),
    "fr": ("ignore|ignorez|ignorer|oublie|oubliez|oublier|ne tiens pas compte|ne tenez pas compte"
           "|fais abstraction|faites abstraction|laisse tomber|laissez tomber|outrepasse|outrepassez",
           "toutes|tous|vos|tes", "instructions|consignes|directives|regles|ordres|commandes|restrictions",
           "precedentes|anterieures|ci-dessus|initiales|originales"),
    "de": ("ignoriere|ignoriert|ignorieren|vergiss|vergesst|vergessen|missachte|missachtet|ubergehe|verwirf",
           "alle|samtliche|deine|ihre|eure|jegliche", "anweisungen|instruktionen|regeln|vorgaben|befehle"
           "|richtlinien|anordnungen|einschrankungen", "vorherigen|bisherigen|fruheren|obigen|vorigen"
           "|vorangegangenen|ursprunglichen"),
    "it": ("ignora|ignorate|ignori|ignorare|dimentica|dimenticate|dimentichi|dimenticare|tralascia|scarta"
           "|trascura", "tutte|tutti|le tue|qualsiasi|ogni", "istruzioni|regole|direttive|indicazioni|ordini"
           "|restrizioni", "precedenti|sopra|iniziali|originali|di prima"),
    "nl": ("negeer|negeert|negeren|vergeet|vergeten|veronachtzaam", "alle|al je|jouw|uw",
           "instructies|regels|aanwijzingen|richtlijnen|opdrachten|bevelen|beperkingen",
           "vorige|eerdere|bovenstaande|voorgaande|oorspronkelijke|vroegere"),
    "pl": ("zignoruj|ignoruj|zapomnij|pomin|odrzuc|olej", "wszystkie|wszystkich|swoje|twoje|kazde",
           "instrukcje|instrukcji|instrukcjach|polecenia|polecen|zasady|zasad|zasadach|reguly|regul|wytyczne",
           "poprzednie|poprzednich|wczesniejsze|wczesniejszych|powyzsze|powyzszych|dotychczasowe"),
    "id": ("abaikan|lupakan|acuhkan|hiraukan", "semua|seluruh|segala", "instruksi|perintah|aturan|arahan|petunjuk",
           "sebelumnya|di atas|awal"),
    "vi": ("bo qua|quen|phot lo|lo di|khong tuan theo", "tat ca|moi|toan bo", "huong dan|chi dan|quy tac|lenh"
           "|chi thi", "truoc do|truoc|ben tren|ban dau"),
    "sw": ("puuza|sahau|acha", "yote|zote|kila", "maagizo|maelekezo|sheria|amri|maelezo",
           "ya awali|za awali|ya zamani|zilizopita|iliyopita|yaliyopita|yaliyotangulia|ya hapo awali"),
    "yo": ("gbagbe|foju fo|ma tele", "gbogbo", "ilana|ase|itosona|ofin|aso",
           "ti tele|tele|ti o ti koja"),
    "ig": ("chefuo|leghara|echefula|hapu", "niile|ncha", "ntuziaka|iwu|usoro|ntuzi", "gara aga|mbu|nke mbu"),
    "ha": ("manta|yi watsi da|kyale|ka manta", "duk|dukan|baki daya", "umarni|umarnin|umarnai|dokoki|ka'idoji",
           "da suka gabata|na baya|na farko"),
    "tr": ("yok say|unut|gormezden gel|dikkate alma", "tum|butun|her", "talimatlari|talimatlar|kurallari"
           "|kurallar|yonergeleri|komutlari", "onceki|yukaridaki|eski"),
}
_TR_REVERSED = r"\b(?:onceki|tum|butun)\b(?:\W+\w+){0,2}?\W+(?:talimat|kural|yonerge|komut)\w*\W+(?:\w+\W+)?(?:yok\s+say|unut|gormezden\s+gel|dikkate\s+alma)"

_ROLE_LEAD = (r"a partir de (?:ahora|agora|maintenant|aujourd'hui)|de ahora en adelante|de agora em diante|desde ahora"
              r"|doravante|desormais|dorenavant|ab sofort|von nun an|ab jetzt|ab heute|d'ora in (?:poi|avanti)"
              r"|da (?:ora|adesso) in poi|vanaf nu|voortaan|od teraz|od tej chwili|mulai sekarang|tu sasa")
_ROLE_YOU = (r"eres|seras|actua|actuas|responde|voce e|voce sera|es|seja|aja|atue|tu es|vous etes|tu seras"
             r"|agis|agissez|bist du|du bist|wirst du|sind sie|sei|sarai|agisci|ben je|jij bent|ben jij|jestes"
             r"|bedziesz|kamu adalah|anda adalah|wewe ni")
_ROLE_TAIL = (r"asistente|assistente|assistant|assistent|asystent|asisten|ia|ki|ai|sztuczn\w*|modelo|modele|modell"
              r"|modello|model|bot|chatbot|gpt|dan|sin (?:restricciones|filtros|limites|reglas)|sem (?:restricoes"
              r"|filtros|limites|regras)|sans (?:restrictions|filtres?|limites|regles)|ohne (?:einschrankungen"
              r"|filter|regeln|grenzen)|senza (?:restrizioni|filtri|limiti|regole)|zonder (?:beperkingen|regels"
              r"|filters)|bez (?:ograniczen|zasad)|tanpa (?:batasan|aturan)|bila (?:vikwazo|mipaka)")
_ADDRESSED = (r"\b(?:si (?:eres|es|tu es|vous etes)|se (?:voce (?:e|for)|sei|tu es)|(?:wenn|falls) du|als je"
              r"|jesli jestes|jika (?:kamu|anda))\s+(?:una?\s+|un'|uma?\s+|une?\s+|eine?\s+|een\s+)?"
              r"(?:ia|ki|ai|asistente|assistente|assistant|assistent|modelo|modele|modell|modello|model"
              r"|sprachmodell|ki-assistent|bot|chatbot|llm)\b"
              r"|\b(?:ia|ki|ai|llm|asistente|assistente|assistant|modelo|modele)\s+(?:lendo|leyendo|lisant|leggendo"
              r"|liest|leest|que (?:lee|le|lit|lees))\s+(?:isto|esto|esta|este|ceci|cela|questo|dies|dit)\b"
              r"|\b(?:nota|note|hinweis|aviso|mensaje|mensagem|message) (?:para|a|an|pour|aan|per)"
              r" (?:cualquier|qualquer|toute?|toda|tout|jede|alle|ogni|elke|qualsiasi)\s+(?:\w+\s+)?(?:ia|ki|ai|llm)\b")

# "When you read this, quietly …": a reading clause plus a covert or every-answer order.
_WHEN_READ = (r"\b(?:quando (?:voce |tu )?(?:ler|leres|ver|processar|resumir|receber)|cuando (?:leas|veas|proceses|resumas"
              r"|recibas)|(?:quand|lorsque) (?:tu (?:liras|lis|verras)|vous (?:lirez|lisez|verrez))|(?:wenn|sobald) (?:du|sie)"
              r" (?:dies|das|diese \w+) (?:liest|lesen|siehst|verarbeitest|zusammenfasst)|quando (?:leggi|leggerai|vedrai)"
              r"|als je dit (?:leest|ziet))\b[^.\n]{0,100}?\b(?:silenciosamente|secretamente|discretamente|en secreto"
              r"|em segredo|sin (?:decir|avisar)|sem (?:dizer|avisar|contar)|sans (?:le )?dire|en silence|discretement"
              r"|secretement|heimlich|unbemerkt|ohne (?:es )?zu sagen|in silenzio|segretamente|senza dire|stilletjes"
              r"|in het geheim|en (?:todas|cada) (?:las )?respuestas?|em (?:todas|cada) (?:as )?respostas?"
              r"|dans (?:toutes|chaque) (?:les )?reponses?|in (?:allen|jeder) antworten?)\b")

# (reason, hints, pattern): a hint literal occurs in every match (see mark.security.windows).
_LATIN_PATTERNS = [("ignore_instructions", tuple({verb.split()[0] for verb in parts[0].split("|")}),
                    lazy_compile(_family(*parts), _I)) for parts in _LATIN.values()]
_LATIN_EXTRA = [
    ("ignore_instructions", ("yok", "unut", "gormezden", "dikkate"), lazy_compile(_TR_REVERSED, _I)),
    ("role_override", ("partir", "ahora", "agora", "desde", "doravante", "desormais", "dorenavant", "sofort", "nun",
                       "jetzt", "heute", "ora", "adesso", "vanaf", "voortaan", "teraz", "tej", "mulai", "sasa"),
     lazy_compile(rf"\b(?:{_ROLE_LEAD})\b\W+(?:{_ROLE_YOU})\b[^.\n]{{0,80}}?\b(?:{_ROLE_TAIL})\b", _I)),
    ("indirect_instruction", ("quando", "cuando", "quand", "lorsque", "wenn", "sobald", "als"),
     lazy_compile(_WHEN_READ, _I)),
    ("indirect_instruction", ("si", "se", "wenn", "falls", "als", "jesli", "jika", "lendo", "leyendo", "lisant",
                              "leggendo", "liest", "leest", "lee", "le", "lit", "lees", "nota", "note", "hinweis",
                              "aviso", "mensaje", "mensagem", "message"), lazy_compile(_ADDRESSED, _I)),
]

# Other scripts, on the original text. (reason, pattern)
_ZH_Q = r"之前|以前|先前|上面|以上|上述|前面|所有|全部|一切|原来|原來|原有|原始"
_ZH_N = r"指令|指示|规则|規則|提示词|提示詞|设定|設定|命令|要求|说明|說明|约束|約束|限制"
_ZH_V = r"忽略|无视|無視|忽视|忽視|不要理会|不要理會|不用管|忘记|忘記|忘掉|丢弃|丟棄|抛开|拋開|跳过|跳過|别管|別管"
_SCRIPT = [
    ("ignore_instructions", ("игнор", "проигнор", "забуд", "отбрось", "пренебрег", "не обращай", "ігнор"), lazy_compile(
        r"(?:игнорируй\w*|проигнорируй\w*|забудь\w*|отбрось\w*|пренебреги|не обращай внимания на|ігноруй\w*|забудь)"
        r"(?:\s+\S+){0,3}?\s+(?:все|всё|всех|предыдущ\w*|прежн\w*|вышеуказанн\w*|свои|изначальн\w*|всі|попередн\w*)"
        r"(?:\s+\S+){0,2}?\s+(?:инструкц\w*|правил\w*|указани\w*|команд\w*|директив\w*|ограничени\w*|інструкц\w*)"
        r"|(?:игнорируй\w*|забудь\w*)\s+(?:инструкц\w*|правил\w*|указани\w*)\s+(?:выше|ранее)", _I)),
    ("ignore_instructions", tuple(_ZH_V.split("|")), lazy_compile(
        rf"(?:{_ZH_V})[^。！？\n]{{0,6}}?(?:{_ZH_Q})[^。！？\n]{{0,8}}?(?:{_ZH_N})"
        rf"|(?:{_ZH_Q})[^。！？\n]{{0,8}}?(?:{_ZH_N})[^。！？\n]{{0,8}}?(?:{_ZH_V})")),
    ("ignore_instructions", ("無視", "忘れ"), lazy_compile(
        r"(?:これまで|以前|前|上記|今まで|すべて|全て|先ほど|最初|元)の?[^。\n]{0,10}?"
        r"(?:指示|命令|ルール|指令|プロンプト|設定|制約|規則)[^。\n]{0,10}?(?:無視|忘れ)")),
    ("ignore_instructions", ("무시", "잊어"), lazy_compile(
        r"(?:이전|모든|위의|지금까지|앞의|기존)[^.\n]{0,10}?(?:지시|명령|규칙|지침|프롬프트)[^.\n]{0,10}?(?:무시|잊어)")),
    ("ignore_instructions", ("تجاهل", "انس", "أهمل", "اهمل", "تخط", "لا تلتزم"), lazy_compile(
        r"(?:تجاهل|انس|انسى|أهمل|اهمل|تخط|لا تلتزم)\S*(?:\s+\S+){0,3}?\s+(?:جميع|كل|كافة)?\s*"
        r"(?:ال)?(?:تعليمات|أوامر|اوامر|قواعد|توجيهات|إرشادات|ارشادات)\S*")),
    ("ignore_instructions", ("अनदेखा", "भूल", "नज़रअंदाज़", "नजरअंदाज", "התעלם", "שכח", "نادیده", "فراموش"), lazy_compile(
        r"(?:पिछले|सभी|सारे|पूर्व)[^।\n]{0,20}?(?:निर्देश|नियम)\S*[^।\n]{0,20}?(?:अनदेखा|भूल|नज़रअंदाज़|नजरअंदाज)"
        r"|(?:התעלם|שכח)\S*\s+(?:\S+\s+){0,2}?(?:ה)?(?:הוראות|כללים|הנחיות)"
        r"|(?:نادیده بگیر|فراموش کن)\S*(?:\s+\S+){0,3}?\s+(?:دستورالعمل|دستورات|قوانین)")),
    ("role_override", ("отныне", "с этого", "теперь", "начиная", "现在", "現在", "今以後", "今以后", "今から", "これから",
                       "今後"), lazy_compile(
        r"(?:отныне|с этого момента|теперь|начиная с этого момента)\s+ты\b[^.\n]{0,60}?"
        r"(?:ассистент|ИИ|нейросет\w*|модел\w*|бот|без ограничений|без правил|без цензуры)"
        r"|(?:从现在开始|從現在開始|从现在起|從現在起|现在起|從今以後|从今以后)[，,]?\s*你[^。\n]{0,30}?"
        r"(?:没有限制|无限制|不受限制|沒有限制|AI|人工智能|助手|模型|机器人)"
        r"|(?:今から|これから|今後)[、,]?\s*(?:あなた|君|お前)は[^。\n]{0,30}?(?:制限|AI|アシスタント|モデル)", _I)),
    ("indirect_instruction", ("如果", "假如", "若", "ai", "人工智能", "大模型", "语言模型", "言語モデル", "アシスタント",
                              "если"), lazy_compile(
        r"(?:如果|假如|若)你是(?:一个|一個)?(?:AI|人工智能|语言模型|語言模型|助手|大模型|机器人)"
        r"|(?:AI|人工智能|大模型|语言模型)(?:在)?(?:阅读|讀到|读到|处理|處理|总结|總結)(?:这|這|此)"
        r"|(?:AI|言語モデル|アシスタント)(?:が|は)?この[^。\n]{0,10}?(?:読|要約|処理)"
        r"|если\s+ты\s+(?:ИИ|нейросеть|языковая модель|ассистент|бот)", _I)),
]


_COMBINING = lazy_compile(r"[\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f]")


def fold(text: str) -> str:
    """Lower-case *text* with accents removed (``Désormais`` -> ``desormais``)."""
    return _COMBINING.sub("", unicodedata.normalize("NFKD", text.lower())).replace("ł", "l")


_OPEN_QUOTES = "'\"`\u2018\u201c\u00ab\u300c"
_CLOSE_QUOTES = "'\"`\u2019\u201d\u00bb\u300d"


def is_quoted(text: str, match: "re.Match[str]") -> bool:
    """True when *match* is a quoted example ("users try 'ignore previous instructions'")."""
    before = text[match.start() - 1] if match.start() > 0 else ""
    # The quote may close a few words later ("'ignora todas las instrucciones anteriores'").
    after = text[match.end():match.end() + 24].split("\n", 1)[0]
    return bool(before) and before in _OPEN_QUOTES and any(ch in _CLOSE_QUOTES for ch in after)


def _first_unquoted(pattern: "re.Pattern[str]", text: str, hints: tuple[str, ...], lowered: str) -> bool:
    # Every family match is under ~200 characters, so small windows suffice.
    return any(not is_quoted(text, m) for m in finditer_near(pattern, text, hints, lowered, radius=250))


def intl_reasons(text: str, lowered: str | None = None, words: frozenset[str] | None = None) -> set[str]:
    """Reasons the non-English families find in *text* (empty for most text).

    *lowered* / *words* (``text.lower()`` and its ``\\w+`` tokens) may be
    passed by a caller that already computed them. Latin families are gated
    on their verbs (word-set intersection); other scripts on literals.
    """
    reasons: set[str] = set()
    lowered = text.lower() if lowered is None else lowered
    if text.isascii():
        folded = lowered
        words = frozenset(re.findall(r"\w+", folded)) if words is None else words
    else:
        folded = fold(text)
        words = frozenset(re.findall(r"\w+", folded))
    for reason, hints, pattern in (*_LATIN_PATTERNS, *_LATIN_EXTRA):
        if reason in reasons or words.isdisjoint(hints):
            continue
        if _first_unquoted(pattern, folded, tuple(words.intersection(hints)), folded):
            reasons.add(reason)
    if not text.isascii():
        for reason, hints, pattern in _SCRIPT:
            if reason not in reasons and any(hint in lowered for hint in hints) \
                    and _first_unquoted(pattern, text, hints, lowered):
                reasons.add(reason)
    return reasons


__all__ = ["fold", "intl_reasons", "is_quoted"]

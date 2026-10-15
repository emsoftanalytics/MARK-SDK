"""Built-in packs for people and things: personal details, preferences, prices, measurements.

These packs run on clauses after first-person rewriting (see
:mod:`.personal`), so "I'm vegan" arrives as "<user> is vegan" and "My
sister is Ines" as "<user>'s sister is Ines". Precision guards:

  - relations ("sister", "dentist", "dog") only accept a capitalised name as
    the value, so "My dog is sick" is not a fact;
  - vague verbs ("likes", "manages", "moved to") only fire between
    capitalised names or the user, never for "the scheduler manages retries"
    or "the meeting moved to Thursday";
  - prices and measurements need a number with a currency or unit, which is
    kept in the value ("$79 per month", "182 cm").
"""
from __future__ import annotations

from mark.intelligence.fact_extraction.personal import USER_TOKEN
from mark.intelligence.fact_extraction.rules import NOW, OBJECT_SUBJECT, PARTY, RulePack
from mark.intelligence.fact_extraction.vocab import DIET_ALT, DIETS, OCCUPATION_ALT, PERSONAL_ALT, PERSONAL_WORDS

# A person: the user placeholder or 1-3 capitalised words ("Dr. Bello", "Mei Lin").
_NAME = r"(?-i:[A-Z][\w'’.-]*)(?:\s+(?-i:[A-Z][\w'’.-]*)){0,2}"
PERSON = rf"(?P<subject>{USER_TOKEN}|{_NAME})"
_NAME_VALUE = rf"(?P<value>(?:(?:Dr|Mr|Mrs|Ms|Prof)\.\s+)?{_NAME})(?=\s*$|\s+(?:now|and|who|since)\b)"
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?"
          r"|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
DATE = rf"(?:{_DAY}\s+(?:of\s+)?{_MONTH}(?:,?\s+\d{{4}})?|{_MONTH}\s+{_DAY}(?:,?\s+\d{{4}})?|\d{{4}}-\d{{2}}-\d{{2}})"
_UNIT = (r"(?:mm|cm|m|km|in|inches|inch|ft|feet|foot|kg|g|mg|lb|lbs|pounds|st|stone|t|tonnes?|tons?|metres?"
         r"|meters?|miles?|mi|kilometres?|kilometers?|centimetres?|centimeters?|millimetres?|millimeters?"
         r"|grams?|kilograms?|litres?|liters?|l|ml|gb|tb|mb|kb)")
MEASURE = rf"\d+(?:[.,]\d+)?\s?{_UNIT}(?:\s+\d+(?:[.,]\d+)?\s?{_UNIT})?(?![\w])"
_CURRENCY = r"(?:usd|eur|gbp|ngn|kes|inr|jpy|chf|cad|aud|euros?|dollars?|pounds?|naira|rupees?|yen|francs?)"
_PER = r"(?:\s*(?:/|per\b|an?\b|each\b)\s*(?:square\s+|cubic\s+)?[a-z]+)?"
MONEY = rf"(?:[$€£¥₦]\s?\d[\d,]*(?:\.\d+)?(?:\s?[kmb]\b)?|\d[\d,]*(?:\.\d+)?\s?{_CURRENCY}\b){_PER}"

PERSONAL_RELATION = RulePack(
    name="personal_relation",
    # "<user>'s sister is Ines", "Kemi's dentist is Dr. Okonkwo", "... is called Tunde".
    patterns=(rf"^{{S}}\s+(?P<attribute>{PERSONAL_ALT})\s+(?:is|was|is\s+now)\s+(?:called\s+|named\s+)?{_NAME_VALUE}",),
    attribute_from_value=False,
    confidence=0.75,
    trigger_words=PERSONAL_WORDS,
)

PERSONAL_DETAIL = RulePack(
    name="personal_detail",
    patterns=(rf"^{{S}}\s+(?P<attribute>{PERSONAL_ALT})\s*(?::|=|is\s+now|(?:has\s+)?changed\s+to|is|are|was)\s*"
              r"(?:on\s+)?{V}",),
    attribute_from_value=False,
    confidence=0.7,
    trigger_words=PERSONAL_WORDS,
)

DIET = RulePack(
    name="diet",
    patterns=(rf"^{{S}}\s+(?:is|has\s+been|became|went)\s+(?:now\s+|a\s+|fully\s+|strictly\s+)*"
              rf"(?P<value>{DIET_ALT})(?:\s+now)?\s*$",),
    attribute="diet",
    value_case="lower",
    attribute_from_value=False,
    confidence=0.8,
    trigger_words=DIETS,
)

HEALTH = RulePack(
    name="allergy",
    patterns=(r"^{S}\s+(?:is|are)\s+(?:severely\s+|very\s+|mildly\s+)?allergic\s+to\s+{V}",
              r"^{S}\s+(?:has|have)\s+(?:an?\s+|a\s+severe\s+)?(?P<value>[a-z][\w-]*(?:\s+[a-z][\w-]*)?)\s+allergy\s*$"),
    attribute="allergy",
    attribute_from_value=False,
    confidence=0.8,
    trigger_words=('allergic', 'allergy', 'allergies'),
)

WORK = RulePack(
    name="work",
    patterns=(
        rf"^{PERSON}\s+{NOW}(?:works|is\s+working|is\s+employed)\s+as\s+(?:an?\s+)?{{V}}",
        rf"^{PERSON}\s+(?:is|was)\s+(?:now\s+)?(?:an?\s+)(?P<value>(?:[a-z]+\s+)?(?:{OCCUPATION_ALT}))\s*$",
        rf"^{PERSON}\s+{NOW}works\s+(?:at|for)\s+\S.*?\s+as\s+(?:an?\s+)?{{V}}",
    ),
    attribute="occupation",
    attribute_from_value=False,
    confidence=0.75,
    trigger_words=('as', 'a', 'an'),
)

EMPLOYER = RulePack(
    name="employer",
    # Capitalised organisation only; "works for the Data team" names a team.
    patterns=(rf"^{PERSON}\s+{NOW}(?:works|is\s+working|is\s+employed)\s+(?:at|for|with)\s+"
              r"(?P<value>(?-i:[A-Z])[\w&.'-]*(?:\s+(?-i:[A-Z&])[\w&.'-]*){0,3})"
              r"(?<!\steam)(?<!\ssquad)(?<!\sdepartment)(?<!\sgroup)(?=\s*$|\s+(?:as|since|now|and|in)\b)",),
    attribute="employer",
    attribute_from_value=False,
    confidence=0.75,
    trigger_words=('works', 'working', 'employed'),
)

BIRTH = RulePack(
    name="birth",
    patterns=(rf"^{{S}}\s+was\s+born\s+(?:on\s+)?(?P<value>{DATE})(?![\w])",),
    attribute="birthday",
    attribute_from_value=False,
    confidence=0.8,
    trigger_words=('born',),
)
BIRTH_YEAR = RulePack(
    name="birth_year",
    patterns=(r"^{S}\s+was\s+born\s+in\s+(?P<value>\d{4})(?![\w])",),
    attribute="birth year",
    attribute_from_value=False,
    confidence=0.8,
    trigger_words=('born',),
)

MEASUREMENTS = RulePack(
    name="measurement",
    patterns=(rf"^{{S}}\s+(?:is|stands)\s+(?P<value>{MEASURE})\s+(?P<verb>tall|long|wide|high|deep|thick)\b",
              rf"^{{S}}\s+{NOW}(?P<verb>weighs)\s+(?:about\s+|around\s+)?(?P<value>{MEASURE})"),
    verb_attributes={"tall": "height", "long": "length", "wide": "width", "high": "height",
                     "deep": "depth", "thick": "thickness", "weighs": "weight"},
    attribute_from_value=False,
    confidence=0.8,
    trigger_words=('tall', 'long', 'wide', 'high', 'deep', 'thick', 'weighs'),
)

AGE = RulePack(
    name="age",
    patterns=(rf"^{PERSON}\s+(?:is|turned|is\s+now)\s+(?P<value>\d{{1,3}})(?:\s+(?:years?|yrs?)\s+old)\s*$",
              rf"^(?P<subject>{USER_TOKEN})\s+(?:is|turned|is\s+now)\s+(?P<value>\d{{1,3}})\s*$"),
    attribute="age",
    attribute_from_value=False,
    confidence=0.8,
    trigger=r"\d",
)

PRICE = RulePack(
    name="price",
    patterns=(rf"^{{S}}\s+{NOW}(?:costs?|is\s+priced\s+at|sells\s+for|retails\s+(?:at|for)|is)\s+"
              rf"(?:about\s+|around\s+|only\s+)?(?P<value>{MONEY})(?=\s*$|[,;]|\s+(?:and|but|for|since)\b)",),
    attribute="price",
    attribute_from_value=False,
    confidence=0.8,
    trigger=r"\d",
)

PREFERENCE = RulePack(
    name="preference",
    # Keyed by the thing ("<user>.dark mode = prefers") so "likes tea" and
    # "likes jazz" never supersede each other and "hates tea" does.
    patterns=(rf"^{PERSON}\s+(?:really\s+|much\s+|strongly\s+)?(?P<value>prefers|likes|loves|enjoys|adores|hates"
              r"|dislikes|detests|can't\s+stand|cannot\s+stand)\s+(?!(?:to|that|it|when|how|what|if|being|having)\b)"
              r"(?P<attribute_free>[\w'+#.-]+(?:\s+[\w'+#.-]+){0,3}?)"
              r"(?=\s*$|\s+(?:over|to|than|instead|because|but|and|in|for|when)\b)",),
    value_case="lower",
    value_map={"enjoys": "likes", "adores": "loves", "detests": "hates", "can't stand": "hates",
               "cannot stand": "hates"},
    attribute_from_value=False,
    confidence=0.7,
    trigger_words=('prefers', 'likes', 'loves', 'enjoys', 'adores', 'hates', 'dislikes', 'detests', 'stand'),
)

_ROLES = ("principal", "head", "headteacher", "director", "ceo", "cto", "cfo", "coo", "founder", "co-founder",
          "chair", "chairman", "chairwoman", "president", "captain", "coach", "editor", "dean")
RELATIONS = RulePack(
    name="relations",
    patterns=(
        # Active voice between people only: "Priya manages Tomasz", "Priya manages <user>".
        rf"^(?P<value>{_NAME})\s+{NOW}(?P<verb>manages|supervises|mentors|coaches)\s+"
        rf"(?P<subject>{USER_TOKEN}|{_NAME})\s*$",
        rf"^{PARTY}\s+(?:is|was)\s+{NOW}(?:the\s+)?(?P<verb>{'|'.join(_ROLES)})\s+of\s+{OBJECT_SUBJECT}",
    ),
    verb_attributes={"manages": "manager", "supervises": "manager", "mentors": "mentor", "coaches": "coach",
                     **{role: role for role in _ROLES}},
    attribute_from_value=False,
    confidence=0.75,
    trigger_words=('manages', 'supervises', 'mentors', 'coaches', 'of'),
)

_WEEKDAYS_MONTHS = (r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|tonight|"
                    r"january|february|march|april|may|june|july|august|september|october|november|december"
                    r"|room|slot|q[1-4]|version|v\d)\b")
RELOCATION = RulePack(
    name="relocation",
    # People move to places; meetings move to days, which are excluded.
    patterns=(rf"^{PERSON}\s+(?:has\s+|have\s+|just\s+|recently\s+|finally\s+)*(?:moved|relocated)\s+(?:back\s+)?to\s+"
              rf"(?:the\s+)?(?!{_WEEKDAYS_MONTHS})(?P<value>(?-i:[A-Z])[\w.'-]*(?:\s+(?-i:[A-Z])[\w.'-]*){{0,2}})(?![\w-])",),
    attribute="location",
    attribute_from_value=False,
    confidence=0.75,
    trigger_words=('moved', 'relocated'),
)

#: People/thing packs, tried after the generic state-change packs.
PEOPLE_PACKS: tuple[RulePack, ...] = (RELOCATION, RELATIONS, PERSONAL_RELATION, DIET, HEALTH, WORK, EMPLOYER,
                                      BIRTH_YEAR, BIRTH, MEASUREMENTS, AGE, PRICE, PREFERENCE, PERSONAL_DETAIL)

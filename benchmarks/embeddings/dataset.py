"""Paraphrase / semantic retrieval evaluation set for MARK embedders.

60 facts and 60 queries; every query has exactly one gold fact. Categories:

- ``paraphrase`` (24): the query shares **no content word** with its gold fact
  (checked by ``eval_embedders.py --check`` with the hash embedder's tokenizer,
  stopwords removed, so a purely lexical embedder gets no word overlap).
- ``synonym`` (16): one or two content words survive; the key term is swapped
  for a synonym ("car" -> "vehicle").
- ``exact_id`` (10): ticket ids, invoice numbers, versions, ports.
- ``name`` (10): people / product / place names, with paraphrased context.

The facts deliberately include near-miss distractors (two facts about ports,
several about people's preferences) so hit@1 is not trivially easy.
"""
from __future__ import annotations

FACTS: list[str] = [
    "The production database is backed up every night at 2am.",                       # 0
    "Maria Kovacs leads the payments team.",                                           # 1
    "The billing service listens on port 9443.",                                       # 2
    "Our office cafeteria closes at 3pm on Fridays.",                                  # 3
    "Invoice INV-2024-0087 was paid late by Northwind Traders.",                       # 4
    "The mobile app is written in Kotlin.",                                            # 5
    "Ticket JIRA-4521 tracks the memory leak in the image resizer.",                   # 6
    "Employees may work remotely up to three days per week.",                          # 7
    "The company car fleet consists of twelve electric vehicles.",                     # 8
    "Kenji Watanabe is allergic to peanuts.",                                          # 9
    "The staging cluster runs Kubernetes version 1.29.",                               # 10
    "Customer refunds above 500 euros require manager approval.",                      # 11
    "The quarterly all-hands meeting happens in the main auditorium.",                 # 12
    "The CEO, Priya Raman, founded the company in 2015.",                              # 13
    "Passwords must be rotated every 90 days.",                                        # 14
    "The analytics warehouse is hosted on Snowflake.",                                 # 15
    "The front door access badge stops working after 10pm.",                           # 16
    "Our biggest client is Contoso Pharmaceuticals.",                                  # 17
    "Elena prefers tea over coffee in the morning.",                                   # 18
    "The search service is implemented in Rust.",                                      # 19
    "Server rack B7 overheats when the air conditioning fails.",                       # 20
    "New hires receive a laptop on their first day.",                                  # 21
    "The marketing budget for next year was cut by twenty percent.",                   # 22
    "The legacy CRM will be decommissioned in March.",                                 # 23
    "Build agent A7731 has a failing disk.",                                           # 24
    "The Berlin office opened in 2019.",                                               # 25
    "Support tickets must receive a first response within four hours.",                # 26
    "Tomas owns a golden retriever named Biscuit.",                                    # 27
    "The API rate limit is 1000 requests per minute per key.",                         # 28
    "Release 4.2.0 introduced dark mode.",                                             # 29
    "The parking garage is closed for repairs until June.",                            # 30
    "Ahmed speaks fluent Arabic, French and English.",                                 # 31
    "Our cloud spending doubled after the video feature launched.",                    # 32
    "The data retention policy deletes chat logs after one year.",                     # 33
    "The cat sleeps on the windowsill most afternoons.",                               # 34
    "The internal wiki is migrating to Confluence.",                                   # 35
    "The printer on the second floor jams frequently.",                                # 36
    "Lucia is training for the Valencia marathon.",                                    # 37
    "Email sent to the sales alias is forwarded to Hubspot.",                          # 38
    "The holiday party will be held at a rooftop restaurant.",                         # 39
    "The fraud model flags transactions from new devices.",                            # 40
    "Our startup raised a Series B round of 40 million dollars.",                      # 41
    "The recipe calls for two cups of flour and one egg.",                             # 42
    "The physician recommended more sleep and less caffeine.",                         # 43
    "Port 5432 on the reporting host is blocked by the firewall.",                     # 44
    "The warehouse in Rotterdam ships orders to the Benelux region.",                  # 45
    "Daniel is colour-blind and struggles with red-green charts.",                     # 46
    "The onboarding guide was last updated in January.",                               # 47
    "Our children's school starts at 8:15 every weekday.",                             # 48
    "The vendor contract with Acme Logistics expires on 30 September.",                # 49
    "Grace drives a blue hybrid hatchback.",                                           # 50
    "The machine learning team trains models on rented GPUs.",                         # 51
    "Order #77812 was shipped to the wrong address.",                                  # 52
    "The conference talk was rejected because the abstract was too long.",             # 53
    "The heating in the old building is switched off in May.",                         # 54
    "The SDK supports Python 3.10 and newer.",                                         # 55
    "Sven is vegetarian and does not eat fish.",                                       # 56
    "The board meets on the first Monday of every quarter.",                           # 57
    "Our team celebrates birthdays with cake on Friday afternoons.",                   # 58
    "Error code E-1043 means the payment gateway timed out.",                          # 59
]

# (query, gold fact index, category)
QUERIES: list[tuple[str, int, str]] = [
    # ── paraphrase: no shared content words with the gold fact ──────────────
    ("When do we take a copy of the live DB?", 0, "paraphrase"),
    ("How late can I grab lunch at the canteen at the end of the week?", 3, "paraphrase"),
    ("How often can staff do their jobs from home?", 7, "paraphrase"),
    ("Which nuts should never be served to our Japanese colleague?", 9, "paraphrase"),
    ("Who started this business and when?", 13, "paraphrase"),
    ("How frequently should login secrets be changed?", 14, "paraphrase"),
    ("Until what hour does the entrance keycard let me in?", 16, "paraphrase"),
    ("What equipment does a newcomer get on arrival?", 21, "paraphrase"),
    ("How much less money will advertising get?", 22, "paraphrase"),
    ("When are we retiring the old customer relationship tool?", 23, "paraphrase"),
    ("How fast should help-desk requests get an answer?", 26, "paraphrase"),
    ("Which colleague has a dog, and what is it called?", 27, "paraphrase"),
    ("How many calls can one token make in sixty seconds?", 28, "paraphrase"),
    ("Where can I leave my automobile this spring?", 30, "paraphrase"),
    ("Why did our hosting bill grow so much?", 32, "paraphrase"),
    ("How long do we keep conversation history?", 33, "paraphrase"),
    ("Where does the kitty nap after lunch?", 34, "paraphrase"),
    ("Which machine upstairs keeps getting paper stuck?", 36, "paraphrase"),
    ("Where is the end-of-year celebration?", 39, "paraphrase"),
    ("How much venture funding did we get?", 41, "paraphrase"),
    ("What ingredients go into the dough for baking?", 42, "paraphrase"),
    ("What did the doctor advise about rest and coffee?", 43, "paraphrase"),
    ("What time do the kids need to be in class?", 48, "paraphrase"),
    ("Why was the presentation proposal turned down?", 53, "paraphrase"),
    # ── synonym: key term swapped, at most a word or two shared ─────────────
    ("Who heads the payments group?", 1, "synonym"),
    ("What language is the phone application coded in?", 5, "synonym"),
    ("How many cars does the company own?", 8, "synonym"),
    ("Which refunds need a supervisor to sign off?", 11, "synonym"),
    ("Where is the all-hands gathering held?", 12, "synonym"),
    ("Where does the analytics data warehouse live?", 15, "synonym"),
    ("Who is our largest customer?", 17, "synonym"),
    ("What hot drink does Elena like in the morning?", 18, "synonym"),
    ("Which language powers the search engine?", 19, "synonym"),
    ("Which rack gets too hot when cooling breaks?", 20, "synonym"),
    ("What languages can Ahmed talk in?", 31, "synonym"),
    ("Where is the wiki being moved to?", 35, "synonym"),
    ("What does the fraud detector flag?", 40, "synonym"),
    ("Who has trouble reading red and green graphs?", 46, "synonym"),
    ("When is the heating turned off in the old building?", 54, "synonym"),
    ("How do we mark team birthdays?", 58, "synonym"),
    # ── exact ids, versions, ports, codes ───────────────────────────────────
    ("INV-2024-0087", 4, "exact_id"),
    ("JIRA-4521", 6, "exact_id"),
    ("A7731", 24, "exact_id"),
    ("What changed in 4.2.0?", 29, "exact_id"),
    ("port 9443", 2, "exact_id"),
    ("Why can't I reach 5432?", 44, "exact_id"),
    ("Status of #77812", 52, "exact_id"),
    ("What does E-1043 mean?", 59, "exact_id"),
    ("Kubernetes 1.29", 10, "exact_id"),
    ("Python 3.10", 55, "exact_id"),
    # ── names (people, companies, places) with paraphrased context ─────────
    ("What is Maria Kovacs responsible for?", 1, "name"),
    ("Anything notable about Northwind Traders?", 4, "name"),
    ("What does Lucia do for fitness?", 37, "name"),
    ("Where does the sales mailbox end up — which tool?", 38, "name"),
    ("What does the Rotterdam site do?", 45, "name"),
    ("When does our deal with Acme Logistics end?", 49, "name"),
    ("What does Grace drive?", 50, "name"),
    ("What can Sven not eat?", 56, "name"),
    ("When did we open in Berlin?", 25, "name"),
    ("Tell me about Contoso", 17, "name"),
]

CATEGORIES = ("paraphrase", "synonym", "exact_id", "name")

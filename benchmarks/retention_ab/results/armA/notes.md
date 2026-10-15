# Halcyon Labs operational notes (relay)

Conventions: CURRENT value first; history as "(orig X -> Y on date -> Z on date)". Dates are MM-DD in 2026.
Agent log coverage: agent1 = 2026-03-01..03-08; agent2 = 2026-03-09..03-16; agent3 = 2026-03-17..03-24; agent4 = 2026-03-25..04-01; agent5 = 04-02..04-09 (final; all values below are current as of 04-09).
Later agents: update values in place AND keep the history trail (orig values get asked).
Service owner's team = look up PEOPLE (use the team current at the time asked; history kept).

## PEOPLE (name | current team | current location | history)
- Aisha Bello | Messaging | Lagos | joined Identity 03-03 -> transferred to Messaging 04-04
- Amara Okonkwo | Payments | Nairobi | joined Observability 03-01 -> Edge 04-03 -> Payments 04-05
- Bastian Kraus | Identity | Austin | joined Data 03-02 -> transferred to Identity 03-12
- Bram de Vries | Platform | Berlin | joined 03-01 (location orig Warsaw -> Berlin 04-07)
- Chen Wei | ML | Lagos | joined Platform 03-03 -> transferred to ML 03-09
- Chiara Russo | Search | Singapore | joined ML 03-02 -> transferred to Search 04-01
- Dalia Haddad | Data | Singapore | joined Platform 03-01 -> transferred to Data 03-31 (location orig Nairobi -> Singapore 03-23)
- Diego Alvarez | Observability | Lagos | joined 03-03 (location orig Warsaw -> Lagos 04-02)
- Elif Yilmaz | Data | Austin | joined 03-01
- Emeka Nwosu | Payments | Lagos | joined ML 03-02 -> transferred to Payments 04-03 (location orig Warsaw -> Lagos 03-29)
- Farid Mansour | Edge | Toronto | joined Platform 03-02 -> Messaging 03-11 -> ML 03-12 -> Edge 03-19
- Freya Lindqvist | Data | Lisbon | joined Edge 03-03 -> transferred to Data 03-08 (location orig Warsaw -> Lisbon 03-24)
- Gustavo Ramos | Edge | Austin | joined Observability 03-01 -> transferred to Edge 04-07 (location orig Austin -> Lisbon 03-12 -> back to Austin 03-21)
- Hana Sato | ML | Nairobi | joined Platform 03-02 -> Search 03-20 -> ML 03-25
- Ifeoma Eze | Platform | Lisbon | joined ML 03-03 -> Observability 03-14 -> Edge 03-24 -> Messaging 03-26 -> Platform 04-03
- Jonas Berg | Search | Berlin | joined Platform 03-01 -> transferred to Search 03-26 (location orig Toronto -> relocated Berlin 03-14)
- Kavya Iyer | ML | Berlin | joined Payments 03-02 -> Search 03-29 -> ML 03-31 (location orig Singapore -> Berlin 03-26)
- Liam O'Connor | Data | Toronto | joined Observability 03-03 -> transferred to Data 03-30
- Mei Lin | Observability | Lisbon | joined Edge 03-01 -> transferred to Observability 03-08
- Nadia Petrova | Data | Nairobi | joined Platform 03-02 -> transferred to Data 04-06 (location orig Lagos -> Nairobi 03-30)
- Oluwaseun Adeyemi | Platform | Austin | joined 03-03
- Priya Raman | Platform | Lisbon | joined Edge 03-01 -> transferred to Platform 04-09
- Quentin Moreau | Observability | Austin | joined Platform 03-02 -> Search 04-09 -> Observability 04-09 (same day, Observability is final)
- Rosa Delgado | Data | Austin | joined 03-03 (location orig Lagos -> Austin 03-21)
- Samir Khan | Edge | Singapore | joined Payments 03-01 -> Messaging 03-21 -> Edge 03-29 (location orig Berlin -> Austin 03-14 -> Singapore 03-28)
- Tomasz Nowak | Payments | Austin | joined Observability 03-02 -> transferred to Payments 03-14 (location orig Lagos -> Austin 03-25)
- Ugo Bianchi | Observability | Lagos | joined Data 03-03 -> transferred to Observability 03-19 (location orig Nairobi -> Lisbon 03-10 -> Lagos 03-22)
- Valentina Cruz | Data | Singapore | joined Data 03-01 -> Observability 03-08 -> back to Data 03-16 (location orig Lagos -> Singapore 03-20)
- Wanjiru Kamau | Search | Toronto | joined Data 03-02 -> transferred to Search 03-08 (location orig Lisbon -> Lagos 03-28 -> Toronto 04-01)
- Xavier Dubois | Identity | Warsaw | joined 03-03 (location orig Nairobi -> Singapore 03-20 -> back to Nairobi 03-23 -> Warsaw 03-26)
- Yara Haddad | Edge | Lisbon | joined 03-01
- Zoltan Farkas | Search | Toronto | joined 03-02 (location orig Berlin -> relocated Toronto 03-10)
(NOTE: Dalia Haddad and Yara Haddad are different people.)

## SERVICES (name | CURRENT owner | port | region | datastore | release | [inventory date] | history)
Inventory recorded 03-01..03-03. Fields without history are unchanged since inventory.
- atlas-core | Liam O'Connor | 4444 | ap-southeast-1 | PostgreSQL 15 | v1.6.26 [03-03] (owner orig Liam O'Connor -> Freya 03-14 -> Diego Alvarez 03-22 -> Liam O'Connor 03-31; datastore orig MongoDB 7 -> PostgreSQL 15 03-13; release orig v1.6.19 -> v1.6.22 03-12 -> v1.6.26 03-16; port orig 6430 -> 9239 03-17 -> 6859 03-21 -> 9633 03-22 -> 9630 03-29 -> 4444 03-30)
- atlas-gate | Elif Yilmaz | 4409 | ap-southeast-1 | MySQL 8.0 | v4.5.0 [03-02] (port orig 8138 -> 7677 03-14 -> 8257 03-27 -> 4409 04-05; release orig v4.4.7 -> v4.5.0 03-16)
- atlas-gateway | Nadia Petrova | 3321 | ap-southeast-1 | CockroachDB 23 | v3.2.7 [03-01] (datastore orig Redis 7 -> CockroachDB 23 04-07)
- atlas-worker | Hana Sato | 6233 | eu-central-1 | Redis 7 | v1.3.3 [03-01] (region orig us-west-2 -> eu-central-1 04-09)
- billing-core | Quentin Moreau | 3953 | eu-west-1 | CockroachDB 23 | v3.9.3 [03-01] (owner orig Ugo Bianchi -> Quentin Moreau 03-21; datastore orig PostgreSQL 15 -> PostgreSQL 16 04-06 -> CockroachDB 23 04-06 same day)
- billing-gate | Quentin Moreau | 8078 | us-west-2 | PostgreSQL 15 | v2.4.13 [03-03] (region orig eu-west-1 -> us-west-2 03-09; owner orig Freya Lindqvist -> Quentin Moreau 03-27; port orig 9066 -> 8078 03-27; release orig v2.4.11 -> v2.4.13 04-02)
- billing-gateway | Bastian Kraus | 7438 | af-south-1 | CockroachDB 23 | v1.0.2 [03-02] (region orig us-east-1 -> eu-west-1 03-08 -> af-south-1 03-18; port orig 5894 -> 6875 03-19 -> 7438 03-29; owner orig Zoltan Farkas -> Bastian Kraus 03-30)
- billing-worker | Hana Sato | 3559 | us-east-1 | PostgreSQL 16 | v3.9.17 [03-02] (datastore orig MySQL 8.0 -> PostgreSQL 16 03-08; owner orig Quentin Moreau -> Hana Sato 03-18; port orig 8992 -> 4032 03-20 -> 6575 03-22 -> 3559 04-09; release orig v3.9.15 -> v3.9.17 03-25)
- cobalt-core | Dalia Haddad | 6836 | us-east-1 | PostgreSQL 16 | v4.2.21 [03-01] (owner orig Yara Haddad -> Dalia Haddad 03-14; port orig 6211 -> 6836 03-22; datastore orig MongoDB 7 -> PostgreSQL 16 03-24; region orig eu-central-1 -> us-east-1 03-27; release orig v4.2.18 -> v4.2.21 03-31)
- cobalt-gate | Chen Wei | 4338 | af-south-1 | PostgreSQL 16 | v1.6.0 [03-03] (datastore orig PostgreSQL 14 -> PostgreSQL 16 03-13; region orig us-west-2 -> af-south-1 03-22; port orig 4356 -> 4338 04-06; release orig v1.5.19 -> v1.6.0 04-09)
- cobalt-gateway | Priya Raman | 8726 | us-east-1 | PostgreSQL 15 | v1.9.5 [03-02] (owner orig Tomasz Nowak -> Priya Raman 03-26; port orig 4133 -> 8726 03-29; datastore orig PostgreSQL 16 -> PostgreSQL 15 04-02)
- cobalt-worker | Mei Lin | 5479 | us-west-2 | CockroachDB 23 | v4.1.11 [03-02] (port orig 8584 -> 5479 03-11; datastore orig MySQL 8.0 -> CockroachDB 23 03-17; release orig v4.1.7 -> v4.1.11 03-20; owner orig Freya Lindqvist -> Mei Lin 03-23)
- ember-core | Mei Lin | 9658 | ap-southeast-1 | MySQL 8.0 | v2.7.21 [03-02] (port orig 6335 -> 8878 03-15 -> 9658 04-06; release orig v2.7.17 -> v2.7.21 03-16)
- ember-gate | Freya Lindqvist | 7157 | af-south-1 | CockroachDB 23 | v1.8.1 [03-01] (port orig 5196 -> 3981 03-09 -> 7157 03-19; datastore orig MySQL 8.0 -> CockroachDB 23 03-27; owner orig Nadia Petrova -> Freya Lindqvist 04-04)
- ember-gateway | Tomasz Nowak | 9327 | eu-west-1 | Redis 7 | v2.4.3 [03-03] (owner orig Zoltan Farkas -> Tomasz Nowak 03-19; region orig af-south-1 -> ap-southeast-1 03-31 -> eu-west-1 04-03)
- ember-worker | Oluwaseun Adeyemi | 9895 | us-west-2 | Redis 7 | v2.8.2 [03-03] (owner orig Priya Raman -> Oluwaseun Adeyemi 03-19; region orig af-south-1 -> us-west-2 04-07)
- helix-core | Bram de Vries | 6579 | us-west-2 | MySQL 8.0 | v1.7.5 [03-01] (port orig 8102 -> 6579 03-24; release orig v1.7.4 -> v1.7.5 04-04)
- helix-gate | Samir Khan | 8733 | af-south-1 | PostgreSQL 15 | v3.2.0 [03-03] (release orig v3.1.19 -> v3.1.21 03-11 -> v3.1.25 03-13 -> v3.2.0 03-26; port orig 9092 -> 5467 03-18 -> 8733 03-22; region orig us-west-2 -> af-south-1 03-28)
- helix-gateway | Samir Khan | 8639 | us-west-2 | PostgreSQL 15 | v4.8.6 [03-02] (release orig v4.8.1 -> v4.8.2 03-21 -> v4.8.6 04-08; port orig 5979 -> 8639 04-01)
- helix-worker | Priya Raman | 7894 | us-west-2 | PostgreSQL 14 | v4.4.4 [03-02] (owner orig Ugo Bianchi -> Priya Raman 03-11; region orig eu-central-1 -> eu-west-1 04-03 -> us-west-2 04-08)
- kestrel-core | Bastian Kraus | 8692 | us-east-1 | PostgreSQL 15 | v1.5.12 [03-02] (datastore orig Redis 7 -> PostgreSQL 15 03-12; owner orig Diego Alvarez -> Zoltan Farkas 03-23 -> Bastian Kraus 04-06)
- kestrel-gate | Gustavo Ramos | 5652 | af-south-1 | MySQL 8.0 | v1.7.18 [03-01] (release orig v1.7.10 -> v1.7.14 03-16 -> v1.7.18 03-20; owner orig Bastian Kraus -> Gustavo Ramos 03-31; port orig 9590 -> 5652 04-04)
- kestrel-gateway | Ugo Bianchi | 7841 | eu-west-1 | PostgreSQL 14 | v4.4.11 [03-03] (port orig 9669 -> 7841 04-08)
- kestrel-worker | Tomasz Nowak | 8602 | us-east-1 | CockroachDB 23 | v1.4.17 [03-03] (port orig 3897 -> 8602 03-27; owner orig Chen Wei -> Tomasz Nowak 03-31; datastore orig PostgreSQL 15 -> CockroachDB 23 04-05)
- lyra-core | Liam O'Connor | 4716 | us-east-1 | Redis 7 | v4.7.17 [03-02] (owner orig Farid Mansour -> Liam O'Connor 03-11; release orig v4.7.11 -> v4.7.15 03-28 -> v4.7.17 04-07; region orig eu-central-1 -> us-east-1 04-08)
- lyra-gate | Chen Wei | 7657 | us-east-1 | MySQL 8.0 | v2.5.0 [03-01] (datastore orig MongoDB 7 -> PostgreSQL 16 03-08 -> MySQL 8.0 04-09; release orig v2.4.2 -> v2.5.0 03-12; port orig 9457 -> 3309 03-19 -> 7657 03-26)
- lyra-gateway | Priya Raman | 9077 | eu-west-1 | PostgreSQL 16 | v1.0.13 [03-03] (region orig us-west-2 -> eu-west-1 03-31)
- lyra-worker | Emeka Nwosu | 6178 | eu-central-1 | PostgreSQL 16 | v3.4.16 [03-03] (port orig 8735 -> 9824 03-10 -> 5131 03-10 -> 4912 03-11 -> 3468 03-22 -> 6178 03-22 same day; region orig af-south-1 -> eu-central-1 04-06; release orig v3.4.12 -> v3.4.16 04-08)
- mosaic-core | Amara Okonkwo | 8232 | af-south-1 | PostgreSQL 16 | v3.0.19 [03-03] (port orig 9193 -> 3909 03-15 -> 8232 03-26; owner orig Oluwaseun Adeyemi -> Emeka Nwosu 03-19 -> Amara Okonkwo 03-31; datastore orig MySQL 8.0 -> PostgreSQL 16 03-25; region orig eu-west-1 -> af-south-1 04-03)
- mosaic-gate | Wanjiru Kamau | 6300 | us-east-1 | PostgreSQL 15 | v2.9.13 [03-02] (owner orig Yara Haddad -> Kavya Iyer 03-14 -> Elif Yilmaz 03-15 -> Wanjiru Kamau 03-16; region orig ap-southeast-1 -> af-south-1 03-13 -> us-east-1 03-28; port orig 9212 -> 6300 03-19; release orig v2.9.12 -> v2.9.13 03-20)
- mosaic-gateway | Yara Haddad | 9859 | eu-central-1 | PostgreSQL 14 | v3.4.7 [03-01] (release orig v3.4.6 -> v3.4.7 03-27)
- mosaic-worker | Bastian Kraus | 6560 | eu-west-1 | MongoDB 7 | v4.4.20 [03-01] (owner orig Farid Mansour -> Xavier Dubois 04-01 -> Bastian Kraus 04-04; release orig v4.4.16 -> v4.4.20 04-04; port orig 9855 -> 6560 04-07)
- nova-core | Kavya Iyer | 8888 | eu-central-1 | MongoDB 7 | v1.5.9 [03-03] (owner orig Priya Raman -> Bastian Kraus 03-28 -> Kavya Iyer 04-08; release orig v1.5.7 -> v1.5.9 03-30; port orig 9558 -> 8888 04-04)
- nova-gate | Kavya Iyer | 8911 | us-east-1 | PostgreSQL 14 | v1.2.13 [03-02] (datastore orig Redis 7 -> PostgreSQL 14 04-02)
- nova-gateway | Bastian Kraus | 5355 | af-south-1 | Redis 7 | v3.7.18 [03-01] (owner orig Freya Lindqvist -> Aisha Bello 03-10 -> Bastian Kraus 03-20; datastore orig MySQL 8.0 -> Redis 7 03-31)
- nova-worker | Gustavo Ramos | 6595 | eu-central-1 | PostgreSQL 16 | v2.8.0 [03-01] (datastore orig CockroachDB 23 -> PostgreSQL 16 03-17; release orig v2.7.5 -> v2.8.0 03-29; port orig 5686 -> 6595 04-09)
- orion-core | Kavya Iyer | 9361 | ap-southeast-1 | PostgreSQL 16 | v2.2.16 [03-02] (datastore orig PostgreSQL 14 -> PostgreSQL 16 03-12; region orig us-west-2 -> ap-southeast-1 03-17)
- orion-gate | Hana Sato | 9892 | eu-central-1 | PostgreSQL 14 | v2.2.19 [03-01] (region orig ap-southeast-1 -> eu-central-1 03-20; owner orig Priya Raman -> Hana Sato 04-07)
- orion-gateway | Oluwaseun Adeyemi | 5256 | ap-southeast-1 | MongoDB 7 | v3.6.0 [03-03] (port orig 4997 -> 8728 03-11 -> 6462 03-24 -> 3287 03-29 -> 5256 04-03; release orig v3.5.9 -> v3.6.0 03-13; owner orig Dalia Haddad -> Samir Khan 03-27 -> Chen Wei 03-31 -> Oluwaseun Adeyemi 04-02)
- orion-worker | Yara Haddad | 4393 | eu-central-1 | MongoDB 7 | v3.6.0 [03-03] (release orig v3.5.3 -> v3.6.0 03-17; region orig af-south-1 -> us-west-2 04-01 -> eu-central-1 04-06; datastore orig MySQL 8.0 -> MongoDB 7 04-01; port orig 4193 -> 4393 04-04; owner orig Chen Wei -> Yara Haddad 04-08)
- pulse-core | Ugo Bianchi | 5993 | eu-west-1 | Redis 7 | v1.9.19 [03-03] (port orig 3977 -> 7495 03-13 -> 5993 04-03)
- pulse-gate | Elif Yilmaz | 5289 | us-west-2 | MongoDB 7 | v4.0.12 [03-02] (port orig 4252 -> 5165 03-16 -> 5454 04-03 -> 5289 04-04; owner orig Xavier Dubois -> Elif Yilmaz 03-25)
- pulse-gateway | Dalia Haddad | 9539 | af-south-1 | CockroachDB 23 | v2.9.17 [03-01] (release orig v2.9.14 -> v2.9.17 03-24; datastore orig PostgreSQL 15 -> CockroachDB 23 04-07)
- pulse-worker | Tomasz Nowak | 4513 | us-west-2 | PostgreSQL 16 | v2.9.0 [03-01] (region orig eu-west-1 -> us-west-2 03-09; owner orig Emeka Nwosu -> Tomasz Nowak 03-28; release orig v2.8.3 -> v2.9.0 03-28; port orig 6288 -> 4513 04-01)
- quartz-core | Chen Wei | 9545 | us-west-2 | PostgreSQL 15 | v2.5.10 [03-03] (owner orig Elif Yilmaz -> Ifeoma Eze 03-11 -> Emeka Nwosu 03-11 same day -> Chen Wei 03-25; port orig 4988 -> 9545 03-13; release orig v2.5.4 -> v2.5.7 03-25 -> v2.5.10 04-01; datastore orig Redis 7 -> PostgreSQL 15 04-04)
- quartz-gate | Emeka Nwosu | 5921 | us-west-2 | MongoDB 7 | v1.2.17 [03-02] (region orig af-south-1 -> us-west-2 03-10; port orig 3133 -> 5921 03-11; owner orig Chen Wei -> Emeka Nwosu 03-23; datastore orig PostgreSQL 15 -> MongoDB 7 03-31)
- quartz-gateway | Chen Wei | 7228 | ap-southeast-1 | PostgreSQL 16 | v2.1.15 [03-01] (datastore orig PostgreSQL 15 -> PostgreSQL 14 03-18 -> PostgreSQL 16 04-08; port orig 3128 -> 7228 03-28)
- quartz-worker | Priya Raman | 7027 | eu-west-1 | MongoDB 7 | v4.3.6 [03-01] (owner orig Elif Yilmaz -> Kavya Iyer 03-09 -> Chiara Russo 03-18 -> Priya Raman 03-23; region orig us-east-1 -> eu-west-1 03-11 -> back to us-east-1 03-28 -> eu-west-1 04-02)
- sable-core | Jonas Berg | 3037 | eu-west-1 | MySQL 8.0 | v2.8.9 [03-01] (owner orig Freya Lindqvist -> Chen Wei 03-10 -> Gustavo Ramos 03-18 -> Jonas Berg 03-27; port orig 8097 -> 3037 04-06)
- sable-gate | Nadia Petrova | 6080 | eu-central-1 | MySQL 8.0 | v1.2.11 [03-03] (release orig v1.2.9 -> v1.2.11 03-08; datastore orig PostgreSQL 16 -> MySQL 8.0 03-13; port orig 7749 -> 6080 03-24)
- sable-gateway | Chiara Russo | 9044 | af-south-1 | MongoDB 7 | v2.3.2 [03-02] (port orig 5645 -> 9503 03-13 -> 4629 03-14 -> 7036 03-16 -> 9044 04-04; datastore orig PostgreSQL 15 -> MySQL 8.0 03-15 -> MongoDB 7 03-30)
- sable-worker | Amara Okonkwo | 9330 | ap-southeast-1 | MySQL 8.0 | v4.5.0 [03-02] (release orig v4.4.9 -> v4.4.11 03-16 -> v4.5.0 03-16 same day; port orig 5210 -> 9330 03-28; datastore orig MongoDB 7 -> MySQL 8.0 03-29)
- titanium-core | Jonas Berg | 9163 | af-south-1 | CockroachDB 23 | v2.7.2 [03-03] (release orig v2.7.0 -> v2.7.2 03-12; port orig 4301 -> 9411 03-14 -> 9163 03-31; region orig af-south-1 -> ap-southeast-1 03-18 -> back to af-south-1 04-02)
- titanium-gate | Samir Khan | 4720 | eu-west-1 | Redis 7 | v2.2.6 [03-02] (owner orig Dalia Haddad -> Samir Khan 03-13; release orig v2.2.4 -> v2.2.6 03-18)
- titanium-gateway | Zoltan Farkas | 9499 | af-south-1 | PostgreSQL 16 | v3.5.15 [03-01] (release orig v3.5.14 -> v3.5.15 03-09; owner orig Aisha Bello -> Zoltan Farkas 03-10; port orig 3772 -> 9178 03-15 -> 9499 03-16; datastore orig PostgreSQL 15 -> PostgreSQL 16 04-08)
- titanium-worker | Hana Sato | 6950 | us-east-1 | PostgreSQL 16 | v3.0.6 [03-01] (datastore orig CockroachDB 23 -> PostgreSQL 16 04-07)
- vega-core | Ifeoma Eze | 8255 | eu-central-1 | CockroachDB 23 | v4.10.0 [03-01] (datastore orig Redis 7 -> PostgreSQL 16 03-19 -> CockroachDB 23 03-26; release orig v4.9.7 -> v4.10.0 03-21; port orig 3438 -> 8255 04-05)
- vega-gate | Elif Yilmaz | 6269 | eu-west-1 | PostgreSQL 14 | v1.5.0 [03-03] (datastore orig MongoDB 7 -> Redis 7 03-15 -> PostgreSQL 14 03-22; port orig 4612 -> 6269 03-26; release orig v1.4.9 -> v1.4.11 03-30 -> v1.5.0 04-04)
- vega-gateway | Yara Haddad | 3492 | eu-central-1 | PostgreSQL 14 | v1.2.3 [03-02] (release orig v1.1.9 -> v1.2.0 03-11 -> v1.2.3 03-25; owner orig Yara Haddad -> Valentina Cruz 03-21 -> back to Yara Haddad 03-30)
- vega-worker | Freya Lindqvist | 8920 | eu-west-1 | Redis 7 | v1.0.19 [03-02] (release orig v1.0.16 -> v1.0.19 03-13; owner orig Kavya Iyer -> Freya Lindqvist 03-16; region orig ap-southeast-1 -> eu-west-1 04-03)
- zephyr-core | Bram de Vries | 5678 | eu-central-1 | MySQL 8.0 | v1.7.0 [03-02] (release orig v1.6.16 -> v1.6.19 03-09 -> v1.7.0 03-13; region orig us-east-1 -> eu-central-1 04-03; port orig 3161 -> 8141 04-05 -> 5678 04-07)
- zephyr-gate | Yara Haddad | 8958 | eu-west-1 | Redis 7 | v1.3.17 [03-01]
- zephyr-gateway | Nadia Petrova | 5037 | af-south-1 | Redis 7 | v1.1.4 [03-03] (release orig v1.1.3 -> v1.1.4 03-20; region orig us-east-1 -> af-south-1 03-21; port orig 6147 -> 5037 03-30; owner orig Chiara Russo -> Nadia Petrova 03-30)
- zephyr-worker | Rosa Delgado | 4525 | us-east-1 | PostgreSQL 16 | v4.1.0 [03-03] (owner orig Dalia Haddad -> Rosa Delgado 03-09; port orig 4377 -> 6780 03-12 -> 4525 03-27; release orig v4.0.17 -> v4.0.20 03-20 -> v4.1.0 04-06; region orig us-west-2 -> us-east-1 04-05)
(Note: titanium-gate port 4720 is unrelated to ticket HAL-4720. ember-gateway port 9327 is unrelated to ticket HAL-9327.)

## TICKETS (id | sev | summary | service | CURRENT assignee | CURRENT status | history)
- HAL-9632 | SEV3 | intermittent 502 errors on checkout | orion-gateway | Oluwaseun Adeyemi | in progress | opened 03-02; in progress 03-08; blocked 03-17; in progress again 03-20
- HAL-7492 | SEV3 | memory leak after deploy | quartz-core | Bastian Kraus | in progress | opened 03-03 assigned Valentina Cruz; reassigned Chen Wei 03-08; in progress 03-11; reassigned Bastian Kraus 03-24
- HAL-7963 | SEV3 | TLS certificate near expiry | lyra-gate | Yara Haddad | resolved | opened 03-04 (open); in progress 03-10; resolved 03-17
- HAL-3558 | SEV2 | slow query on invoice listing | quartz-worker | Nadia Petrova | in progress | opened 03-05 assigned Freya Lindqvist; reassigned Nadia Petrova 03-20; in progress 03-23
- HAL-5695 | SEV2 | duplicate push notifications | nova-core | Valentina Cruz | in progress | opened 03-06; in progress 04-05
- HAL-1223 | SEV3 | stale cache after failover | pulse-worker | Chen Wei | resolved | opened 03-07 assigned Bram de Vries; in progress 03-08; blocked 03-11; in progress again 03-15; reassigned Aisha Bello 03-19; blocked again 03-25; reassigned Chen Wei 03-28; in progress 04-05; resolved 04-09
- HAL-2850 | SEV1 | OOM kills during nightly batch | billing-core | Liam O'Connor | open | opened 03-08 assigned Ifeoma Eze; reassigned Liam O'Connor 03-25
- HAL-8501 | SEV3 | login loop on Safari | helix-gateway | Emeka Nwosu | in progress | opened 03-09 assigned Gustavo Ramos; in progress 03-20; reassigned Emeka Nwosu 04-01
- HAL-4720 | SEV3 | missing metrics for p99 latency | atlas-core | Nadia Petrova | resolved | opened 03-10 assigned Liam O'Connor; in progress 03-12; resolved 03-17; reassigned Nadia Petrova 03-17 (after resolve)
- HAL-8557 | SEV3 | webhook retries exhausted | cobalt-worker | Nadia Petrova | in progress | opened 03-11 assigned Farid Mansour; in progress 03-13; reassigned Nadia Petrova 04-07
- HAL-5455 | SEV1 | index rebuild stuck at 80 percent | titanium-worker | Liam O'Connor | open | opened 03-12
- HAL-1691 | SEV3 | rate limiter too aggressive | helix-gate | Diego Alvarez | in progress | opened 03-13; in progress 03-29
- HAL-8570 | SEV2 | export CSV truncates unicode | atlas-worker | Chiara Russo | in progress | opened 03-14 assigned Oluwaseun Adeyemi; reassigned Bastian Kraus 03-15; reassigned Chiara Russo 03-17; in progress 03-17
- HAL-2346 | SEV2 | backup job exceeds window | zephyr-gateway | Priya Raman | in progress | opened 03-15; in progress 04-09
- HAL-3891 | SEV2 | presence status flapping | vega-core | Diego Alvarez | in progress | opened 03-16; in progress 03-28
- HAL-5410 | SEV3 | feature store skew between train and serve | pulse-gateway | Aisha Bello | in progress | opened 03-17 assigned Aisha Bello; reassigned Chen Wei 03-19; back to Aisha Bello 03-20; in progress 03-24
- HAL-8991 | SEV1 | CDN purge not propagating | orion-gate | Amara Okonkwo | in progress | opened 03-18; in progress 03-22
- HAL-7175 | SEV1 | queue consumer lag spikes | orion-gate | Dalia Haddad | open | opened 03-19
- HAL-7584 | SEV1 | session tokens not revoked on logout | quartz-worker | Valentina Cruz | open | opened 03-20 assigned Bram de Vries; reassigned Valentina Cruz 03-25
- HAL-3984 | SEV1 | report totals off by one cent | atlas-worker | Aisha Bello | in progress | opened 03-21; in progress 03-25
- HAL-1041 | SEV1 | scheduler fires jobs twice | titanium-gate | Ugo Bianchi | in progress | opened 03-22; in progress 03-26
- HAL-2098 | SEV2 | search ranking ignores synonyms | billing-core | Farid Mansour | open | opened 03-23
- HAL-2960 | SEV2 | alert storm during maintenance | helix-worker | Hana Sato | resolved | opened 03-24 assigned Mei Lin; in progress 03-27; reassigned Hana Sato 03-30; resolved 04-06
- HAL-7003 | SEV2 | mailer bounces from Gmail | kestrel-gateway | Ifeoma Eze | open | opened 03-25 assigned Oluwaseun Adeyemi; reassigned Ifeoma Eze 04-03
- HAL-3091 | SEV1 | gRPC deadline exceeded under load | titanium-core | Elif Yilmaz | open | opened 03-26
- HAL-2067 | SEV2 | audit log gaps after rotation | zephyr-worker | Samir Khan | open | opened 03-27
- HAL-8086 | SEV2 | edge nodes serve stale TLS chain | atlas-worker | Freya Lindqvist | in progress | opened 03-28 assigned Diego Alvarez; reassigned Freya Lindqvist 03-29; in progress 04-02
- HAL-5397 | SEV2 | worker pool starves on large uploads | titanium-worker | Valentina Cruz | open | opened 03-29
- HAL-9261 | SEV3 | billing retries double-charge | pulse-gate | Wanjiru Kamau | open | opened 03-30
- HAL-3995 | SEV2 | dashboard timezone off by an hour | zephyr-core | Mei Lin | open | opened 03-31
- HAL-9327 | SEV1 | kestrel sync drops deletes | quartz-gateway | Valentina Cruz | open | opened 04-01 (note: summary says "kestrel" but service is quartz-gateway)
- HAL-3265 | SEV1 | mosaic thumbnails render black | pulse-gateway | Rosa Delgado | open | opened 04-02 (summary says "mosaic" but service is pulse-gateway)
- HAL-8360 | SEV3 | titanium core panics on nil config | orion-gate | Kavya Iyer | open | opened 04-03 (summary says "titanium" but service is orion-gate)
- HAL-3992 | SEV2 | sable gateway leaks file descriptors | mosaic-gate | Gustavo Ramos | in progress | opened 04-04; in progress 04-05 (summary says "sable" but service is mosaic-gate)
- HAL-6898 | SEV3 | vault-rotator skips one secret | zephyr-core | Freya Lindqvist | open | opened 04-05
- HAL-4945 | SEV1 | canary metrics missing tags | mosaic-worker | Bastian Kraus | open | opened 04-06
(Tickets with no status change after opening are "open".)

## DECISIONS (ADRs)
- ADR-007 (03-04): all internal service-to-service calls use gRPC with mTLS.
- ADR-011 (03-05): feature flags managed in the cobalt-scheduler config store.
- ADR-013 (03-06): log retention 45 days application logs, 400 days audit logs.
- ADR-019 (03-07): canary stage gets 7 percent of production traffic for 30 minutes.
- ADR-023 (03-08): secrets rotate every 21 days via the vault-rotator job.
- ADR-029 (03-09): public APIs are versioned in the URL path, never in headers.
(No new ADRs 03-17..04-09.)

## OTHER FACTS
- ONCALL: primary on-call = Rosa Delgado (handed over 04-07). History: rotation started with Hana Sato (03-02) -> Samir Khan 03-17 -> Rosa Delgado 04-07.
- OFFICE: guest Wi-Fi network = halcyon-guest-7 (03-03).
- PLAN: disaster-recovery drill now scheduled for 2026-04-16 (postponed on 04-02). Originally scheduled for 2026-04-02 (set 03-05).
- FINANCE: Q2 infrastructure budget = EUR 455,000 (revised 03-26). Original: approved at EUR 412,000 on 03-06.
- SOCIAL (low value): floor 3 coffee machine repaired 03-08, 03-09, 03-13, 03-18, 03-19, 03-25; new hires orientation went well 03-09, 03-16, 03-18, 03-28; standup notes posted in #eng-daily 03-10, 03-23, 04-02 (x2), 04-04, 04-06, 04-08; parking garage closed for cleaning 03-15, 03-21, 03-24 (x3), 03-27, 04-01 (x4); floor 2 printer out of toner 03-16, 03-23, 04-03, 04-05; all-hands recording on intranet 03-19, 03-23, 03-25, 03-31, 04-01, 04-09; team lunch moved to Thursday 03-21, 03-25, 03-29, 04-09; reminder to submit expense reports 03-26, 03-28, 03-30, 04-03, 04-07.

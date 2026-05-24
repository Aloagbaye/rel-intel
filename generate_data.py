"""
Phase 1 — Synthetic Relationship Dataset Generator
Mimics RelIntel's data model: companies, contacts, deals, interactions

Output:
  data/companies.json      — 50 companies
  data/contacts.json       — 200 contacts (4 per company avg)
  data/deals.json          — 150 deals (companies ↔ deals)
  data/interactions.json   — 500 interaction notes (emails, meetings, calls)
  data/schema.md           — data model documentation
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────

N_COMPANIES   = 50
N_CONTACTS    = 200
N_DEALS       = 150
N_INTERACTIONS = 500

SECTORS = [
    "Fintech", "HealthTech", "Climate Tech", "Enterprise SaaS",
    "Consumer Tech", "Cybersecurity", "Deep Tech", "EdTech",
    "PropTech", "Logistics Tech"
]

STAGES = ["Lead", "Qualified", "Due Diligence", "Term Sheet", "Closed Won", "Closed Lost", "Passed"]

DEAL_TYPES = ["Series A", "Series B", "Series C", "Seed", "Pre-Seed", "Growth Equity"]

INTERACTION_TYPES = ["email", "meeting", "call", "linkedin_message", "event"]

RELATIONSHIP_STRENGTHS = ["weak", "moderate", "strong", "champion"]

TITLES = [
    "CEO", "CTO", "CFO", "COO", "VP Engineering", "VP Product",
    "Head of Growth", "Founder", "Co-Founder", "Partner",
    "Managing Director", "Principal", "Associate", "Director of Sales"
]

TEAM_MEMBERS = [
    {"id": "usr_001", "name": "Priya Mehta",    "email": "priya@vc-firm.com"},
    {"id": "usr_002", "name": "James Okafor",   "email": "james@vc-firm.com"},
    {"id": "usr_003", "name": "Sarah Chen",     "email": "sarah@vc-firm.com"},
    {"id": "usr_004", "name": "Marcus Webb",    "email": "marcus@vc-firm.com"},
    {"id": "usr_005", "name": "Leila Nazari",   "email": "leila@vc-firm.com"},
]

# ─── Generators ───────────────────────────────────────────────────────────────

def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def rand_date(start_days_ago=730, end_days_ago=0) -> str:
    delta = random.randint(end_days_ago, start_days_ago)
    d = datetime.now() - timedelta(days=delta)
    return d.strftime("%Y-%m-%d")

def rand_datetime(start_days_ago=730, end_days_ago=0) -> str:
    delta = random.randint(end_days_ago, start_days_ago)
    d = datetime.now() - timedelta(days=delta, hours=random.randint(0,23), minutes=random.randint(0,59))
    return d.strftime("%Y-%m-%dT%H:%M:%S")

# ── Companies ─────────────────────────────────────────────────────────────────

def generate_companies() -> list[dict]:
    companies = []
    for _ in range(N_COMPANIES):
        sector = random.choice(SECTORS)
        founded = random.randint(2010, 2023)
        company = {
            "company_id":        make_id("co"),
            "name":              fake.company().replace(",", "").replace(".", "") + " " + random.choice(["AI", "Labs", "Tech", "IO", "HQ", ""]).strip(),
            "sector":            sector,
            "stage":             random.choice(["Seed", "Series A", "Series B", "Series C", "Growth"]),
            "founded_year":      founded,
            "hq_city":           fake.city(),
            "hq_country":        random.choice(["USA", "USA", "USA", "Canada", "UK", "Germany", "France", "Israel", "India", "Singapore"]),
            "headcount":         random.choice([5, 12, 25, 50, 100, 200, 500, 1000]),
            "website":           f"https://www.{fake.domain_name()}",
            "description":       f"{sector} company building {fake.bs()}.",
            "linkedin_url":      f"https://linkedin.com/company/{fake.slug()}",
            "last_interaction":  rand_date(365, 0),
            "relationship_strength": random.choice(RELATIONSHIP_STRENGTHS),
            "tags":              random.sample(["portfolio", "prospect", "passed", "warm intro", "cold outreach", "referral"], k=random.randint(1,3)),
            "created_at":        rand_date(700, 400),
            "updated_at":        rand_date(100, 0),
        }
        companies.append(company)
    return companies

# ── Contacts ──────────────────────────────────────────────────────────────────

def generate_contacts(companies: list[dict]) -> list[dict]:
    contacts = []
    # Assign 3–5 contacts per company, ensure all companies have at least one
    company_ids = [c["company_id"] for c in companies]
    assigned = {cid: [] for cid in company_ids}

    for _ in range(N_CONTACTS):
        company_id = random.choice(company_ids)
        contact = {
            "contact_id":        make_id("ct"),
            "first_name":        fake.first_name(),
            "last_name":         fake.last_name(),
            "email":             fake.email(),
            "linkedin_url":      f"https://linkedin.com/in/{fake.slug()}",
            "title":             random.choice(TITLES),
            "company_id":        company_id,
            "location":          fake.city() + ", " + fake.country_code(),
            "relationship_owner": random.choice(TEAM_MEMBERS)["id"],
            "relationship_strength": random.choice(RELATIONSHIP_STRENGTHS),
            "last_interaction":  rand_date(365, 0),
            "interaction_count": random.randint(1, 40),
            "notes":             fake.paragraph(nb_sentences=2),
            "tags":              random.sample(["decision maker", "champion", "technical", "gatekeeper", "influencer", "warm"], k=random.randint(1,3)),
            "created_at":        rand_date(700, 400),
            "updated_at":        rand_date(60, 0),
        }
        assigned[company_id].append(contact["contact_id"])
        contacts.append(contact)
    return contacts

# ── Deals ─────────────────────────────────────────────────────────────────────

def generate_deals(companies: list[dict], contacts: list[dict]) -> list[dict]:
    deals = []
    contacts_by_company = {}
    for c in contacts:
        contacts_by_company.setdefault(c["company_id"], []).append(c["contact_id"])

    for _ in range(N_DEALS):
        company = random.choice(companies)
        stage = random.choice(STAGES)
        deal_type = random.choice(DEAL_TYPES)
        created = rand_date(600, 30)
        company_contacts = contacts_by_company.get(company["company_id"], [])

        deal = {
            "deal_id":           make_id("dl"),
            "name":              f"{company['name']} — {deal_type}",
            "company_id":        company["company_id"],
            "deal_type":         deal_type,
            "stage":             stage,
            "amount_usd":        random.choice([500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000]),
            "currency":          "USD",
            "lead_partner":      random.choice(TEAM_MEMBERS)["id"],
            "deal_team":         random.sample([t["id"] for t in TEAM_MEMBERS], k=random.randint(1,3)),
            "key_contacts":      random.sample(company_contacts, k=min(len(company_contacts), random.randint(1,3))) if company_contacts else [],
            "created_at":        created,
            "updated_at":        rand_date(30, 0),
            "close_date":        rand_date(180, 0) if stage in ["Closed Won", "Closed Lost"] else None,
            "sector":            company["sector"],
            "tags":              random.sample(["high priority", "founder led", "competitive", "exclusive", "warm intro", "board seat"], k=random.randint(1,2)),
            "memo_url":          f"https://drive.google.com/file/{uuid.uuid4().hex[:12]}",
        }
        deals.append(deal)
    return deals

# ── Interaction Notes ─────────────────────────────────────────────────────────

MEETING_TEMPLATES = [
    "Met with {contact} ({title}) at {company}. {summary} Next steps: {next_steps}",
    "Intro call with {contact} from {company}. {summary} Will follow up on {topic}.",
    "Board meeting at {company} with {contact}. Key discussion: {summary}. Action items: {next_steps}",
    "Catch-up with {contact} ({title}). {summary}",
    "Partner meeting at {company} offices. Attendees: {contact} + team. {summary}",
]

EMAIL_TEMPLATES = [
    "Re: {topic} — {contact} shared {summary}. Responded with {next_steps}.",
    "Received update from {contact} at {company} re: {topic}. {summary}",
    "Follow-up email from {contact}. {summary} Asked for {topic}.",
    "Intro email from {contact} ({title}, {company}). {summary}",
    "Thread with {contact} about {topic}. {summary} Next: {next_steps}",
]

CALL_TEMPLATES = [
    "Quick call with {contact} ({company}). {summary} Will reconnect in {timeframe}.",
    "Investor update call with {contact}. {summary}. Key metric: {metric}.",
    "Reference call on {company} with {contact} ({title}). {summary}",
    "Due diligence call: {contact} walked through {topic}. {summary}",
    "Pipeline review call. Discussed {company} — {summary}. {next_steps}",
]

def fill_template(template: str, contact: dict, company: dict) -> str:
    topics = ["ARR growth", "product roadmap", "team expansion", "GTM strategy",
              "technical architecture", "competitive landscape", "customer traction",
              "fundraising timeline", "board composition", "unit economics"]
    summaries = [
        f"They're seeing strong traction in {company['sector']} with {random.randint(20,200)}% YoY growth.",
        f"The team is expanding rapidly — now at {company['headcount']} people and hiring aggressively.",
        f"Product is live with {random.randint(10,500)} enterprise customers and {random.randint(60,95)}% NRR.",
        f"Founder is deeply technical with prior exit at {fake.company()}.",
        f"Competitive moat is clear — {random.randint(2,5)} years of proprietary data and strong network effects.",
        f"Burn rate is under control at ${random.randint(200,800)}K/month with {random.randint(12,36)} months runway.",
        f"They've signed {random.randint(3,20)} LOIs in the last quarter — pipeline is strong.",
        f"Technical differentiation is significant — {random.randint(5,30)} patents pending.",
        f"NPS is {random.randint(50,85)} and customer churn is below {random.randint(2,8)}%.",
        f"Co-founder relationship is solid — complementary skill sets across technical and go-to-market.",
    ]
    next_steps_options = [
        "Schedule follow-up in 2 weeks.",
        "Send term sheet draft by Friday.",
        "Intro to portfolio company for reference check.",
        "Review data room materials.",
        "Loop in technical partner for deep dive.",
        "Check back after Series B closes.",
        "Refer to Sarah for HealthTech diligence.",
        "Request updated financial model.",
        "Connect with reference customers.",
        "Share relevant portfolio operator.",
    ]
    return template.format(
        contact=f"{contact['first_name']} {contact['last_name']}",
        title=contact["title"],
        company=company["name"],
        topic=random.choice(topics),
        summary=random.choice(summaries),
        next_steps=random.choice(next_steps_options),
        timeframe=random.choice(["2 weeks", "next quarter", "30 days", "after their board meeting"]),
        metric=f"${random.randint(1,50)}M ARR",
    )

def generate_interactions(companies: list[dict], contacts: list[dict], deals: list[dict]) -> list[dict]:
    interactions = []
    company_map = {c["company_id"]: c for c in companies}
    contacts_by_company = {}
    for c in contacts:
        contacts_by_company.setdefault(c["company_id"], []).append(c)

    deal_map = {}
    for d in deals:
        deal_map.setdefault(d["company_id"], []).append(d["deal_id"])

    for _ in range(N_INTERACTIONS):
        company = random.choice(companies)
        cid = company["company_id"]
        company_contacts = contacts_by_company.get(cid, [])
        if not company_contacts:
            continue
        contact = random.choice(company_contacts)
        itype = random.choice(INTERACTION_TYPES)

        if itype == "meeting":
            template = random.choice(MEETING_TEMPLATES)
        elif itype == "email":
            template = random.choice(EMAIL_TEMPLATES)
        elif itype == "call":
            template = random.choice(CALL_TEMPLATES)
        else:
            template = random.choice(MEETING_TEMPLATES + EMAIL_TEMPLATES)

        body = fill_template(template, contact, company)

        interaction = {
            "interaction_id":    make_id("ix"),
            "type":              itype,
            "date":              rand_datetime(540, 0),
            "company_id":        cid,
            "contact_ids":       [contact["contact_id"]] + (
                                     [random.choice(company_contacts)["contact_id"]]
                                     if len(company_contacts) > 1 and random.random() > 0.6 else []
                                 ),
            "deal_id":           random.choice(deal_map.get(cid, [None])),
            "logged_by":         random.choice(TEAM_MEMBERS)["id"],
            "team_members_cc":   random.sample([t["id"] for t in TEAM_MEMBERS], k=random.randint(0,2)),
            "subject":           f"{itype.replace('_',' ').title()}: {company['name']}",
            "body":              body,
            "sentiment":         random.choice(["positive", "positive", "positive", "neutral", "negative"]),
            "tags":              random.sample(["diligence", "portfolio", "intro", "follow-up", "reference", "update", "fundraising"], k=random.randint(1,3)),
            "source_type":       itype,
            "created_at":        rand_datetime(540, 0),
        }
        interactions.append(interaction)

    # Sort by date desc
    interactions.sort(key=lambda x: x["date"], reverse=True)
    return interactions

# ── Schema Docs ───────────────────────────────────────────────────────────────

SCHEMA_MD = """# RelIntel RAG — Data Model (Phase 1)

## Entity Overview

| Entity | Count | Description |
|---|---|---|
| Companies | 50 | Portfolio companies, prospects, and passed deals |
| Contacts | 200 | Founders, executives, and key stakeholders |
| Deals | 150 | Investment opportunities across deal stages |
| Interactions | 500 | Emails, meetings, calls, and LinkedIn messages |

## Schema Details

### companies.json
| Field | Type | Description |
|---|---|---|
| company_id | str | Unique identifier (co_xxxxxxxx) |
| name | str | Company name |
| sector | str | Industry vertical |
| stage | str | Company growth stage |
| founded_year | int | Year founded |
| hq_city / hq_country | str | Headquarters location |
| headcount | int | Approximate employee count |
| description | str | One-line company description |
| relationship_strength | str | weak / moderate / strong / champion |
| tags | list[str] | Freeform classification tags |
| last_interaction | date | Most recent interaction date |

### contacts.json
| Field | Type | Description |
|---|---|---|
| contact_id | str | Unique identifier (ct_xxxxxxxx) |
| first_name / last_name | str | Contact name |
| email | str | Contact email |
| title | str | Job title |
| company_id | str | FK → companies |
| relationship_owner | str | Team member ID (usr_xxx) |
| relationship_strength | str | weak / moderate / strong / champion |
| interaction_count | int | Total logged interactions |
| tags | list[str] | decision maker / champion / technical / etc. |

### deals.json
| Field | Type | Description |
|---|---|---|
| deal_id | str | Unique identifier (dl_xxxxxxxx) |
| name | str | Deal name (Company — Round) |
| company_id | str | FK → companies |
| deal_type | str | Seed / Series A / B / C / Growth |
| stage | str | Lead → Qualified → Due Diligence → Term Sheet → Closed |
| amount_usd | int | Deal size in USD |
| lead_partner | str | FK → team member |
| deal_team | list[str] | FKs → team members |
| key_contacts | list[str] | FKs → contacts |
| sector | str | Inherited from company |

### interactions.json
| Field | Type | Description |
|---|---|---|
| interaction_id | str | Unique identifier (ix_xxxxxxxx) |
| type | str | email / meeting / call / linkedin_message / event |
| date | datetime | Interaction timestamp |
| company_id | str | FK → companies |
| contact_ids | list[str] | FKs → contacts |
| deal_id | str | FK → deals (nullable) |
| logged_by | str | FK → team member |
| subject | str | Short title/subject |
| body | str | Full interaction text (primary RAG chunk source) |
| sentiment | str | positive / neutral / negative |
| tags | list[str] | diligence / portfolio / intro / follow-up / etc. |
| source_type | str | Same as type — for metadata filtering |

## Metadata Fields for RAG Filtering

The following fields are designed as pre-filter candidates in the retrieval pipeline:

- `company_id` — retrieve all interactions for a specific company
- `deal_id` — retrieve all interactions related to a deal
- `source_type` — filter by interaction type (e.g., only meetings)
- `sentiment` — filter by tone
- `logged_by` — filter by team member
- `date` — range filtering (last 90 days, last year, etc.)
- `sector` — cross-company sector queries
- `tags` — semantic category filtering
"""

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating companies...")
    companies = generate_companies()

    print("Generating contacts...")
    contacts = generate_contacts(companies)

    print("Generating deals...")
    deals = generate_deals(companies, contacts)

    print("Generating interactions...")
    interactions = generate_interactions(companies, contacts, deals)

    # Write files
    for name, data in [
        ("companies", companies),
        ("contacts", contacts),
        ("deals", deals),
        ("interactions", interactions),
    ]:
        path = DATA_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ {path.name}: {len(data)} records")

    with open(DATA_DIR / "schema.md", "w") as f:
        f.write(SCHEMA_MD)
    print("  ✓ schema.md written")

    # Stats
    print("\n── Dataset Summary ──────────────────────────────")
    print(f"  Companies:    {len(companies)}")
    print(f"  Contacts:     {len(contacts)}")
    print(f"  Deals:        {len(deals)}")
    print(f"  Interactions: {len(interactions)}")

    sectors = {}
    for c in companies:
        sectors[c["sector"]] = sectors.get(c["sector"], 0) + 1
    print(f"\n  Sector distribution:")
    for s, n in sorted(sectors.items(), key=lambda x: -x[1]):
        print(f"    {s}: {n}")

    stages = {}
    for d in deals:
        stages[d["stage"]] = stages.get(d["stage"], 0) + 1
    print(f"\n  Deal stage distribution:")
    for s, n in sorted(stages.items(), key=lambda x: -x[1]):
        print(f"    {s}: {n}")

    itypes = {}
    for i in interactions:
        itypes[i["type"]] = itypes.get(i["type"], 0) + 1
    print(f"\n  Interaction type distribution:")
    for t, n in sorted(itypes.items(), key=lambda x: -x[1]):
        print(f"    {t}: {n}")

    total_chars = sum(len(i["body"]) for i in interactions)
    print(f"\n  Total interaction body text: {total_chars:,} chars")
    print(f"  Avg body length: {total_chars // len(interactions)} chars")
    print("\n✅ Phase 1 complete.")

if __name__ == "__main__":
    main()

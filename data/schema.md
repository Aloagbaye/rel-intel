# RelIntel RAG — Data Model (Phase 1)

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

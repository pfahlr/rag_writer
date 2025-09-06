# Crossref API – `/works` Endpoint

[See the entire API documentation site here](https://api.crossref.org/swagger-ui/index.html)

The `/works` endpoint returns a list of works (journal articles, conference proceedings, books, components, etc). By default, it returns **20 items per page**.

---

## 🔍 Queries

### Free-form Search

```http
/works?query=renear+ontologies
```

Matches works with terms *renear* or *ontologies*.

### Field Queries

Search specific metadata fields:

```http
/works?query.author=richard+feynman
```

Combine multiple queries (ANDed together):

```http
/works?query.title=room+at+the+bottom&query.author=richard+feynman
```

#### Supported Field Queries

* `query.affiliation` – contributor affiliations
* `query.author` – author names
* `query.bibliographic` – citation lookup (titles, authors, ISSNs, years)
* `query.chair` – chair names
* `query.container-title` – journal/publication name
* `query.contributor` – author, editor, chair, translator names
* `query.degree` – degree
* `query.description` – description
* `query.editor` – editor names
* `query.event-acronym`, `query.event-location`, `query.event-name`, `query.event-sponsor`, `query.event-theme`
* `query.funder-name` – funder name
* `query.publisher-location`, `query.publisher-name`
* `query.standards-body-acronym`, `query.standards-body-name`
* `query.title` – title
* `query.translator` – translator names

---

## 📑 Sorting

Use `sort` and `order` (default `desc`).

Example:

```http
/works?query=josiah+carberry&sort=published&order=asc
```

Supported sort fields:

* `created`, `deposited`, `indexed`, `updated`
* `issued`, `published`, `published-online`, `published-print`
* `is-referenced-by-count`, `references-count`
* `relevance`, `score`

---

## 📊 Facets

Retrieve summary statistics:

```http
/works?facet=type-name:*
```

Supported facets include:

* `affiliation`, `archive`, `assertion`, `assertion-group`
* `category-name`, `container-title`
* `funder-doi`, `funder-name`
* `issn`, `journal-issue`, `journal-volume`
* `license`, `link-application`, `orcid`, `publisher-name`
* `relation-type`, `ror-id`, `source`, `type-name`, `update-type`

---

## 🎛️ Filters

Select items by criteria:

```http
/works?filter=type:dataset
```

Combine filters with commas (AND) or repeat a filter (OR):

```http
/works?filter=is-update:true,from-pub-date:2014-03-03,funder:10.13039/100000001,funder:10.13039/100000050
```

Special dot filters apply to related record types:

```http
/works?filter=award.number:CBET-0756451,award.funder:10.13039/100000001
```

### Date Filters

* Format: ISO 8601 (`YYYY`, `YYYY-MM-DD`, `YYYY-MM-DDThh:mm:ss`)
* Inclusive
* Examples:

  * `from-pub-date:2018-09-18`
  * `from-created-date:2016-02-29,until-created-date:2016-02-29`
  * `until-created-date:2010-06`
  * `from-update-date:2022-03-01T12,until-update-date:2022-03-01T12`

---

## 🔧 Parameters

| Name     | Type    | Description                                      |
| -------- | ------- | ------------------------------------------------ |
| `rows`   | integer | Number of rows per page (default: 20, max: 1000) |
| `offset` | integer | Number of rows to skip                           |
| `order`  | string  | `asc` or `desc`                                  |
| `sort`   | string  | Field to sort by                                 |
| `facet`  | string  | Facet field and count limit                      |
| `sample` | integer | Randomly sample N works                          |
| `select` | string  | Comma-separated list of fields to return         |
| `filter` | string  | Comma-separated list of filters                  |
| `cursor` | string  | Deep paging cursor                               |
| `mailto` | string  | Email address for polite pool                    |
| `query`  | string  | Free text query                                  |

---

## 📄 Pagination

* **Offset-based:**

  ```http
  /works?query=allen+renear&rows=5&offset=5
  ```

* **Cursor-based (deep paging, unlimited):**

  ```http
  /members/311/works?filter=type:journal-article&cursor=*
  ```

Use `next-cursor` in response for subsequent pages.

---

## 🎲 Sampling

Random results:

```http
/works?sample=10
```

---

## ✅ Response Example

```json
{
  "status": "ok",
  "message-type": "work-list",
  "message-version": "1.0.0",
  "message": {
    "items-per-page": 20,
    "query": {
      "start-index": 0,
      "search-terms": "string"
    },
    "total-results": 100,
    "next-cursor": "string",
    "items": [
      {
        "DOI": "10.xxxx/xxxx",
        "title": ["Example Work"],
        "author": [
          { "given": "Jane", "family": "Doe", "ORCID": "0000-0001-..." }
        ],
        "publisher": "Example Publisher",
        "issued": { "date-parts": [[2024, 5, 1]] },
        "URL": "https://doi.org/10.xxxx/xxxx",
        "type": "journal-article"
      }
    ]
  }
}
```

---

# 📑 Quick Reference Cheat Sheet

## Core Params

| Param                                                    | Description                                         |
| -------------------------------------------------------- | --------------------------------------------------- |
| `query`                                                  | Free-text search query                              |
| `query.author`, `query.title`, `query.affiliation`, etc. | Field-specific queries                              |
| `rows`                                                   | Number of items per page (max 1000)                 |
| `offset`                                                 | Skip N results                                      |
| `sort`                                                   | Sort field (`issued`, `created`, `relevance`, etc.) |
| `order`                                                  | `asc` / `desc`                                      |
| `facet`                                                  | Summary stats by field                              |
| `filter`                                                 | Filter results (dates, type, funder, etc.)          |
| `cursor`                                                 | Cursor string for deep paging                       |
| `sample`                                                 | Return random sample                                |
| `select`                                                 | Restrict fields in response                         |
| `mailto`                                                 | Identify your client via email                      |

## Filters (Examples)

* `type:dataset`
* `is-update:true`
* `from-pub-date:2018-01-01`
* `until-created-date:2016-02-29`
* `award.number:CBET-0756451`

## Sort Fields

* `created`, `updated`, `issued`, `published`
* `published-online`, `published-print`
* `relevance`, `score`
* `is-referenced-by-count`, `references-count`

---

✅ Now you have **matching unified docs** for both **Open Library** and **Crossref** — detailed explanations followed by cheat sheets.

Would you like me to bundle these into a **single combined reference file** (`api_docs.md`) so you can keep all your API notes together in one place?

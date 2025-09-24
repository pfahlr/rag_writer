# arXiv API Documentation

[See the entire API documentation site here](https://info.arxiv.org/help/api/user-manual.html)

The arXiv API provides programmatic access to hundreds of thousands of e-prints hosted on [arXiv.org](http://arxiv.org). Results are returned in **Atom 1.0 XML format**, which makes them easy for programs to parse.

---

## 🔍 API QuickStart

* Base URL:

  ```
  http://export.arxiv.org/api/query
  ```
* Example (search for articles containing *electron*):

  ```
  http://export.arxiv.org/api/query?search_query=all:electron
  ```
* Combine terms:

  ```
  http://export.arxiv.org/api/query?search_query=all:electron+AND+all:proton
  ```

---

## 📑 Query Parameters

| Name           | Type    | Default    | Required | Description                                                   |
| -------------- | ------- | ---------- | -------- | ------------------------------------------------------------- |
| `search_query` | string  | None       | No       | Query string (fielded or free text)                           |
| `id_list`      | string  | None       | No       | Comma-delimited list of arXiv IDs                             |
| `start`        | integer | 0          | No       | Index of first result (0-based)                               |
| `max_results`  | integer | 10         | No       | Number of results to return (max 2000 per request, 30k total) |
| `sortBy`       | string  | relevance  | No       | `relevance`, `lastUpdatedDate`, `submittedDate`               |
| `sortOrder`    | string  | descending | No       | `ascending` or `descending`                                   |

---

## 🔎 search\_query Construction

Each article is divided into searchable fields. Use prefixes to target them:

| Prefix | Field                              |
| ------ | ---------------------------------- |
| `ti`   | Title                              |
| `au`   | Author                             |
| `abs`  | Abstract                           |
| `co`   | Comment                            |
| `jr`   | Journal Reference                  |
| `cat`  | Subject Category                   |
| `rn`   | Report Number                      |
| `id`   | Identifier (use `id_list` instead) |
| `all`  | All fields                         |

### Examples

* All articles by Adrian Del Maestro:

  ```
  search_query=au:del_maestro
  ```
* Works with "checkerboard" in title:

  ```
  search_query=ti:checkerboard
  ```
* Filter with Boolean logic:

  ```
  search_query=au:del_maestro+AND+ti:checkerboard
  search_query=au:del_maestro+ANDNOT+ti:checkerboard
  search_query=au:del_maestro+ANDNOT+(ti:checkerboard+OR+ti:Pyrochlore)
  search_query=au:del_maestro+AND+ti:"quantum criticality"
  ```

---

## 📄 Pagination

Use `start` and `max_results` for paging:

```
http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=10
http://export.arxiv.org/api/query?search_query=all:electron&start=10&max_results=10
```

⚠️ Notes:

* Max per request: 2000 results
* Max total: 30,000 results
* Insert **3-second delay** between requests to be polite

---

## 🗂️ Response Format (Atom 1.0)

Every response is an Atom `<feed>` with metadata and one or more `<entry>` elements.

### Feed Metadata

* `<title>` – Canonicalized query string
* `<id>` – Unique query ID
* `<updated>` – Last updated (midnight daily)
* `<link>` – Canonical query URL
* `<opensearch:totalResults>` – Total results count
* `<opensearch:startIndex>` – Start index
* `<opensearch:itemsPerPage>` – Results per page

### Entry Metadata

* `<title>` – Article title
* `<id>` – Article URL (abs page)
* `<published>` – Date first submitted
* `<updated>` – Date this version submitted
* `<summary>` – Abstract
* `<author>` – Author(s); optional `<arxiv:affiliation>`
* `<link>` – Links to abstract, PDF, and DOI
* `<category>` – Subject classifications
* `<arxiv:primary_category>` – Primary subject class
* `<arxiv:comment>` – Author comments
* `<arxiv:journal_ref>` – Journal reference
* `<arxiv:doi>` – DOI if available

---

## ⚠️ Errors

Errors are also returned as Atom feeds. Examples:

| Query                       | Error                           |
| --------------------------- | ------------------------------- |
| `?start=not_an_int`         | start must be an integer        |
| `?start=-1`                 | start must be >= 0              |
| `?max_results=not_an_int`   | max\_results must be an integer |
| `?max_results=-1`           | max\_results must be >= 0       |
| `?id_list=1234.1234`        | malformed id                    |
| `?id_list=cond—mat/0709123` | malformed id                    |

---

## 🖥️ Code Examples

### Python 3

```python
import urllib.request as libreq

url = "http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=1"
with libreq.urlopen(url) as response:
    data = response.read()
print(data)
```

### Ruby

```ruby
require 'net/http'
require 'uri'

url = URI.parse('http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=1')
res = Net::HTTP.get_response(url)
puts res.body
```

### PHP

```php
<?php
$url = 'http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=1';
$response = file_get_contents($url);
print_r($response);
?>
```

---

# 📑 Quick Reference Cheat Sheet

## Core Params

| Param          | Description                                          |
| -------------- | ---------------------------------------------------- |
| `search_query` | Query string (with field prefixes and Boolean logic) |
| `id_list`      | Comma-separated arXiv IDs                            |
| `start`        | Start index (0-based)                                |
| `max_results`  | Number of results (≤2000 per request, ≤30k total)    |
| `sortBy`       | `relevance`, `lastUpdatedDate`, `submittedDate`      |
| `sortOrder`    | `ascending`, `descending`                            |

## Field Prefixes

* `ti:` Title
* `au:` Author
* `abs:` Abstract
* `co:` Comment
* `jr:` Journal Reference
* `cat:` Category
* `rn:` Report Number
* `all:` All fields

## Boolean Operators

* `AND`
* `OR`
* `ANDNOT`

## Grouping

* Parentheses: `%28 ... %29`
* Quotes: `%22phrase here%22`
* Spaces: `+`

## Versioning

* Use `id_list=cond-mat/0207270` for latest version
* Use `id_list=cond-mat/0207270v1` for specific version

## 🧩 Integration Notes

The metadata scanner uses the [arXiv API](https://arxiv.org/help/api/user-manual) as a
fallback when a DOI cannot be resolved via Crossref. The API is queried at
`https://export.arxiv.org/api/query` with the `id_list` parameter set to the arXiv
identifier extracted from a DOI such as `10.48550/arXiv.XXXX`.

Because the response is an Atom feed, the scanner parses the first `<entry>` element to
obtain fields including title, authors, publication date, and DOI.

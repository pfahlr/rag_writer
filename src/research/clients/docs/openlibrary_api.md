# Open Library API Documentation

[See the entire API documentation site here](https://openlibrary.org/developers/api)

Open Library provides a suite of RESTful APIs for accessing book, author, subject, and cover data. Responses are available in **JSON**, **YAML**, and **RDF/XML** formats.

⚠️ **Note:** For frequent use (multiple calls/minute), set a `User-Agent` header with your application name + contact info, or requests may be blocked.

---

## 🔍 Book Search API

**Endpoint:**

```
https://openlibrary.org/search.json
```

### Parameters

* `q` – Search query (Solr syntax)
* `fields` – Fields to return (`*` for all; use `availability` to fetch availability info)
* `sort` – `new`, `old`, `random`, `key`, or other facets
* `lang` – Preferred language (ISO 639-1 code, e.g. `fr`)
* `offset` / `limit` – Pagination
* `page` / `limit` – Alternative pagination (page starts at 1)

### Examples

```http
/search.json?q=the+lord+of+the+rings
/search.json?title=the+lord+of+the+rings
/search.json?author=tolkien&sort=new
/search.json?q=the+lord+of+the+rings&page=2
```

**Response:**

```json
{
  "start": 0,
  "num_found": 629,
  "docs": [
    {
      "title": "The Lord of the Rings",
      "author_name": ["J. R. R. Tolkien"],
      "first_publish_year": 1954,
      "key": "OL27448W",
      "edition_count": 120,
      "cover_i": 258027,
      "has_fulltext": true,
      "public_scan_b": true
    }
  ]
}
```

---

## 📚 Works & Editions APIs

### Works API

```
https://openlibrary.org/works/{work_id}.json
https://openlibrary.org/works/{work_id}/editions.json
```

Example:

* Work metadata: `https://openlibrary.org/works/OL45804W.json`
* Editions: `https://openlibrary.org/works/OL45804W/editions.json`
* Ratings: `https://openlibrary.org/works/OL18020194W/ratings.json`
* Bookshelves: `https://openlibrary.org/works/OL18020194W/bookshelves.json`

### Editions API

```
https://openlibrary.org/books/{edition_id}.json
```

Example:
`https://openlibrary.org/books/OL7353617M.json`

### ISBN API

```
https://openlibrary.org/isbn/{isbn}.json
```

Redirects to the edition page. Example:
`https://openlibrary.org/isbn/9780140328721.json`

---

## 👤 Authors API

### Search Authors

```
https://openlibrary.org/search/authors.json?q={name}
```

Example:

```
/search/authors.json?q=j%20k%20rowling
```

### Author Data

```
https://openlibrary.org/authors/{author_id}.json
```

Example:
`https://openlibrary.org/authors/OL23919A.json`

### Works by Author

```
https://openlibrary.org/authors/{author_id}/works.json
```

Supports `limit` and `offset`.

---

## 📖 Subjects API

### Works by Subject

```
/subjects/{subject}.json
```

Example:
`/subjects/love.json`

### With details

```
/subjects/{subject}.json?details=true
```

Includes related subjects, publishers, authors, and publishing history.

### Parameters

* `details` – Include extended metadata
* `ebooks=true` – Only include works with ebooks
* `published_in=1500-1600` – Filter by year range
* `limit` / `offset` – Pagination

---

## 🔍 Search Inside API (Experimental)

Search text within digitized books.

**Example:**

```
https://ia800204.us.archive.org/fulltext/inside.php?item_id=designevaluation25clin&doc=designevaluation25clin&path=/27/items/designevaluation25clin&q="library science"
```

Response includes page count, OCR text matches, and bounding boxes.

---

## 📘 Read API

Turn identifiers (ISBN, LCCN, OCLC, OLID) into links to readable/borrowable books.

### Single Request

```
http://openlibrary.org/api/volumes/brief/{id-type}/{id-value}.json
```

Example:
`http://openlibrary.org/api/volumes/brief/isbn/0596156715.json`

### Multi Request

```
http://openlibrary.org/api/volumes/brief/json/{request-list}
```

---

## 🖼️ Covers API

**Endpoint Pattern:**

```
https://covers.openlibrary.org/b/{key}/{value}-{size}.jpg
```

* Keys: `ISBN`, `OLID`, `OCLC`, `LCCN`, `ID`
* Sizes: `S` (small), `M` (medium), `L` (large)

Example:
`https://covers.openlibrary.org/b/isbn/0385472579-S.jpg`

**Author photos:**

```
https://covers.openlibrary.org/a/olid/{author_id}-{size}.jpg
```

Example:
`https://covers.openlibrary.org/a/olid/OL229501A-S.jpg`

---

## 📝 Recent Changes API

### Endpoints

* `http://openlibrary.org/recentchanges.json`
* `http://openlibrary.org/recentchanges/{YYYY}.json`
* `http://openlibrary.org/recentchanges/{YYYY}/{MM}.json`
* `http://openlibrary.org/recentchanges/{YYYY}/{MM}/{DD}.json`
* `http://openlibrary.org/recentchanges/{KIND}.json`

### Parameters

* `limit` (default 100, max 1000)
* `offset` (max 10000)
* `bot=true|false` – filter by bot/human changes

---

## 📋 Lists API

* Endpoints:

  ```
  /people/{username}/lists.json
  /works/{id}/lists.json
  /books/{id}/lists.json
  ```
* Lists contain works, editions, seeds, and subjects.
* Actions: create, read, search, add/remove seeds, fetch editions/subjects.
* Limited to 100 results per query.
* Params:

  * `limit`, `offset` – pagination
  * JSON body params for POST (e.g. `name`, `description`, `seeds`, `tags`)

---

## ⚡ Bulk Access

For large-scale use, do **not** scrape APIs. Instead, use monthly data dumps from [Open Library Bulk Downloads](https://openlibrary.org/developers/dumps).

---

# 📑 Quick Reference Cheat Sheet

## Search API

| Param    | Type    | Description                                          |
| -------- | ------- | ---------------------------------------------------- |
| `q`      | string  | Search query (Solr syntax)                           |
| `fields` | string  | Comma-separated fields (`*` for all, `availability`) |
| `sort`   | string  | `new`, `old`, `random`, `key`, etc.                  |
| `lang`   | string  | Preferred language (ISO 639-1)                       |
| `offset` | integer | Pagination start index                               |
| `limit`  | integer | Page size                                            |
| `page`   | integer | Page number (starts at 1)                            |

## Works & Editions

* `/works/{id}.json`
* `/works/{id}/editions.json`
* `/books/{id}.json`
* `/isbn/{isbn}.json`

## Authors API

| Endpoint                   | Params            |
| -------------------------- | ----------------- |
| `/search/authors.json`     | `q`               |
| `/authors/{id}.json`       | none              |
| `/authors/{id}/works.json` | `limit`, `offset` |

## Subjects API

| Param          | Description                     |
| -------------- | ------------------------------- |
| `details`      | `true` for extended metadata    |
| `ebooks`       | `true` = only works with ebooks |
| `published_in` | Year range filter               |
| `limit`        | Max works to return             |
| `offset`       | Pagination offset               |

## Covers API

Pattern:
`/b/{key}/{value}-{size}.jpg`

* Keys: ISBN, OLID, OCLC, LCCN, ID
* Sizes: S, M, L
* `default=false` returns 404 instead of blank

Author photos: `/a/olid/{id}-{size}.jpg`

## RecentChanges API

| Param    | Description                                   |
| -------- | --------------------------------------------- |
| `limit`  | Max entries (default 100, max 1000)           |
| `offset` | Max 10000                                     |
| `bot`    | `true` for bots only, `false` for humans only |

## Lists API

* `limit`, `offset` for pagination
* JSON body fields for creation: `name`, `description`, `tags`, `seeds`

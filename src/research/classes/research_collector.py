class ResearchCollector:
    """Core functions for collecting, storing, and parsing article metadata."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("research/out")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.output_dir / "manifest.json"
        self.articles: List[ArticleMetadata] = []
        self.current_index = 0
        self.load_manifest()

    def load_manifest(self) -> None:
        if not self.manifest_file.exists():
            return
        try:
            data = json.loads(self.manifest_file.read_text(encoding='utf-8'))
            entries = data.get('entries') if isinstance(data, dict) else data
            for item in entries or []:
                item_copy = dict(item)
                item_copy.pop('authors_str', None)
                self.articles.append(ArticleMetadata(**item_copy))
        except Exception:
            # tolerate malformed manifests
            pass

    def save_manifest(self) -> None:
        payload = {"version": 1, "entries": [a.to_dict() for a in self.articles]}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def fetch_html_from_url(self, url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0) -> str:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        r.raise_for_status()
        return r.text

    def load_html_from_file(self, file_path: str) -> str:
        return Path(file_path).read_text(encoding='utf-8')


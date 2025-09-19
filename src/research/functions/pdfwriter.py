import uuid
import shutil
from pathlib import Path
from typing import Optional, Dict
import requests
from pypdf import PdfWriter
from functions.filelogger import _fllog
from functions.globals import headers


def save_pdf(url: str, filename: str, metadata: Dict[str,str], save_path: str, tmp_path: str = "/tmp") -> Optional[str]:
  try:
    temporary_pdf = download_to_temporary_storage(url, Path(tmp_path))
  except Exception as e:
    _fllog("error: "+str(e))
    return None
  _fllog(str(temporary_pdf))
  _fllog('Opening with pdfwriter')
  writer = PdfWriter(clone_from=str(temporary_pdf))
  try:
    _fllog("trying to add metadata")
    writer.add_metadata(metadata)
    _fllog("metadata added")
    dest_path = Path(save_path) / filename
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    _fllog("trying to write to destination filepath")
    with dest_path.open("wb") as fp:
      writer.write(fp)
    return str(dest_path)
  except Exception as e:
    _fllog("Exception:"+str(e))
    return None

def download_to_temporary_storage(url: str, dest: Path) -> Path:
  filename = build_temporary_filename()
  file_path = dest / filename
  dest.mkdir(parents=True, exist_ok=True)
  if not url.startswith("http://") and not url.startswith("https://"):
    # Local file path
    src = Path(url.replace('file://', ''))
    shutil.copyfile(src, file_path)
    return file_path
  # Download via HTTP
  resp = requests.get(url, headers=headers)
  ct = resp.headers.get('Content-Type','')
  if 'pdf' not in ct.lower():
    raise IOError(f"wrong response content type, got: {ct}")
  with file_path.open('wb') as f:
    f.write(resp.content)
  return file_path

def build_temporary_filename() -> str:
  return f"tmp_{uuid.uuid4().hex}.pdf"


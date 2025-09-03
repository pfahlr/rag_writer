import uuid
import shutil
from pathlib import Path
from typing import Optional, Dict
import requests
from pypdf import PdfWriter
from functions.filelogger import _fllog

headers={
  'sec-ch-ua':'"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
  'Content-Type': "application/json; charset=utf-8",
  'Accept-Language': "en-US,en;q=0.9",
  'Accept-Ranges': "bytes",
  'sec-ch-ua-platform': "Linux",
  'sec-fetch-dest': "empty",
  'Priority': "u=4, i",
  'Dnt': "1",
  'sec-ch-ua-mobile': '?0',
  'Accept': "*/*",
  'Accept-Encoding': "gzip, deflate, br, zstd",
  'sec-fetch-mode': "cors",
  'sec-fetch-site': "same-origin",
  'User-Agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
}

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


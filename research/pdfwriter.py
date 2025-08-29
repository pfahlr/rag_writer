import time
from rich.pretty import pprint
from markov_word_generator import MarkovWordGenerator, WordType
from pypdf import PdfWriter
import shutil

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

def save_pdf(url, filename, metadata, save_path, tmp_path="/tmp"):
  temporary_pdf = download_to_temporary_storage(url, tmp_path)
  writer = PdfWriter(clone_from=temporary_pdf)
  writer.add_metadata(metadata)
  with open(f"{save_path}/{filename}", "wb") as fp: 
    writer.write(fp)

def download_to_temporary_storage(url, dest):

  try:
    filename = build_temporary_filename()
    file_path =f"{dest}/{filename}"
    # add handler for manually downloaded pdfs.
    if "file://" in url: 
      url.replace('file://', '') 
      try:
        shutil.move(url, file_path)
      except FileNotFoundError:
        print(f"Error: Source file '{source_file}' not found.")
        return None
      except Exception as e:
        print(f"Move file failed: {e}")      
        return None
    else: 
      response = requests.get(url)
      if response.ok:
        with open(file_path, 'wb') as f: 
          f.write(response.content)
      else:
        print(f"Download failed: {response.status}")
        return None

    return file_path

  except requests.exceptions.RequestException as e:
    print(f"Download Failed: {e}")
    return None

def build_temporary_filename():
  unix_timestamp = time.time()
  pprint(unix_timestamp)

  # Generate a random word in English by predicting the probability of each new character based on its last 4 last characters
  generator = MarkovWordGenerator(
    markov_length=5,
    language='en',
    word_type=WordType.WORD,
  )

  tmp = generator.generate_word()
  pprint(tmp)
  tmpfilename = f"{tmp}"
  pprint(tmpfilename)
  tmpfilename = f"{tmpfilename}_{unix_timestamp}"   
  ppriunt(tmpfilename)

  generator = MarkovWordGenerator(
    markov_length=7,
    language='en',
    word_type=WordType.WORD,
  )

  tmp = generator.generate_word()
  pprint(tmp)
  tmpfilename = f"{tmpfilename}_{tmp}.pdf"
  pprint(tmpfilename)
  return tmpfilename

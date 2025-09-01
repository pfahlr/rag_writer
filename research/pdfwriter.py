import time
from rich.pretty import pprint
from markov_word_generator import MarkovWordGenerator, WordType
from pypdf import PdfWriter
import shutil
import requests
import math
from filelogger import _fllog

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
  _fllog(tmp_path)
  try:
    temporary_pdf = download_to_temporary_storage(url, tmp_path)
  except Exception as e:
    _fllog("error: "+str(e))
    return None
  _fllog(str(temporary_pdf))
  _fllog('Opening with pdfwriter')
  writer = PdfWriter(clone_from=temporary_pdf)
  
  try:
    _fllog("trying to add metadata")
    writer.add_metadata(metadata)
    _fllog("metadata added")
    download_dest_filepath = f"{save_path}/{filename}"
    _fllog("got destination filepath:")
    _fllog(download_dest_filepath)
    _fllog("trying to write to destination filepath")
    with open(download_dest_filepath, "wb") as fp: 
      writer.write(fp)
    return download_dest_filepath

  except Exception as e:
    _fllog("Exception:"+str(e))
    return None

def download_to_temporary_storage(url, dest):
  filename = build_temporary_filename()
  _fllog(filename)
  _fllog("attempt to write this pdf at:")
  file_path =f"{dest}/{filename}"
  _fllog(file_path)
  # add handler for manually downloaded pdfs.
  if "file://" in url: 
    _fllog("got local filepath")
    url = url.replace('file://', '')
    _fllog(url) 
    try:
      shutil.move(url, file_path)
      _fllog('file moved to temp storage successful')
    except FileNotFoundError:
      print(f"Error: Source file '{source_file}' not found.")
      return None
    except Exception as e:
      print(f"Move file failed: {e}")      
      return None
  else: 
    _fllog('gonna try downloading this shit')
    _fllog('from')
    _fllog(url)
    try:
      response = requests.get(url)
    except requests.exceptions.HTTPError as err:
      print(f"HTTP Error occurred: {err}")
      return None
    except requests.exceptions.RequestException as err:
      print(f"Other Request Error occurred: {err}")
      return None
    
    if response.headers['Content-Type'] != "application/pdf":
      _fllog("wrong response content type, got:"+response.headers['Content-Type'])
      raise IOError("wrong response content type, got:"+response.headers['Content-Type'])
    else:
      _fllog("response is ok, writing pdf temporty")
      with open(file_path, 'wb') as f: 
        f.write(response.content)
  _fllog('file moved to temp storage successful')
  return file_path


def build_temporary_filename():
  # Generate a random word in English by predicting the probability of each new character based on its last 4 last characters
  generator = MarkovWordGenerator(
    markov_length=5,
    language='en',
    word_type=WordType.WORD,
  )

  tmp = generator.generate_word()
  _fllog(tmp)
  _fllog('|')
  tmpfilename = f"{tmp}"
  _fllog(tmpfilename)
  

  generator = MarkovWordGenerator(
    markov_length=7,
    language='en',
    word_type=WordType.WORD,
  )

  tmp = generator.generate_word()
  _fllog(tmp)
  _fllog('|')
  tmpfilename = f"{tmpfilename}_{tmp}.pdf"
  _fllog(tmpfilename)
  _fllog('|')
  return tmpfilename

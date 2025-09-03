def _fllog(s: str):
  try:
    with open('/home/rick/Desktop/python_log.log','a+', encoding='utf-8') as f:
      f.write(">"+str(s)+"\n")
      f.flush()
  except Exception:
    pass


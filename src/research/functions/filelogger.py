def _fllog(s: str):
  try:
    with open('/app/logs/debug/collector_ui.log','a+', encoding='utf-8') as f:
      f.write(">"+str(s)+"\n")
      f.flush()
  except Exception:
    pass


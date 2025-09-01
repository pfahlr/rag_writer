def _fllog(str):
  with open('/home/rick/Desktop/python_log.log','a+') as f:
    f.write(">"+str+"\n")
    f.flush()


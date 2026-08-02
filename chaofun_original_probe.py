#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'chaofun_original'
OUT.mkdir(parents=True,exist_ok=True)
S=requests.Session();S.headers.update({'User-Agent':'Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/140.0 Mobile Safari/537.36','Accept':'application/json, text/plain, */*','Referer':'https://choa.fun/p/1100912'})
endpoints={
 'post':('https://choa.fun/api/get_post_info',{'postId':'1100912'}),
 'comments_old':('https://choa.fun/api/v0/list_comments',{'postId':'1100912','pageNum':'1','pageSize':'500','order':'old'}),
 'comments_new':('https://choa.fun/api/v0/list_comments',{'postId':'1100912','pageNum':'1','pageSize':'500','order':'new'}),
 'comments_hot':('https://choa.fun/api/v0/list_comments',{'postId':'1100912','pageNum':'1','pageSize':'500','order':'hot'}),
}
summary={}
for name,(url,params) in endpoints.items():
 try:
  r=S.get(url,params=params,timeout=60)
  print(name,r.status_code,r.url,len(r.content),flush=True)
  text=r.text
  (OUT/f'{name}_{r.status_code}.txt').write_text(text,encoding='utf-8',errors='replace')
  try: data=r.json()
  except Exception: data=None
  summary[name]={'status':r.status_code,'url':r.url,'bytes':len(r.content),'json':data}
 except Exception as e:
  summary[name]={'error':repr(e)};print(name,e,flush=True)
(OUT/'all.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

# Produce a compact flat text extraction for rapid inspection.
texts=[]
def walk(x,path=''):
 if isinstance(x,dict):
  for k,v in x.items():
   np=f'{path}/{k}'
   if isinstance(v,str) and len(v.strip())>0: texts.append({'path':np,'text':v})
   else: walk(v,np)
 elif isinstance(x,list):
  for i,v in enumerate(x):walk(v,f'{path}[{i}]')
for name,payload in summary.items():
 if isinstance(payload,dict) and payload.get('json') is not None: walk(payload['json'],name)
(OUT/'texts.json').write_text(json.dumps(texts,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'texts.txt').open('w',encoding='utf-8') as f:
 for item in texts:f.write(f"{item['path']}\t{item['text']}\n")

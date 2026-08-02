#!/usr/bin/env python3
from __future__ import annotations
import json, re
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'douban_comments'/'chaofun_media'
OUT.mkdir(parents=True,exist_ok=True)
S=requests.Session();S.headers.update({'User-Agent':'Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/140.0 Mobile Safari/537.36','Accept':'application/json, text/plain, */*','Referer':'https://choa.fun/p/1100912'})
api='https://choa.fun/api/v0/list_comments'
r=S.get(api,params={'postId':'1100912','pageNum':'1','pageSize':'500','order':'old'},timeout=60);r.raise_for_status();payload=r.json()
rows=payload.get('data') or []
index=[]
for i,c in enumerate(rows):
 names=c.get('imageNames') or ''
 if isinstance(names,list): names=','.join(str(x) for x in names)
 names=[x.strip() for x in str(names).split(',') if x.strip()]
 files=[]
 for j,name in enumerate(names):
  url=name if name.startswith('http') else 'https://i.chao-fan.com/'+name.lstrip('/')
  ext=Path(url.split('?',1)[0]).suffix or '.jpg'
  fn=f'{i:03d}_{j:02d}_{Path(url.split("?",1)[0]).stem}{ext}'
  try:
   q=S.get(url,timeout=60);q.raise_for_status()
   if len(q.content)<500: raise ValueError(f'small {len(q.content)}')
   (OUT/fn).write_bytes(q.content);files.append({'file':fn,'url':url,'bytes':len(q.content)})
   print(i,j,q.status_code,len(q.content),url,flush=True)
  except Exception as e:
   files.append({'url':url,'error':repr(e)});print('ERR',i,j,url,repr(e),flush=True)
 index.append({'comment_index':i,'comment_id':c.get('id'),'author':(c.get('userInfo') or {}).get('userName'),'text':c.get('text'),'images':files})
(OUT/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'index.txt').open('w',encoding='utf-8') as f:
 for row in index:
  if row['images']:
   f.write(f"[{row['comment_index']}] {row['author']}: {row['text']}\n")
   for im in row['images']:f.write('  '+json.dumps(im,ensure_ascii=False)+'\n')

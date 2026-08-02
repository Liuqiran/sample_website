from __future__ import annotations
import base64, hashlib, hmac, json, urllib.parse
from datetime import datetime, timezone
from pathlib import Path
import requests

TOPIC='491451970'
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'douban_scan_thread'
OUT.mkdir(parents=True,exist_ok=True)
S=requests.Session();S.headers.update({'User-Agent':'api-client/1 com.douban.frodo/7.22.0.beta9(231) Android/23 product/Mate40 platform/mobile','Accept':'application/json'})

def get(path, extra=None):
    url='https://frodo.douban.com/api/v2'+path
    ts=datetime.now(timezone.utc).strftime('%Y%m%d')
    raw='&'.join(['GET',urllib.parse.quote(urllib.parse.urlparse(url).path,safe=''),ts])
    sig=base64.b64encode(hmac.new(b'bf7dddc7c9cfe6f7',raw.encode(),hashlib.sha1).digest()).decode()
    params={'apiKey':'0dad551ec0f84ed02907ff5c42e8ec70','_ts':ts,'_sig':sig};params.update(extra or {})
    r=S.get(url,params=params,timeout=60);r.raise_for_status();return r.json()

detail=get(f'/group/topic/{TOPIC}')
comments=get(f'/group/topic/{TOPIC}/comments',{'start':0,'count':100})
(OUT/'detail.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'comments.json').write_text(json.dumps(comments,ensure_ascii=False,indent=2),encoding='utf-8')
texts=[]
for c in comments.get('comments',[]):
    texts.append({'time':c.get('create_time'),'author':(c.get('author') or {}).get('name'),'text':c.get('text'),'photos':c.get('photos')})
summary={'title':detail.get('title'),'create_time':detail.get('create_time'),'author':(detail.get('author') or {}).get('name'),'abstract':detail.get('abstract'),'content':detail.get('content'),'comments':texts}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))

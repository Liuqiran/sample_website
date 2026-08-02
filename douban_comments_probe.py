from __future__ import annotations
import base64,hashlib,hmac,json,urllib.parse
from datetime import datetime,timezone
from pathlib import Path
import requests
R=Path(__file__).resolve().parent
O=R/'results'/'douban_comments';O.mkdir(parents=True,exist_ok=True)
s=requests.Session();s.headers.update({'User-Agent':'api-client/1 com.douban.frodo/7.22.0.beta9(231) Android/23 product/Mate40 platform/mobile','Accept':'application/json'})
def get(path,extra=None):
 url='https://frodo.douban.com/api/v2'+path;ts=datetime.now(timezone.utc).strftime('%Y%m%d');raw='&'.join(['GET',urllib.parse.quote(urllib.parse.urlparse(url).path,safe=''),ts]);sig=base64.b64encode(hmac.new(b'bf7dddc7c9cfe6f7',raw.encode(),hashlib.sha1).digest()).decode();p={'apiKey':'0dad551ec0f84ed02907ff5c42e8ec70','_ts':ts,'_sig':sig};p.update(extra or {});r=s.get(url,params=p,timeout=45);return r.status_code,r.text
paths=[('/group/topic/312383214',{}),('/group/topic/312383214/comments',{'start':0,'count':100}),('/group/topic/312383214/replies',{'start':0,'count':100}),('/group/topic/312383214/comments',{'start':100,'count':100})]
for i,(p,q) in enumerate(paths):
 st,txt=get(p,q);(O/f'{i}_{st}.txt').write_text(txt,encoding='utf-8');print(i,p,st,len(txt))

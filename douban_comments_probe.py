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
 st,txt=get(p,q);(O/f'{i}_{st}.txt').write_text(txt,encoding='utf-8');print(i,p,st,len(txt),flush=True)

# The original puzzle was first posted on ChaoFun as post 1100912.
# Fetch its public post metadata and every comment order exposed by the app API.
c=requests.Session();c.headers.update({'User-Agent':'Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/140.0 Mobile Safari/537.36','Accept':'application/json, text/plain, */*','Referer':'https://choa.fun/p/1100912'})
endpoints={
 'chaofun_post':('https://choa.fun/api/get_post_info',{'postId':'1100912'}),
 'chaofun_comments_old':('https://choa.fun/api/v0/list_comments',{'postId':'1100912','pageNum':'1','pageSize':'500','order':'old'}),
 'chaofun_comments_new':('https://choa.fun/api/v0/list_comments',{'postId':'1100912','pageNum':'1','pageSize':'500','order':'new'}),
 'chaofun_comments_hot':('https://choa.fun/api/v0/list_comments',{'postId':'1100912','pageNum':'1','pageSize':'500','order':'hot'}),
}
summary={}
for name,(url,params) in endpoints.items():
 try:
  r=c.get(url,params=params,timeout=60)
  text=r.text
  (O/f'{name}_{r.status_code}.txt').write_text(text,encoding='utf-8',errors='replace')
  try:data=r.json()
  except Exception:data=None
  summary[name]={'status':r.status_code,'url':r.url,'bytes':len(r.content),'json':data}
  print(name,r.status_code,len(r.content),flush=True)
 except Exception as e:
  summary[name]={'error':repr(e)};print(name,repr(e),flush=True)
(O/'chaofun_all.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
texts=[]
def walk(x,path=''):
 if isinstance(x,dict):
  for k,v in x.items():
   np=f'{path}/{k}'
   if isinstance(v,str) and v.strip():texts.append({'path':np,'text':v})
   else:walk(v,np)
 elif isinstance(x,list):
  for i,v in enumerate(x):walk(v,f'{path}[{i}]')
for name,payload in summary.items():
 if isinstance(payload,dict) and payload.get('json') is not None:walk(payload['json'],name)
(O/'chaofun_texts.json').write_text(json.dumps(texts,ensure_ascii=False,indent=2),encoding='utf-8')
with (O/'chaofun_texts.txt').open('w',encoding='utf-8') as f:
 for item in texts:f.write(f"{item['path']}\t{item['text']}\n")

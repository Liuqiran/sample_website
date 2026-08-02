from __future__ import annotations
import json,re,urllib.parse
from pathlib import Path
import requests
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'reverse_image'
OUT.mkdir(parents=True,exist_ok=True)
IMG='https://img9.doubanio.com/view/group_topic/l/public/p735803186.webp'
Q=urllib.parse.quote(IMG,safe='')
URLS={
 'google_lens':f'https://lens.google.com/uploadbyurl?url={Q}&hl=en',
 'google_legacy':f'https://www.google.com/searchbyimage?image_url={Q}&client=app&hl=en',
 'bing':f'https://www.bing.com/images/searchbyimage/upload?cbir=sbi&imgurl={Q}&rdr=1&first=1&tsc=ImageBasicHover',
 'yandex':f'https://yandex.com/images/search?rpt=imageview&url={Q}',
 'tineye':f'https://tineye.com/search?url={Q}',
}
S=requests.Session();S.headers.update({'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140.0 Safari/537.36','Accept-Language':'en-US,en;q=0.9'})
summary={}
for name,url in URLS.items():
 try:
  r=S.get(url,timeout=90,allow_redirects=True)
  html=r.text
  (OUT/f'{name}.html').write_text(html,encoding='utf-8',errors='ignore')
  soup=BeautifulSoup(html,'html.parser')
  for x in soup(['script','style','noscript']):x.decompose()
  text=' '.join(soup.get_text(' ',strip=True).split())
  links=[]
  for a in BeautifulSoup(html,'html.parser').find_all('a',href=True):
   h=a['href'];t=' '.join(a.get_text(' ',strip=True).split())
   if h.startswith('/url?q='):h=urllib.parse.parse_qs(urllib.parse.urlsplit(h).query).get('q',[h])[0]
   if h.startswith('http') and h not in [x['url'] for x in links]: links.append({'url':h,'text':t[:300]})
  summary[name]={'status':r.status_code,'final_url':r.url,'title':soup.title.get_text(' ',strip=True) if soup.title else '', 'text':text[:10000],'links':links[:100]}
  print(name,r.status_code,r.url,len(html),summary[name]['title'])
 except Exception as e:
  summary[name]={'error':repr(e)};print(name,e)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

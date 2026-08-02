#!/usr/bin/env python3
from __future__ import annotations
import base64, hashlib, hmac, io, json, urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import requests
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'douban_candidates'
OUT.mkdir(parents=True,exist_ok=True)
TOPIC='312383214'
url=f'https://frodo.douban.com/api/v2/group/topic/{TOPIC}'
ts=datetime.now(timezone.utc).strftime('%Y%m%d')
path=urllib.parse.urlparse(url).path
raw='&'.join(['GET',urllib.parse.quote(path,safe=''),ts])
sig=base64.b64encode(hmac.new(b'bf7dddc7c9cfe6f7',raw.encode(),hashlib.sha1).digest()).decode()
s=requests.Session(); s.headers.update({'User-Agent':'api-client/1 com.douban.frodo/7.22.0.beta9(231) Android/23 product/Mate40 platform/mobile','Accept':'application/json'})
r=s.get(url,params={'apiKey':'0dad551ec0f84ed02907ff5c42e8ec70','_ts':ts,'_sig':sig},timeout=45); r.raise_for_status(); data=r.json()

def walk(x:Any,ctx=''):
    if isinstance(x,dict):
        for k,v in x.items():
            n=f'{ctx}/{k}'
            if isinstance(v,str) and v.startswith('http') and ('doubanio.com' in v or 'douban.com' in v): yield n,v
            yield from walk(v,n)
    elif isinstance(x,list):
        for i,v in enumerate(x): yield from walk(v,f'{ctx}[{i}]')

items=[]; seen=set()
for ctx,u in walk(data):
    if not any(t in ctx.lower() for t in ('image','photo','large','origin','raw','url')): continue
    # Try common Douban size substitutions as separate candidates.
    variants=[u]
    for old,new in [('/s_ratio_poster/','/raw/'),('/m_ratio_poster/','/raw/'),('/l_ratio_poster/','/raw/'),('/view/group_topic/small/public/','/view/group_topic/l/public/'),('/view/group_topic/large/public/','/view/group_topic/l/public/')]:
        if old in u: variants.append(u.replace(old,new))
    for v in variants:
        if v in seen: continue
        seen.add(v)
        try:
            q=s.get(v,timeout=45); q.raise_for_status(); im=Image.open(io.BytesIO(q.content)).convert('RGB')
            if im.width<300 or im.height<300: continue
            idx=len(items)+1
            full=OUT/f'{idx:02d}_full.jpg'; full.write_bytes(q.content)
            thumb=im.copy(); thumb.thumbnail((420,420)); canvas=Image.new('RGB',(440,480),'white'); canvas.paste(thumb,((440-thumb.width)//2,5));
            d=ImageDraw.Draw(canvas); d.multiline_text((8,430),f'#{idx} {im.width}x{im.height}\n{ctx[-75:]}',fill='black')
            canvas.save(OUT/f'{idx:02d}_thumb.jpg',quality=72,optimize=True)
            items.append({'index':idx,'width':im.width,'height':im.height,'context':ctx,'url':v,'full_file':str(full.relative_to(ROOT))})
        except Exception as e:
            pass
(OUT/'metadata.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# Douban image candidates','',f'Found **{len(items)}** downloadable images.','']
for x in items: lines.append(f"- #{x['index']} — {x['width']}×{x['height']} — `{x['context']}` — {x['url']}")
(OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(f'found {len(items)} images')

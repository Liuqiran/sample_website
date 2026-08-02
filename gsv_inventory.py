#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
from streetlevel import streetview

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'gsv_inventory'
OUT.mkdir(parents=True,exist_ok=True)
AREAS={
 'estrella': {'south':19.4680,'north':19.4815,'west':-99.1240,'east':-99.1080},
 'valle_dorado': {'south':19.5450,'north':19.5615,'west':-99.2250,'east':-99.2040},
}
Z=17

def tile_xy(lat,lon,z=Z):
 n=2**z
 x=int((lon+180.0)/360.0*n)
 latr=math.radians(lat)
 y=int((1.0-math.asinh(math.tan(latr))/math.pi)/2.0*n)
 return x,y

summary={}
for area,b in AREAS.items():
 x0,y_s=tile_xy(b['south'],b['west']);x1,y_n=tile_xy(b['north'],b['east'])
 xmin,xmax=sorted((x0,x1));ymin,ymax=sorted((y_n,y_s))
 panos={};tile_counts={};errors=[]
 print(area,'tiles',xmin,xmax,ymin,ymax,flush=True)
 for x in range(xmin,xmax+1):
  for y in range(ymin,ymax+1):
   try:
    rows=streetview.get_coverage_tile(x,y)
    tile_counts[f'{x},{y}']=len(rows)
    for p in rows:
     if b['south']-.001<=p.lat<=b['north']+.001 and b['west']-.001<=p.lon<=b['east']+.001:
      panos[p.id]=p
    print(area,x,y,len(rows),'unique',len(panos),flush=True)
   except Exception as e:
    errors.append(f'{x},{y}: {e!r}');print('ERR',area,x,y,e,flush=True)
 data=[]
 for p in panos.values():
  data.append({'id':p.id,'lat':p.lat,'lon':p.lon,'heading':p.heading,'date':str(p.date) if p.date else None,'source':str(p.source) if p.source else None,'copyright':p.copyright_message})
 data.sort(key=lambda q:(q['lat'],q['lon']))
 (OUT/f'{area}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 summary[area]={'bbox':b,'tiles':tile_counts,'pano_count':len(data),'errors':errors}
 print(area,'TOTAL',len(data),flush=True)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

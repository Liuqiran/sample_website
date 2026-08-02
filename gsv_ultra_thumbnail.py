#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw
from streetlevel import streetview

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'gsv_ultra'
CACHE=ROOT/'.cache'/'gsv_ultra'
TARGET=ROOT/'results'/'douban_candidates'/'21_full.jpg'
CENTER=(19.474720,-99.118565)
RADIUS=700.0
CELL=58.0
YAW_STEP=45
UA='Mozilla/5.0 (compatible; exact-location-thumbnail-scan/1.0)'

@dataclass
class Row:
    pano_id:str; lat:float; lon:float; date:str; yaw:int
    score:float=-1.0; good:int=0; inliers:int=0; ratio:float=0.0; coverage:float=0.0
    crop:str=''; image_path:str=''; error:str=''

def dist(lat,lon):
    dy=(lat-CENTER[0])*111320.0
    dx=(lon-CENTER[1])*111320.0*math.cos(math.radians(CENTER[0]))
    return math.hypot(dx,dy)

def tile_xy(lat,lon,z=17):
    n=2**z;x=int((lon+180)/360*n);lr=math.radians(lat)
    y=int((1-math.asinh(math.tan(lr))/math.pi)/2*n)
    return x,y

def date_str(x):
    if not x:return ''
    y=getattr(x,'year',None);m=getattr(x,'month',None);d=getattr(x,'day',None)
    return f'{y:04d}-{m:02d}'+(f'-{d:02d}' if d else '') if y and m else str(x)

def inventory():
    latd=RADIUS/111320.0;lond=RADIUS/(111320.0*math.cos(math.radians(CENTER[0])))
    b={'south':CENTER[0]-latd,'north':CENTER[0]+latd,'west':CENTER[1]-lond,'east':CENTER[1]+lond}
    x0,ys=tile_xy(b['south'],b['west']);x1,yn=tile_xy(b['north'],b['east'])
    panos={};errors=[]
    for x in range(min(x0,x1),max(x0,x1)+1):
        for y in range(min(yn,ys),max(yn,ys)+1):
            try:
                for p in streetview.get_coverage_tile(x,y):
                    if p.lat is not None and p.lon is not None and dist(float(p.lat),float(p.lon))<=RADIUS:
                        panos[p.id]=p
            except Exception as e:errors.append(f'{x},{y}: {e!r}')
    lat_scale=111320.0;lon_scale=lat_scale*math.cos(math.radians(CENTER[0]))
    cells={}
    for p in panos.values():
        gx=int((p.lon-b['west'])*lon_scale/CELL);gy=int((p.lat-b['south'])*lat_scale/CELL)
        cx=b['west']+(gx+.5)*CELL/lon_scale;cy=b['south']+(gy+.5)*CELL/lat_scale
        dd=math.hypot((p.lon-cx)*lon_scale,(p.lat-cy)*lat_scale)
        if (gx,gy) not in cells or dd<cells[(gx,gy)][0]:cells[(gx,gy)]=(dd,p)
    return list(cells.values()),{'bbox':b,'raw':len(panos),'sampled':len(cells),'errors':errors,'radius_m':RADIUS,'cell_m':CELL}

def target_features():
    img=cv2.imread(str(TARGET))
    if img is None:raise RuntimeError('target missing')
    if max(img.shape[:2])>1200:
        s=1200/max(img.shape[:2]);img=cv2.resize(img,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
    h,w=img.shape[:2]
    crops={'full':img,'upper':img[:int(h*.80),:],'left':img[:int(h*.75),:int(w*.68)],'depth':img[int(h*.08):int(h*.84),int(w*.15):int(w*.86)]}
    sift=cv2.SIFT_create(nfeatures=4200,contrastThreshold=.012,edgeThreshold=14)
    out={}
    for n,c in crops.items():out[n]=(c,*sift.detectAndCompute(cv2.cvtColor(c,cv2.COLOR_BGR2GRAY),None))
    OUT.mkdir(parents=True,exist_ok=True);Image.open(TARGET).save(OUT/'target.jpg')
    return out

def download(row:Row):
    CACHE.mkdir(parents=True,exist_ok=True)
    p=CACHE/f'{row.pano_id}_{row.yaw}.jpg';row.image_path=str(p)
    if p.exists() and p.stat().st_size>3000:return row
    url='https://streetviewpixels-pa.googleapis.com/v1/thumbnail'
    params={'cb_client':'maps_sv.tactile','w':560,'h':460,'pitch':-4,'panoid':row.pano_id,'yaw':row.yaw,'thumbfov':104}
    try:
        r=requests.get(url,params=params,headers={'User-Agent':UA},timeout=35);r.raise_for_status()
        if len(r.content)<3000:raise ValueError(f'small {len(r.content)}')
        p.write_bytes(r.content)
    except Exception as e:row.error=f'download {e!r}'
    return row

def score(row:Row,tfs):
    if row.error:return row
    im=cv2.imread(row.image_path)
    if im is None:row.error='decode';return row
    sift=cv2.SIFT_create(nfeatures=3500,contrastThreshold=.012,edgeThreshold=14)
    kp2,des2=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None)
    if des2 is None:return row
    bf=cv2.BFMatcher(cv2.NORM_L2);best=(-1,0,0,0,0,'')
    for n,(crop,kp1,des1) in tfs.items():
        if des1 is None:continue
        pairs=bf.knnMatch(des1,des2,k=2);good=[m for m,q in pairs if m.distance<.70*q.distance]
        ins=0;ratio=0.;cov=0.
        if len(good)>=6:
            a=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2);b=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            _,mask=cv2.findHomography(a,b,cv2.RANSAC,5.5)
            if mask is not None:
                ok=mask.ravel().astype(bool);ins=int(ok.sum());ratio=ins/len(good)
                if ins>=4:
                    pts=a[ok,0,:];x0,y0=pts.min(0);x1,y1=pts.max(0);cov=float(max(0,(x1-x0)*(y1-y0))/(crop.shape[0]*crop.shape[1]))
        sc=ins*12+min(len(good),80)*.15+min(cov,.5)*30+ratio*2.5
        if sc>best[0]:best=(sc,len(good),ins,ratio,cov,n)
    row.score,row.good,row.inliers,row.ratio,row.coverage,row.crop=best
    return row

def main():
    start=time.time();tfs=target_features();cells,diag=inventory();print('inventory',diag,flush=True)
    rows=[]
    for _,p in cells:
        for yaw in range(0,360,YAW_STEP):rows.append(Row(str(p.id),float(p.lat),float(p.lon),date_str(getattr(p,'date',None)),yaw))
    with ThreadPoolExecutor(max_workers=24) as ex:
        downloaded=[]
        for i,f in enumerate(as_completed([ex.submit(download,r) for r in rows]),1):
            downloaded.append(f.result())
            if i%500==0:print('downloaded',i,'/',len(rows),flush=True)
    ranked=[]
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i,f in enumerate(as_completed([ex.submit(score,r,tfs) for r in downloaded]),1):
            ranked.append(f.result())
            if i%250==0:print('scored',i,'/',len(rows),flush=True)
    ranked=sorted([r for r in ranked if r.score>=0],key=lambda r:r.score,reverse=True)
    diag['views']=len(rows);diag['scored']=len(ranked);diag['elapsed']=time.time()-start
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'diagnostics.json').write_text(json.dumps(diag,ensure_ascii=False,indent=2))
    with (OUT/'ranked.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=Row.__dataclass_fields__.keys());w.writeheader();[w.writerow(asdict(r)) for r in ranked]
    lines=['# Ultra thumbnail scan','',f"- sampled panos: **{diag['sampled']}**",f"- views scored: **{len(ranked)}**",'', '|rank|score|inliers|good|coverage|date|coordinates|yaw|pano|','|---:|---:|---:|---:|---:|---|---|---:|---|']
    cards=[]
    for i,r in enumerate(ranked[:100],1):
        lines.append(f'|{i}|{r.score:.2f}|{r.inliers}|{r.good}|{r.coverage:.3f}|{r.date}|{r.lat:.7f}, {r.lon:.7f}|{r.yaw}|`{r.pano_id}`|')
        if i<=72:
            try:
                im=Image.open(r.image_path).convert('RGB');im.thumbnail((440,410));c=Image.new('RGB',(460,490),'white');c.paste(im,((460-im.width)//2,3));ImageDraw.Draw(c).multiline_text((5,425),f'#{i} score={r.score:.2f} inl={r.inliers} good={r.good} cov={r.coverage:.3f}\n{r.lat:.7f},{r.lon:.7f} yaw={r.yaw} date={r.date or "?"}\n{r.pano_id}',fill='black',spacing=2);c.save(OUT/f'top_{i:03d}_{r.pano_id}_{r.yaw}.jpg',quality=76,optimize=True);cards.append(c)
            except:pass
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    if cards:
        cols=3;sheet=Image.new('RGB',(cols*460,math.ceil(len(cards)/cols)*490),'white')
        for i,c in enumerate(cards):sheet.paste(c,((i%cols)*460,(i//cols)*490))
        sheet.save(OUT/'contact_sheet.jpg',quality=80,optimize=True)
    print('TOP',flush=True)
    for r in ranked[:40]:print(json.dumps(asdict(r),ensure_ascii=False),flush=True)
if __name__=='__main__':main()

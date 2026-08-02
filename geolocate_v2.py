#!/usr/bin/env python3
from __future__ import annotations
import base64, csv, io, json, math, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable
import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'v2'
CACHE=ROOT/'.cache'/'v2'
CFG=json.loads((ROOT/'config.json').read_text())
BBOX=CFG['bbox']
UA='Mozilla/5.0 (compatible; exact-street-geolocator/2.0)'

@dataclass
class Photo:
    id:str; sequence_id:str; lat:float|None; lng:float|None; heading:float|None
    url:str; date:str=''; score:float=-1.; good:int=0; inliers:int=0
    inlier_ratio:float=0.; coverage:float=0.; crop:str=''; path:str=''; error:str=''

def walk(x:Any)->Iterable[dict[str,Any]]:
    if isinstance(x,dict):
        yield x
        for v in x.values(): yield from walk(v)
    elif isinstance(x,list):
        for v in x: yield from walk(v)

def first(d,*keys):
    for k in keys:
        if k in d and d[k] not in (None,''): return d[k]

def fnum(x):
    try:return float(x)
    except:return None

def urlfix(u):
    if not u:return ''
    u=str(u).replace('[[sizeprefix]]','proc')
    if u.startswith('//'):u='https:'+u
    if u.startswith('http://'):u='https://'+u[7:]
    if not u.startswith('http'):u='https://'+u.lstrip('/')
    return u

def parse(payload,default_seq=''):
    photos={}
    keys=('fileurlProc','fileurlLTh','fileurl','procUrl','lth_name','lthName','imageUrl','photoUrl','name','th_name')
    for d in walk(payload):
        u=urlfix(first(d,*keys)); pid=first(d,'id','photoId','photo_id')
        if not u or pid is None or 'photo' not in u.lower():continue
        seq=first(d,'sequenceId','sequence_id','sequence')
        if isinstance(seq,dict):seq=first(seq,'id','sequenceId')
        seq=str(seq or default_seq)
        p=Photo(str(pid),seq,fnum(first(d,'lat','latitude','matchLat')),fnum(first(d,'lng','lon','longitude','matchLng')),fnum(first(d,'heading','gpsHeading','direction')),u,str(first(d,'dateAdded','date','createdAt') or ''))
        photos[p.id]=p
    return list(photos.values())

def get_json(s,url,params,tries=4):
    err=None
    for n in range(tries):
        try:
            r=s.get(url,params=params,timeout=60);r.raise_for_status();return r.json()
        except Exception as e:err=e;time.sleep(2**n)
    raise RuntimeError(err)

def in_bbox(p,margin=.002):
    return p.lat is not None and p.lng is not None and BBOX['south']-margin<=p.lat<=BBOX['north']+margin and BBOX['west']-margin<=p.lng<=BBOX['east']+margin

def collect():
    s=requests.Session();s.headers.update({'User-Agent':UA,'Accept':'application/json'})
    endpoint='https://api.openstreetcam.org/2.0/photo/'
    lats=np.linspace(BBOX['south'],BBOX['north'],7);lngs=np.linspace(BBOX['west'],BBOX['east'],7)
    seeds={};seqs=set(map(str,[47063,713,25524,29410,476998,24341,47451]))
    for i,(lat,lng) in enumerate((a,b) for a in lats for b in lngs):
        try:
            data=get_json(s,endpoint,{'lat':f'{lat:.7f}','lng':f'{lng:.7f}','zoomLevel':17,'join':'sequence','orderBy':'id','orderDirection':'desc'})
            for p in parse(data):
                seeds[p.id]=p
                if p.sequence_id:seqs.add(p.sequence_id)
        except Exception as e:print('seed error',lat,lng,e)
    print('seed photos',len(seeds),'sequences',sorted(seqs))
    allp=dict(seeds)
    for si,seq in enumerate(sorted(seqs),1):
        seen=set();page=1
        while page<=30:
            try:data=get_json(s,endpoint,{'sequenceId':seq,'page':page,'itemsPerPage':500})
            except Exception as e:print('sequence error',seq,page,e);break
            batch=parse(data,seq)
            new=[p for p in batch if p.id not in seen]
            for p in new:seen.add(p.id)
            inside=[p for p in new if in_bbox(p)]
            for p in inside:allp[p.id]=p
            print(f'seq {si}/{len(seqs)} {seq} page {page}: {len(batch)} rows, {len(inside)} inside, total {len(allp)}')
            if not batch or not new or len(batch)<100:break
            page+=1
    return list(allp.values())

def decode_target():
    raw=base64.b64decode((ROOT/'data'/'target_embed.b64').read_text().strip())
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'target.jpg').write_bytes(raw)
    arr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if arr is None:raise RuntimeError('target decode failed')
    return arr

def crops(img):
    h,w=img.shape[:2]
    return {'full':img,'left':img[:int(h*.78),:int(w*.62)],'middle':img[:int(h*.88),int(w*.18):int(w*.82)],'street':img[int(h*.20):,int(w*.30):]}

def feats(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift=cv2.SIFT_create(nfeatures=3500,contrastThreshold=.015,edgeThreshold=12)
    kp,des=sift.detectAndCompute(g,None)
    return kp,des

def target_features(img):return {name:(im,*feats(im)) for name,im in crops(img).items()}

def download(p):
    CACHE.mkdir(parents=True,exist_ok=True);fp=CACHE/f'{p.id}.jpg';p.path=str(fp)
    if fp.exists() and fp.stat().st_size>3000:return p
    try:
        r=requests.get(p.url,headers={'User-Agent':UA},timeout=45);r.raise_for_status()
        if len(r.content)<3000:raise ValueError('small image')
        fp.write_bytes(r.content)
    except Exception as e:p.error='download '+repr(e)
    return p

def match_one(p,tfs):
    if p.error:return p
    im=cv2.imread(p.path)
    if im is None:p.error='decode';return p
    if max(im.shape[:2])>1400:
        sc=1400/max(im.shape[:2]);im=cv2.resize(im,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA)
    kp2,des2=feats(im)
    if des2 is None:return p
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    best=None
    for name,(crop,kp1,des1) in tfs.items():
        if des1 is None:continue
        pairs=matcher.knnMatch(des1,des2,k=2)
        good=[m for m,n in pairs if m.distance<.67*n.distance]
        ins=0;ratio=0.;coverage=0.
        if len(good)>=6:
            a=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            b=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            H,mask=cv2.findHomography(a,b,cv2.RANSAC,5.0)
            if mask is not None:
                ok=mask.ravel().astype(bool);ins=int(ok.sum());ratio=ins/len(good)
                if ins>=4:
                    pts=a[ok,0,:];x0,y0=pts.min(0);x1,y1=pts.max(0)
                    coverage=float(max(0,(x1-x0)*(y1-y0))/(crop.shape[0]*crop.shape[1]))
        score=ins + .08*len(good) + 8*min(coverage,.35) + 2*ratio
        row=(score,len(good),ins,ratio,coverage,name)
        if best is None or row[0]>best[0]:best=row
    if best:
        p.score,p.good,p.inliers,p.inlier_ratio,p.coverage,p.crop=best
    return p

def douban_selfcheck(target):
    def phash(im):
        g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);g=cv2.resize(g,(32,32)).astype(np.float32);d=cv2.dct(g)[:8,:8];m=np.median(d[1:]);return sum(int(x)<<i for i,x in enumerate((d>m).flatten()))
    th=phash(target);rows=[]
    src=ROOT/'results'/'douban_candidates'
    if src.exists():
        for f in sorted(src.glob('*_full.jpg')):
            im=cv2.imread(str(f))
            if im is None:continue
            rows.append(((th^phash(im)).bit_count(),f.name,im.shape[1],im.shape[0]))
    rows.sort();(OUT/'douban_selfcheck.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

def make_outputs(rows):
    OUT.mkdir(parents=True,exist_ok=True)
    rows=sorted(rows,key=lambda p:p.score,reverse=True)
    with (OUT/'ranked.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=asdict(rows[0]).keys() if rows else Photo.__dataclass_fields__.keys());w.writeheader()
        for p in rows:w.writerow(asdict(p))
    md=['# Full-sequence KartaView match','',f'- candidates in bounding box: **{len(rows)}**','', '|rank|score|photo|sequence|coordinates|good|inliers|ratio|coverage|crop|','|---:|---:|---:|---:|---|---:|---:|---:|---:|---|']
    for i,p in enumerate(rows[:50],1):md.append(f'|{i}|{p.score:.3f}|{p.id}|{p.sequence_id}|{p.lat},{p.lng}|{p.good}|{p.inliers}|{p.inlier_ratio:.3f}|{p.coverage:.3f}|{p.crop}|')
    (OUT/'report.md').write_text('\n'.join(md)+'\n')
    for i,p in enumerate(rows[:20],1):
        try:
            im=Image.open(p.path).convert('RGB');im.thumbnail((500,340));c=Image.new('RGB',(520,390),'white');c.paste(im,((520-im.width)//2,3));ImageDraw.Draw(c).multiline_text((5,350),f'#{i} score {p.score:.2f} id {p.id}\n{p.lat},{p.lng} good {p.good} inliers {p.inliers} cov {p.coverage:.2f}',fill='black');c.save(OUT/f'top_{i:02d}_{p.id}.jpg',quality=55,optimize=True)
        except:pass

def main():
    target=decode_target();douban_selfcheck(target);tfs=target_features(target)
    photos=collect();print('bbox candidates',len(photos))
    with ThreadPoolExecutor(max_workers=16) as ex:photos=[f.result() for f in as_completed([ex.submit(download,p) for p in photos])]
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i,f in enumerate(as_completed([ex.submit(match_one,p,tfs) for p in photos]),1):
            rows.append(f.result())
            if i%100==0:print('matched',i)
    make_outputs([p for p in rows if p.score>=0])
if __name__=='__main__':main()

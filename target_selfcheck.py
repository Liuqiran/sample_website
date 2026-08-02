from pathlib import Path
import base64,json
import cv2,numpy as np
R=Path(__file__).resolve().parent
raw=base64.b64decode((R/'data/target_embed.b64').read_text())
t=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
def ph(im):
 g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);g=cv2.resize(g,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32);d=cv2.dct(g)[:8,:8];m=np.median(d[1:]);v=0
 for b in (d>m).flatten():v=(v<<1)|int(b)
 return v
h=ph(t); rows=[]
for f in sorted((R/'results/douban_candidates').glob('*_full.jpg')):
 im=cv2.imread(str(f));
 if im is None:continue
 rows.append({'distance':(h^ph(im)).bit_count(),'file':f.name,'width':im.shape[1],'height':im.shape[0]})
rows.sort(key=lambda x:x['distance'])
out=R/'results'/'target_selfcheck.json';out.write_text(json.dumps(rows,indent=2))
print(rows[:10])

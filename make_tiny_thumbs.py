from pathlib import Path
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parent
src=ROOT/'results'/'douban_candidates'
out=ROOT/'results'/'douban_tiny'
out.mkdir(parents=True,exist_ok=True)
for i in range(1,35):
    p=src/f'{i:02d}_full.jpg'
    if not p.exists(): continue
    im=Image.open(p).convert('RGB')
    im.thumbnail((120,140))
    c=Image.new('RGB',(128,160),'white')
    c.paste(im,((128-im.width)//2,2))
    ImageDraw.Draw(c).text((4,144),f'#{i} {Image.open(p).width}x{Image.open(p).height}',fill='black')
    c.save(out/f'{i:02d}.jpg',quality=28,optimize=True)

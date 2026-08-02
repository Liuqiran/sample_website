from pathlib import Path

path = Path(__file__).with_name('geolocate_v3.py')
source = path.read_text(encoding='utf-8')
source = source.replace(
    'enumerate((a, b) for a in lats for b in lngs, 1)',
    'enumerate(((a, b) for a in lats for b in lngs), 1)',
)
exec(compile(source, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path)})

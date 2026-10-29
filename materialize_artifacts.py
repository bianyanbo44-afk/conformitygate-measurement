from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent
def materialize():
    for rel,record in json.loads((root/'ARTIFACT_PARTS.json').read_text()).items():
        path=root/rel
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']:continue
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as out:
            for part in record['parts']:out.write((root/part).read_bytes())
        assert path.stat().st_size==record['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],rel
    print('All multipart scientific artifacts verified.',flush=True)
if __name__=='__main__':materialize()

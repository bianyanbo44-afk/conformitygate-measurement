from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
from materialize_artifacts import materialize
materialize()
scripts=root/'revision_20261029'/'src'
stages=[('revision_data.py',[]),('revision_engine.py',['--mode','main','--jobs','4']),('revision_engine.py',['--mode','loco','--jobs','2']),('revision_statistics.py',[]),('revision_uncertainty.py',[]),('revision_integrity.py',[]),('revision_legacy_diagnostics.py',[]),('revision_followup.py',[]),('revision_harmonized.py',['--jobs','4']),('revision_figures.py',[]),('followup_figures.py',[])]
for name,args in stages:
    print('Running',name,*args,flush=True)
    subprocess.run([sys.executable,str(scripts/name),*args],cwd=root,check=True)
print('All analysis stages completed.',flush=True)

"""Portable geometry check for this project's single panels and three-panel row."""
import json
from pathlib import Path

def require_matplotlib_panel_alignment(fig,json_out=None,tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True,**kwargs):
    fig.canvas.draw();w,h=fig.get_size_inches()*72
    boxes=[[a.get_position().x0*w,a.get_position().y0*h,a.get_position().width*w,a.get_position().height*h] for a in fig.axes]
    passed=True
    if len(boxes)>1:
        for col in [1,2,3]:passed=passed and max(b[col] for b in boxes)-min(b[col] for b in boxes)<=tolerance_pt
        gaps=[boxes[i+1][0]-boxes[i][0]-boxes[i][2] for i in range(len(boxes)-1)]
        if gaps:passed=passed and max(gaps)-min(gaps)<=gutter_tolerance_pt
    result={'passed':bool(passed),'panels':len(boxes),'boxes_pt':boxes,'tolerance_pt':tolerance_pt}
    if json_out:Path(json_out).write_text(json.dumps(result,indent=2))
    if strict and not passed:raise ValueError('Panel geometry mismatch')
    return result

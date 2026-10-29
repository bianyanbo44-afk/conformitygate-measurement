"""Portable physical alignment gate for this project's equal rectangular grids."""
import json
from pathlib import Path

def require_matplotlib_panel_alignment(fig,json_out=None,tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True,axes=None,**kwargs):
    fig.canvas.draw();w,h=fig.get_size_inches()*72
    selected=list(axes) if axes is not None else [a for a in fig.axes if a.get_visible() and not a.get_label().startswith('<colorbar')]
    boxes=[[a.get_position().x0*w,a.get_position().y0*h,a.get_position().width*w,a.get_position().height*h] for a in selected]
    passed=True;checks=[]
    def groups(direction):
        # Near-centre grouping identifies the row/column before enforcing strict edge tolerance.
        size=2 if direction==0 else 3
        pending=sorted(range(len(boxes)),key=lambda i:boxes[i][direction]+boxes[i][size]/2)
        output=[]
        for i in pending:
            center=boxes[i][direction]+boxes[i][size]/2
            if output:
                first=output[-1][0];other=boxes[first][direction]+boxes[first][size]/2
                if abs(center-other)<.30*min(boxes[i][size],boxes[first][size]):output[-1].append(i);continue
            output.append([i])
        return output
    for dimension in [0,1]:
        for group in groups(dimension):
            if len(group)<2:continue
            # A column shares left/right edges; a row shares bottom/top edges.
            spans=[boxes[i][dimension] for i in group]
            ends=[boxes[i][dimension]+boxes[i][dimension+2] for i in group]
            across=1-dimension
            extents=[boxes[i][across+2] for i in group]
            ok=max(spans)-min(spans)<=tolerance_pt and max(ends)-min(ends)<=tolerance_pt and max(extents)-min(extents)<=tolerance_pt
            ordered=sorted(group,key=lambda i:boxes[i][across])
            gaps=[boxes[b][across]-boxes[a][across]-boxes[a][across+2] for a,b in zip(ordered,ordered[1:])]
            ok=ok and all(g>=-tolerance_pt for g in gaps)
            if len(gaps)>1:ok=ok and max(gaps)-min(gaps)<=gutter_tolerance_pt
            passed=passed and ok;checks.append({'group':group,'direction':'column' if dimension==0 else 'row','passed':bool(ok)})
    result={'passed':bool(passed),'panels':len(boxes),'boxes_pt':boxes,'tolerance_pt':tolerance_pt,'checks':checks,'scope':'Single panels and equal rectangular row/column grids'}
    if json_out:Path(json_out).write_text(json.dumps(result,indent=2))
    if strict and not passed:raise ValueError('Panel geometry mismatch')
    return result

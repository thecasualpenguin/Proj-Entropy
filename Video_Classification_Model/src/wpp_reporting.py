"""Legacy-layout report adapter for binary WPP runs; never re-evaluates a checkpoint."""
from __future__ import annotations
import csv, json, os, hashlib
from pathlib import Path
from typing import Any, Mapping
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from wpp_metrics import video_metrics

LEGACY_METRICS = ("accuracy", "macro_precision", "weighted_precision", "macro_recall", "weighted_recall", "macro_f1", "weighted_f1")
LEGACY_COLUMNS = ("epoch", "train_loss", "validation_loss", *[f"train_{x}" for x in LEGACY_METRICS],
                  *[f"validation_{x}" for x in LEGACY_METRICS], "learning_rate", "duration_seconds")

REPORT_SCHEMA_VERSION = "wpp-legacy-report-v1"

def _sha(path: Path) -> str | None:
    if not path.is_file(): return None
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()

def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n"); os.replace(tmp,path)

def _history(rows: list[Mapping[str,Any]]) -> list[dict[str,Any]]:
    out=[]
    for item in rows:
        val=item.get("validation", {}); train=item.get("train_online", {}); row={key: None for key in LEGACY_COLUMNS}
        row.update(epoch=item.get("epoch"), train_loss=item.get("train_weighted_objective"),
                   validation_loss=val.get("weighted_ce"), learning_rate=item.get("learning_rate"))
        # WPP v1 stored validation predictions/metrics, but not epoch train predictions.
        for metric in LEGACY_METRICS: row[f"train_{metric}"]=train.get(metric)
        per=val.get("per_class", {})
        supports=np.asarray([x.get("support",0) for x in per.values()],dtype=float)
        for metric in LEGACY_METRICS:
            value=val.get(metric)
            if value is None and metric != "accuracy" and per:
                key=metric.split("_",1)[-1] if "_" in metric else metric
                values=np.asarray([x.get(key,0.) for x in per.values()],dtype=float)
                value=float(np.average(values,weights=supports)) if metric.startswith("weighted_") and supports.sum() else float(values.mean())
            row[f"validation_{metric}"]=value
        out.append(row)
    return out

def _best(history: list[Mapping[str,Any]]) -> dict[str,Any]:
    answer={"train":{},"validation":{}}
    for split in answer:
        for metric in LEGACY_METRICS:
            key=f"{split}_{metric}"; values=[(r["epoch"],r[key]) for r in history if r.get(key) is not None]
            if values:
                value=max(v for _,v in values); answer[split][metric]={"value":value,"epoch":min(e for e,v in values if v==value)}
    return {k:v for k,v in answer.items() if v}

def _csv(path: Path, rows: list[Mapping[str,Any]], columns: tuple[str,...]) -> None:
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=columns); w.writeheader(); w.writerows(rows)
    os.replace(tmp,path)

def _curves(out: Path, history: list[Mapping[str,Any]]) -> None:
    for stem, keys, ylabel in (("loss_curves",("train_loss","validation_loss"),"weighted video CE / online objective"),("accuracy_curves",("train_accuracy","validation_accuracy"),"accuracy"),("precision_curves",("train_macro_precision","train_weighted_precision","validation_macro_precision","validation_weighted_precision"),"precision"),("recall_curves",("train_macro_recall","train_weighted_recall","validation_macro_recall","validation_weighted_recall"),"recall"),("f1_curves",("train_macro_f1","train_weighted_f1","validation_macro_f1","validation_weighted_f1"),"F1")):
        plt.figure(); plotted=False
        for key in keys:
            xy=[(x["epoch"],x[key]) for x in history if x.get(key) is not None]
            if xy: plt.plot(*zip(*xy),label=key); plotted=True
        if plotted: plt.xlabel("epoch");plt.ylabel(ylabel);plt.legend();plt.tight_layout();plt.savefig(out/f"{stem}.png",dpi=140)
        plt.close()

def regenerate(run: str|Path) -> Path:
    run=Path(run); out=run/"evaluations"; out.mkdir(exist_ok=True)
    raw_history=json.loads((run/"history.json").read_text()); history=_history(raw_history)
    _dump(out/"per_epoch_metrics.json",history); _csv(out/"per_epoch_metrics.csv",history,LEGACY_COLUMNS); _curves(out,history)
    canonical=json.loads((run/"test_once.json").read_text()); videos=canonical["per_video"]
    labels=[v["label"] for v in videos]; probabilities=[v["probabilities"] for v in videos]
    report=video_metrics(labels,probabilities)
    names=("real","aigc"); pred=np.asarray(probabilities).argmax(1); matrix=confusion_matrix(labels,pred,labels=[0,1])
    per_class=[{"class":name,"precision":report["per_class"][name]["precision"],"recall":report["per_class"][name]["recall"],"f1":report["per_class"][name]["f1"],"sample_count":report["per_class"][name]["support"]} for name in names]
    _dump(out/"test_per_class_metrics.json",per_class); _csv(out/"test_per_class_metrics.csv",per_class,("class","precision","recall","f1","sample_count"))
    for normalized, filename in ((False,"test_confusion_matrix_raw.csv"),(True,"test_confusion_matrix_normalized.csv")):
        value=matrix.astype(float) if normalized else matrix
        if normalized: value=np.divide(value,value.sum(1,keepdims=True),out=np.zeros_like(value),where=value.sum(1,keepdims=True)!=0)
        with (out/filename).open("w",newline="") as f:
            w=csv.writer(f);w.writerow(["true\\pred",*names]);[w.writerow([names[i],*row.tolist()]) for i,row in enumerate(value)]
        fig,ax=plt.subplots(); image=ax.imshow(value,cmap="Blues");fig.colorbar(image,ax=ax);ax.set(xticks=range(2),yticks=range(2),xticklabels=names,yticklabels=names,xlabel="Predicted",ylabel="True",title="Confusion matrix")
        for i in range(2):
            for j in range(2): ax.text(j,i,format(value[i,j],'.2f' if normalized else '.0f'),ha='center',va='center')
        fig.tight_layout();fig.savefig(out/filename.replace('.csv','.png'),dpi=150);plt.close(fig)
    status=json.loads((run/'status.json').read_text()) if (run/'status.json').is_file() else {}
    selected=status.get('best',{}); selected_epoch=selected.get('epoch'); selected_raw=next((x for x in raw_history if x.get('epoch')==selected_epoch),{})
    selected_validation=selected_raw.get('validation',{})
    if selected_epoch is not None and not selected_raw: raise ValueError("selected checkpoint epoch is absent from saved history")
    if selected.get('macro_f1') is not None and selected_validation.get('macro_f1') is not None and abs(float(selected['macro_f1'])-float(selected_validation['macro_f1'])) > 1e-12:
        raise ValueError("selected checkpoint/status macro-F1 differs from saved selected epoch")
    provenance={"schema_version":REPORT_SCHEMA_VERSION,"report_generator":"src/wpp_reporting.py","report_generator_sha256":_sha(Path(__file__)),
        "input_sha256":{"canonical_test_once":_sha(run/'test_once.json'),"history":_sha(run/'history.json'),"config":_sha(run/'config.yaml'),"status":_sha(run/'status.json'),"best_checkpoint":_sha(run/'checkpoints'/'best.pt'),"latest_checkpoint":_sha(run/'checkpoints'/'latest.pt')},
        "selected_checkpoint":{"path":str(run/'checkpoints'/'best.pt'),"sha256":_sha(run/'checkpoints'/'best.pt'),"epoch":selected_epoch,"validation_metrics":selected_validation,"selection_criterion":"validation macro-F1","tie_break":"lower validation weighted CE"},
        "loss_definition":"weighted video cross-entropy / online training objective","test_weighted_ce":canonical.get('weighted_ce')}
    summary={metric:report[metric] for metric in LEGACY_METRICS}; summary.update({"test_loss":canonical.get("weighted_ce"),"test_weighted_ce":canonical.get("weighted_ce"),"checkpoint":str(run/"checkpoints"/"best.pt"),"best_observed_per_epoch_metrics":_best(history),"class_coverage":{"configured_classes":list(names),"present_classes":[names[x] for x in sorted(set(labels))],"absent_classes":[],"class_distribution":{"real":labels.count(0),"aigc":labels.count(1)},"unexpected_classes":[]},"wpp_reporting_metadata":provenance})
    _dump(out/"summarized_metrics.json",summary)
    _dump(run/"wpp_diagnostics"/"reporting_limitations.json",{"canonical_test_source":"test_once.json","test_re_evaluated":False,"unavailable_history_fields":[f"train_{m}" for m in LEGACY_METRICS],"duration_seconds":"WPP history stored wall-clock timestamp, not per-epoch duration; omitted as null.","loss_definition":"weighted video cross-entropy / online training objective","selection_criterion":"validation macro-F1; lower validation weighted CE is tie-break only (not validation-loss selection).","report_provenance":provenance})
    return out

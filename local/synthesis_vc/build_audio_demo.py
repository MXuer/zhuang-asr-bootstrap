from pathlib import Path
import csv
import json
import os


ROOT = Path.cwd()
OUT = ROOT / "outputs/audio-demo"
OLD = ROOT / "outputs/openvoice-v2-20voices"
NEW = ROOT / "outputs/openvoice-v2-20voices-improved"


def rel(path):
    return os.path.relpath(Path(path).resolve(), OUT.resolve())


def copy_index(src_dir):
    speakers = {}
    with (src_dir / "selected_speakers.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            speakers[r["speaker_id"]] = r.get("demo_reference_wav") or r.get("reference_wav")
    rows = []
    with (src_dir / "manifest.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return speakers, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old_speakers, old_rows = copy_index(OLD)
    new_speakers, new_rows = copy_index(NEW)
    new_by_key = {(r["speaker_id"], Path(r["source_wav"]).name): r for r in new_rows if Path(r["converted_wav"]).exists()}

    items = []
    for r in old_rows:
        src = ROOT / r["source_wav"]
        old = ROOT / r["converted_wav"]
        key = (r["speaker_id"], src.name)
        new = ROOT / new_by_key[key]["converted_wav"] if key in new_by_key else None
        if not src.exists() or not old.exists():
            continue
        ref = ROOT / (new_speakers.get(r["speaker_id"]) or old_speakers[r["speaker_id"]])
        items.append({
            "label": f'{r.get("source_id") or src.stem} / {r["speaker_id"]}',
            "speaker": r["speaker_id"],
            "text": r.get("original_text") or "",
            "source": rel(src),
            "reference": rel(ref),
            "converted": rel(old),
            "improved": rel(new) if new and new.exists() else "",
        })

    (OUT / "app.json").write_text(json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")
    (OUT / "index.html").write_text(HTML, encoding="utf-8")
    print(OUT / "index.html", len(items))


HTML = r"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>壮语 TTS 音色转换对比</title>
<style>
body{margin:0;font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f6f7f9;color:#14171f}
main{max-width:1180px;margin:0 auto;padding:24px}
header{display:flex;gap:12px;align-items:center;justify-content:space-between;margin-bottom:16px}
h1{font-size:22px;margin:0}
select{width:min(760px,100%);padding:8px;border:1px solid #cfd5df;border-radius:6px;background:white}
.text{white-space:pre-wrap;background:white;border:1px solid #e1e5eb;border-radius:6px;padding:12px;max-height:170px;overflow:auto}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
.full{grid-column:1/-1}
.panel{background:white;border:1px solid #e1e5eb;border-radius:6px;padding:12px}
.panel h2{font-size:15px;margin:0 0 8px}
audio{width:100%;height:36px}
canvas{width:100%;height:90px;background:#fbfcfe;border:1px solid #edf0f5;border-radius:4px;margin-top:8px}
.muted{color:#647084}
@media(max-width:760px){.grid{grid-template-columns:1fr}header{display:block}select{margin-top:10px}}
</style>
<main>
  <header><h1>壮语 TTS 音色转换对比</h1><select id="pick"></select></header>
  <section class="text" id="text"></section>
  <section class="grid">
    <div class="panel full"><h2>mms-tts-zyb 合成语音</h2><audio controls id="source"></audio><canvas id="sourceWave"></canvas></div>
    <div class="panel"><h2>参考语音</h2><audio controls id="reference"></audio><canvas id="referenceWave"></canvas></div>
    <div class="panel"><h2>OpenVoiceV2 转换后</h2><audio controls id="converted"></audio><canvas id="convertedWave"></canvas></div>
    <div class="panel full"><h2>缓解失真版 <span class="muted">3参考 + 12s分段 + tau=0.2</span></h2><audio controls id="improved"></audio><canvas id="improvedWave"></canvas></div>
  </section>
</main>
<script>
let data;
const ids=["source","reference","converted","improved"];
fetch("app.json").then(r=>r.json()).then(j=>{data=j.items; init();});
function init(){
  const pick=document.getElementById("pick");
  data.forEach((x,i)=>{const o=document.createElement("option");o.value=i;o.textContent=x.label;pick.appendChild(o);});
  pick.onchange=()=>show(data[pick.value]);
  show(data[0]);
}
function show(x){
  document.getElementById("text").textContent=x.text || "manifest 中无原文";
  ids.forEach(id=>{
    const a=document.getElementById(id);
    a.src=x[id]||"";
    a.parentElement.style.display=x[id]?"block":"none";
    if(x[id]) drawWave(x[id], document.getElementById(id+"Wave"));
  });
}
async function drawWave(url, canvas){
  const ctx=canvas.getContext("2d"), dpr=devicePixelRatio||1;
  canvas.width=canvas.clientWidth*dpr; canvas.height=canvas.clientHeight*dpr;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const buf=await fetch(url).then(r=>r.arrayBuffer());
  const ac=new AudioContext();
  const audio=await ac.decodeAudioData(buf);
  await ac.close();
  const ch=audio.getChannelData(0), w=canvas.width, h=canvas.height, mid=h/2, step=Math.ceil(ch.length/w);
  ctx.strokeStyle="#2b6cb0"; ctx.beginPath();
  for(let x=0;x<w;x++){
    let min=1,max=-1;
    for(let i=x*step;i<Math.min((x+1)*step,ch.length);i++){const v=ch[i]; if(v<min)min=v; if(v>max)max=v;}
    ctx.moveTo(x, mid+min*mid*.9); ctx.lineTo(x, mid+max*mid*.9);
  }
  ctx.stroke();
}
</script>
</html>
"""


if __name__ == "__main__":
    main()

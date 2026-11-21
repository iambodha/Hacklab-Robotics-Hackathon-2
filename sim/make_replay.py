"""Build side.mp4 + replay.html for a recorded sim session (see recorder.py).

  python3 sim/make_replay.py <session_dir>     # default: newest under sim/sessions

replay.html is self-contained next to side.mp4 and looks/: the side-camera video, joint
traces (measured + commanded) with a cursor, joint bars, and the tool-call log with the
images the agent saw. Click a log entry or the chart to jump there.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(session):
    session = Path(session)
    meta = json.loads((session / 'session.json').read_text())
    fps = meta['fps']
    states = [json.loads(l) for l in (session / 'state.jsonl').read_text().splitlines() if l.strip()]
    events = [json.loads(l) for l in (session / 'events.jsonl').read_text().splitlines() if l.strip()]
    frames = sorted((session / 'frames').glob('*.jpg'))
    ffmpeg = shutil.which('ffmpeg')
    if frames and ffmpeg:
        tmp = session / 'side.tmp.mp4'  # rename when complete: a killed run never leaves a broken side.mp4
        subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', str(session / 'frames/%06d.jpg'),
                        '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-crf', '23',
                        '-movflags', '+faststart', str(tmp)], check=True)
        tmp.replace(session / 'side.mp4')
    data = {'meta': meta, 'fps': fps,
            'states': [[s['t'], s['norm'], s['goal'], s['load'], s['status']] for s in states],
            'events': [{k: e.get(k, e.get(k.replace('_end', ''))) for k in
                        ('n', 't', 'frame', 'frame_end', 'tool', 'args', 'result', 'looks', 'truth')} for e in events]}
    html = REPLAY_HTML.replace('/*DATA*/null', json.dumps(data, separators=(',', ':')))
    (session / 'replay.html').write_text(html, encoding='utf-8')
    return session / 'replay.html'


REPLAY_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>ElRobot sim — replay</title>
<style>
body{margin:0;font:13px/1.4 system-ui,Segoe UI,sans-serif;background:#111418;color:#e6e6e6}
header{padding:8px 14px;background:#1a1f26;display:flex;gap:14px;align-items:center}
header b{font-size:15px} header span{color:#8a96a3} button{background:#2a313b;color:#e6e6e6;border:0;border-radius:4px;padding:4px 9px;cursor:pointer} button.on{background:#3d8bfd}
main{display:grid;grid-template-columns:minmax(520px,1.25fr) 1fr;gap:12px;padding:12px;height:calc(100vh - 62px);box-sizing:border-box}
.left{display:flex;flex-direction:column;gap:10px;min-height:0;overflow:auto}
video{width:100%;border-radius:6px;background:#000}
.card{background:#1a1f26;border-radius:6px;padding:10px}
canvas{width:100%;height:230px;display:block;cursor:crosshair}
.joint{display:grid;grid-template-columns:28px 1fr 52px 46px;gap:8px;align-items:center;margin:2px 0}
.bar{position:relative;height:12px;background:#2a313b;border-radius:3px}
.bar .v{position:absolute;top:0;bottom:0;left:0;border-radius:3px}
.bar .g{position:absolute;top:-3px;bottom:-3px;width:2px;background:#fff}
.num{font-variant-numeric:tabular-nums;text-align:right;color:#cfd6dd} .hot{color:#ff6b6b}
#saw img{width:49%;border-radius:4px} #saw .cap{color:#8a96a3;margin-bottom:4px}
#log{overflow:auto;min-height:0}
.ev{border-bottom:1px solid #2a313b;padding:6px 4px;cursor:pointer;opacity:.55}
.ev.past{opacity:1} .ev.cur{background:#223047;opacity:1}
.ev .h{display:flex;gap:8px;overflow-wrap:anywhere} .ev .t{color:#8a96a3;width:58px;flex:none} .ev .tool{color:#7fd1ff;font-weight:600}
.ev pre{margin:3px 0 0;white-space:pre-wrap;color:#b8c2cc;font-size:11px} .err{color:#ff6b6b!important}
.ev img{height:60px;margin:3px 3px 0 0;border-radius:3px}
.truth{color:#8fd18f;font-size:11px}
</style></head><body>
<header><b>ElRobot simulation — replay</b><span id="info"></span>
<span style="margin-left:auto">speed</span><span id="speeds"></span><label><input type="checkbox" id="skip" checked> skip idle</label></header>
<main><div class="left">
<video id="vid" src="side.mp4" controls muted></video>
<div class="card"><canvas id="chart"></canvas><div id="legend" style="color:#8a96a3;font-size:11px;margin-top:4px"></div></div>
<div class="card" id="bars"></div>
<div class="card" id="saw"></div>
</div><div class="card" id="log"></div></main>
<script>
const fa=a=>JSON.stringify(a,(k,v)=>typeof v==='number'?Math.round(v*1000)/1000:v).replace(/"/g,'').replace(/,/g,', ');const fr=r=>{const f=(r||'').split('\n')[0];return f.startsWith('joint ')||f.startsWith('camera ')?'':f};
const D=/*DATA*/null, fps=D.fps, S=D.states, E=D.events;
const NJ=S.length?S[0][1].length:8;
const C=['#3d8bfd','#ff9f43','#2ed573','#ff6b81','#a29bfe','#00d2d3','#feca57','#c8d6e5'].slice(0,NJ);
const vid=document.getElementById('vid'), cv=document.getElementById('chart'), ctx=cv.getContext('2d');
const T=S.length?S[S.length-1][0]:0;
document.getElementById('info').textContent=`${D.meta.started} · ${E.length} tool calls · ${T.toFixed(0)} s · ${E.filter(e=>e.tool==='move_to'||e.tool==='gripper').length} motions`;
document.getElementById('legend').innerHTML=C.map((c,i)=>`<span style="color:${c}">■ j${i+1}</span>`).join(' &nbsp;')+' &nbsp; solid = measured, dashed = commanded · ticks = tool calls';
[1,2,4,8,16].forEach(r=>{const b=document.createElement('button');b.textContent=r+'×';b.onclick=()=>{vid.playbackRate=r;[...document.querySelectorAll('#speeds button')].forEach(x=>x.classList.toggle('on',x===b))};document.getElementById('speeds').append(b);if(r===1)b.classList.add('on')});
const frameAt=t=>Math.max(0,Math.min(S.length-1,Math.round(t*fps)));
function draw(){const w=cv.width=cv.clientWidth*devicePixelRatio,h=cv.height=cv.clientHeight*devicePixelRatio;ctx.clearRect(0,0,w,h);
 const x=i=>i/Math.max(1,S.length-1)*w, y=v=>h-6-(h-12)*v;
 ctx.strokeStyle='#2a313b';ctx.lineWidth=1;[0,.5,1].forEach(v=>{ctx.beginPath();ctx.moveTo(0,y(v));ctx.lineTo(w,y(v));ctx.stroke()});
 E.forEach(e=>{ctx.strokeStyle=e.tool==='look'?'#3a4552':'#56606b';ctx.beginPath();ctx.moveTo(x(e.frame),0);ctx.lineTo(x(e.frame),10*devicePixelRatio);ctx.stroke()});
 for(let j=0;j<NJ;j++){for(const k of [2,1]){ctx.strokeStyle=C[j];ctx.lineWidth=(k===1?1.6:1)*devicePixelRatio;ctx.setLineDash(k===2?[4,4]:[]);ctx.beginPath();
  const step=Math.max(1,Math.floor(S.length/w));for(let i=0;i<S.length;i+=step){const v=S[i][k][j];i?ctx.lineTo(x(i),y(v)):ctx.moveTo(x(i),y(v))}ctx.stroke()}}
 ctx.setLineDash([]);const f=frameAt(vid.currentTime);ctx.strokeStyle='#fff';ctx.lineWidth=1.5*devicePixelRatio;ctx.beginPath();ctx.moveTo(x(f),0);ctx.lineTo(x(f),h);ctx.stroke()}
cv.onclick=ev=>{const r=cv.getBoundingClientRect();vid.currentTime=(ev.clientX-r.left)/r.width*(S.length-1)/fps};
const log=document.getElementById('log');
E.forEach(e=>{const d=document.createElement('div');d.className='ev';const first=fr(e.result);
 const tr=e.truth&&e.truth.objects?Object.entries(e.truth.objects).map(([k,v])=>`${k}: ${v.in_bin?'IN '+v.in_bin:'on table'}`).join(' · '):'';
 d.innerHTML=`<div class="h"><span class="t">${e.t.toFixed(1)}s</span><span class="tool">${e.tool}</span><span>${Object.keys(e.args).length?fa(e.args):''}</span></div>`+
 (first?`<pre class="${/^(ABORTED|ERROR)/.test(first)?'err':''}">${first.replace(/</g,'&lt;')}</pre>`:'')+e.looks.map(l=>`<img src="looks/${l}">`).join('')+(tr?`<div class="truth">ground truth: ${tr}</div>`:'');
 d.onclick=()=>{vid.currentTime=e.frame/fps};e.el=d;log.append(d)});
function update(){const f=frameAt(vid.currentTime), s=S[f];if(!s)return;
 document.getElementById('bars').innerHTML=s[1].map((v,i)=>`<div class="joint"><b style="color:${C[i]}">j${i+1}</b><div class="bar"><div class="v" style="width:${(v*100).toFixed(1)}%;background:${C[i]}"></div><div class="g" style="left:${(s[2][i]*100).toFixed(1)}%"></div></div><span class="num">${v.toFixed(3)}</span><span class="num ${s[3][i]>.9||s[4][i]?'hot':''}">${s[4][i]?'0x'+s[4][i].toString(16):Math.round(s[3][i]*100)+'%'}</span></div>`).join('')+`<div style="color:#8a96a3;font-size:11px">t=${s[0].toFixed(1)}s · bar = measured, white tick = commanded goal, % = servo load</div>`;
 let cur=null;E.forEach(e=>{const past=e.frame<=f;e.el.classList.toggle('past',past);e.el.classList.remove('cur');if(past)cur=e});
 if(cur){cur.el.classList.add('cur');if(!vid.paused)cur.el.scrollIntoView({block:'nearest'})}
 const lk=[...E].reverse().find(e=>e.frame<=f&&e.looks.length);
 document.getElementById('saw').innerHTML=lk?`<div class="cap">what the agent saw at ${lk.t.toFixed(1)}s (wrist · overview)</div>`+lk.looks.map(l=>`<img src="looks/${l}">`).join(' '):'<div class="cap">the agent has not looked yet</div>';
 if(document.getElementById('skip').checked&&!vid.paused){const next=E.find(e=>e.frame>f);
  const busy=E.some(e=>e.frame<=f&&f<=e.frame_end+1.5*fps)||(next&&next.frame-f<1.5*fps);
  if(!busy&&next)vid.currentTime=Math.max(0,(next.frame-fps)/fps)}
 draw()}
vid.addEventListener('timeupdate',update);vid.addEventListener('seeked',update);window.addEventListener('resize',draw);
(function loop(){if(!vid.paused)update();requestAnimationFrame(loop)})();update();
</script></body></html>"""


if __name__ == '__main__':
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        sessions = sorted((HERE / 'sessions').glob('*/session.json'))
        if not sessions: raise SystemExit('no sessions under sim/sessions; pass a session directory')
        target = sessions[-1].parent
    print(build(target))

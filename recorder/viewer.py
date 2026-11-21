"""Self-contained replay viewer page served at /recordings/<id>/viewer (plain HTML/JS, no build step).

It pulls everything from the API relative to its own URL: ./  (session + summary), ./state, ./index,
./events, ./frames/<cam>/<n>.jpg. Layout and colours follow sim/make_replay.py.
"""

VIEWER_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>recording viewer</title>
<style>
:root{color-scheme:dark}
body{margin:0;font:13px/1.4 system-ui,-apple-system,Segoe UI,sans-serif;background:#111418;color:#e6e6e6}
header{padding:8px 14px;background:#1a1f26;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header b{font-size:15px} .dim{color:#8a96a3}
button,select{background:#2a313b;color:#e6e6e6;border:0;border-radius:4px;padding:4px 9px;cursor:pointer;font:inherit}
button.on{background:#3d8bfd} button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid #7fd1ff;outline-offset:1px}
main{display:grid;grid-template-columns:minmax(560px,1.5fr) minmax(320px,1fr);gap:12px;padding:12px;height:calc(100vh - 110px);box-sizing:border-box}
.left,.right{display:flex;flex-direction:column;gap:10px;min-height:0;overflow:auto}
.card{background:#1a1f26;border-radius:6px;padding:10px}
.cams{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.cams figure{margin:0} .cams img{width:100%;aspect-ratio:299/224;object-fit:contain;background:#000;border-radius:4px;display:block}
.cams figcaption{color:#8a96a3;font-size:11px;margin-top:3px;font-variant-numeric:tabular-nums}
canvas{width:100%;display:block;cursor:crosshair} canvas.pos{height:200px} canvas.small{height:90px}
.ctitle{color:#8a96a3;font-size:11px;margin:2px 0}
#scrub{padding:6px 14px 0;background:#161a20;position:relative}
#scrub input{width:100%;margin:0}
#ticks{position:relative;height:12px;margin:0 7px}
#ticks span{position:absolute;top:0;width:2px;height:10px;background:#feca57;cursor:pointer}
#ticks span.life{background:#56606b}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
td,th{padding:2px 5px;text-align:right;border-bottom:1px solid #232a33} th{color:#8a96a3;font-weight:500}
td:first-child,th:first-child{text-align:left}
.hot{color:#ff6b6b} .ok{color:#8fd18f}
.gbar{height:14px;background:#2a313b;border-radius:3px;position:relative;margin-top:4px}
.gbar div{position:absolute;inset:0 auto 0 0;background:#2ed573;border-radius:3px}
#events{font-size:12px}
.ev{display:flex;gap:8px;padding:3px 2px;border-bottom:1px solid #232a33;cursor:pointer;opacity:.55}
.ev.past{opacity:1} .ev.cur{background:#223047}
.ev .t{color:#8a96a3;width:54px;flex:none;font-variant-numeric:tabular-nums} .ev .k{font-weight:600;color:#7fd1ff}
.ev.marker .k{color:#feca57}
.legend span{margin-right:8px}
</style></head><body>
<header><b id="title">recording</b><span class="dim" id="info"></span>
<span style="margin-left:auto" class="dim">bus</span><select id="bus" aria-label="bus"></select>
<button id="play" aria-label="play or pause">&#9654; play</button>
<span class="dim">speed</span><span id="speeds"></span>
<span id="clock" class="dim" style="font-variant-numeric:tabular-nums;min-width:120px;text-align:right"></span></header>
<div id="scrub"><input id="pos" type="range" min="0" max="1000" step="1" value="0" aria-label="time"><div id="ticks"></div></div>
<main><div class="left">
<div class="card cams" id="cams"></div>
<div class="card"><div class="ctitle">position (ticks 0..4095) &mdash; solid measured, dashed goal; yellow lines = markers</div><canvas class="pos" id="c_pos"></canvas>
<div class="ctitle">load (&plusmn;0.1 %)</div><canvas class="small" id="c_load"></canvas>
<div class="ctitle">current (raw, &asymp;6.5 mA/unit)</div><canvas class="small" id="c_cur"></canvas>
<div class="legend" id="legend" style="font-size:11px;margin-top:4px"></div></div>
</div><div class="right">
<div class="card"><div class="ctitle">gripper opening (j8, 0 closed .. 1 open)</div><div id="grip"></div></div>
<div class="card"><table id="joints"></table></div>
<div class="card"><div class="ctitle">events &amp; markers (click to jump)</div><div id="events"></div></div>
</div></main>
<script>
const C=['#3d8bfd','#ff9f43','#2ed573','#ff6b81','#a29bfe','#00d2d3','#feca57','#c8d6e5'];
const J=['j1','j2','j3','j4','j5','j6','j7','j8'];
const $=id=>document.getElementById(id);
let S=[],T=[],E=[],IDX={},META={},DUR=1,cur=0,playing=false,speed=1,bus=null,last=performance.now();
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function bs(a,t){let lo=0,hi=a.length;while(lo<hi){const m=(lo+hi)>>1;if(a[m]<=t)lo=m+1;else hi=m}return lo}
function nearest(a,t){if(!a.length)return -1;const i=bs(a,t);if(i<=0)return 0;if(i>=a.length)return a.length-1;return a[i]-t<t-a[i-1]?i:i-1}
function prev(a,t){return a.length?Math.max(0,bs(a,t)-1):-1}
async function j(u){const r=await fetch(u);if(!r.ok)throw new Error(u+' '+r.status);return r.json()}
async function load(){
 const s=await j('./');META=s.session;DUR=Math.max(0.001,(s.summary&&s.summary.duration)||META.duration||0);
 const every=Math.max(0.02,DUR/6000);
 const [st,ix,ev]=await Promise.all([j('./state?every='+every.toFixed(3)),j('./index'),j('./events')]);
 S=st.rows;T=S.map(r=>r.t);IDX=ix.cameras;E=ev.events;
 if(T.length)DUR=Math.max(DUR,T[T.length-1]);
 $('title').textContent=META.name||META.id;
 const sm=s.summary||{};
 $('info').textContent=`${META.started_iso||''} · ${DUR.toFixed(1)} s · ${sm.state_rows||S.length} state rows @ ${sm.state_hz||'?'} Hz · ${Object.keys(IDX).length} cams · ${(E.filter(e=>e.type==='marker')).length} markers${META.note?' · '+META.note:''}${META.stopped?'':' · RECORDING'}`;
 const buses=[...new Set(S.flatMap(r=>Object.keys(r.buses||{})))];
 $('bus').innerHTML=buses.map(b=>{const r=S.find(x=>x.buses&&x.buses[b]);const role=r&&r.buses[b].role;return `<option value="${esc(b)}">${esc(b)}${role?' ('+role+')':''}</option>`}).join('');
 bus=buses.find(b=>{const r=S.find(x=>x.buses&&x.buses[b]);return r&&r.buses[b].role==='follower'})||buses[0];$('bus').value=bus||'';
 $('cams').innerHTML=Object.entries(IDX).map(([h,c])=>`<figure><img id="img_${h}" alt="${esc(c.role||h)} camera"><figcaption id="cap_${h}">${esc(c.role||h)}</figcaption></figure>`).join('')||'<div class="dim">no camera frames in this recording</div>';
 $('legend').innerHTML=J.map((n,i)=>`<span style="color:${C[i]}">&#9632; ${n}</span>`).join('');
 $('ticks').innerHTML=E.filter(e=>e.type!=='start'&&e.type!=='stop').map((e,i)=>`<span class="${e.type==='marker'?'':'life'}" style="left:${(e.t/DUR*100).toFixed(2)}%" title="${esc(e.t.toFixed(2)+'s '+(e.label||e.type))}" data-t="${e.t}"></span>`).join('');
 $('ticks').onclick=ev=>{if(ev.target.dataset.t)seek(+ev.target.dataset.t)};
 $('events').innerHTML=E.map((e,i)=>`<div class="ev ${e.type==='marker'?'marker':''}" data-i="${i}"><span class="t">${e.t.toFixed(2)}s</span><span class="k">${esc(e.type==='marker'?e.label:e.type)}</span><span class="dim">${esc(evText(e))}</span></div>`).join('');
 $('events').onclick=ev=>{const d=ev.target.closest('.ev');if(d)seek(E[+d.dataset.i].t)};
 draw();update();
}
function evText(e){const o=Object.assign({},e);['t','wall','type','label'].forEach(k=>delete o[k]);const s=JSON.stringify(o);return s==='{}'?'':s.replace(/[{}"]/g,'').slice(0,160)}
function jv(r,n){return r&&r.buses&&r.buses[bus]?r.buses[bus][n]:null}
function series(cv,key,lo,hi){
 const ctx=cv.getContext('2d'),w=cv.width=cv.clientWidth*devicePixelRatio,h=cv.height=cv.clientHeight*devicePixelRatio;
 ctx.clearRect(0,0,w,h);const x=t=>t/DUR*w,y=v=>h-4-(h-8)*(v-lo)/(hi-lo||1);
 ctx.strokeStyle='#2a313b';ctx.lineWidth=1;[lo,(lo+hi)/2,hi].forEach(v=>{ctx.beginPath();ctx.moveTo(0,y(v));ctx.lineTo(w,y(v));ctx.stroke()});
 E.forEach(e=>{if(e.type!=='marker')return;ctx.strokeStyle='#feca5788';ctx.beginPath();ctx.moveTo(x(e.t),0);ctx.lineTo(x(e.t),h);ctx.stroke()});
 const step=Math.max(1,Math.floor(S.length/w));
 for(let k=0;k<J.length;k++)for(const f of key){ctx.strokeStyle=C[k];ctx.lineWidth=(f.dash?1:1.5)*devicePixelRatio;ctx.setLineDash(f.dash?[4,4]:[]);ctx.beginPath();let pen=false;
  for(let i=0;i<S.length;i+=step){const m=jv(S[i],J[k]);const v=m?m[f.k]:null;if(v==null){pen=false;continue}const yy=y(Math.max(lo,Math.min(hi,v)));pen?ctx.lineTo(x(T[i]),yy):ctx.moveTo(x(T[i]),yy);pen=true}ctx.stroke()}
 ctx.setLineDash([]);
 cv._base=ctx.getImageData(0,0,w,h);
}
function draw(){series($('c_pos'),[{k:'goal',dash:1},{k:'pos'}],0,4095);series($('c_load'),[{k:'load'}],-1000,1000);series($('c_cur'),[{k:'current'}],-500,500);cursor()}
function cursor(){for(const id of ['c_pos','c_load','c_cur']){const cv=$(id),ctx=cv.getContext('2d');if(!cv._base)continue;ctx.putImageData(cv._base,0,0);const xx=cur/DUR*cv.width;ctx.strokeStyle='#fff';ctx.lineWidth=1.5*devicePixelRatio;ctx.beginPath();ctx.moveTo(xx,0);ctx.lineTo(xx,cv.height);ctx.stroke()}}
['c_pos','c_load','c_cur'].forEach(id=>$(id).addEventListener('click',ev=>{const r=ev.target.getBoundingClientRect();seek((ev.clientX-r.left)/r.width*DUR)}));
function seek(t){cur=Math.max(0,Math.min(DUR,t));update()}
const imgBusy={};
function update(){
 $('pos').value=Math.round(cur/DUR*1000);$('clock').textContent=`${cur.toFixed(2)} / ${DUR.toFixed(2)} s`;
 for(const [h,c] of Object.entries(IDX)){const i=nearest(c.t,cur);if(i<0)continue;const n=c.n[i],img=$('img_'+h);
  if(img&&img.dataset.n!=String(n)&&!imgBusy[h]){imgBusy[h]=1;img.onload=img.onerror=()=>{imgBusy[h]=0};img.dataset.n=n;img.src=`./frames/${h}/${n}.jpg`;
   $('cap_'+h).textContent=`${c.role||h} · ${h} · frame ${n} · t=${c.t[i].toFixed(2)}s${c.lat&&c.lat[i]!=null?' · latency '+(c.lat[i]*1000).toFixed(0)+' ms':''}`}}
 const r=S[prev(T,cur)];
 const b=r&&r.buses?r.buses[bus]:null;
 if(b){const g=b.gripper;$('grip').innerHTML=g==null?'<span class="dim">uncalibrated (range_min == range_max) or j8 missing</span>':`<div class="gbar"><div style="width:${(g*100).toFixed(1)}%"></div></div><div style="font-variant-numeric:tabular-nums">${g.toFixed(3)}</div>`;
  $('joints').innerHTML='<tr><th>joint</th><th>pos</th><th>goal</th><th>norm</th><th>range</th><th>load</th><th>cur</th><th>°C</th><th>V</th><th>trq</th><th>st</th></tr>'+J.map((n,i)=>{const m=b[n];if(!m)return `<tr><td style="color:${C[i]}">${n}</td><td colspan="10" class="dim">missing</td></tr>`;
   return `<tr title="${esc(m.error||'')}"><td style="color:${C[i]}">${n}</td><td>${m.pos??''}</td><td>${m.goal??''}</td><td>${m.norm==null?'&mdash;':m.norm.toFixed(3)}</td><td class="dim">${m.range_min}-${m.range_max}${m.frozen?'':'*'}</td><td class="${Math.abs(m.load||0)>800?'hot':''}">${m.load??''}</td><td>${m.current??''}</td><td>${m.temp??''}</td><td>${m.volt??''}</td><td class="${m.torque?'ok':'dim'}">${m.torque?'on':'off'}</td><td class="${m.status||m.error?'hot':''}">${m.error?'ERR':(m.status?'0x'+m.status.toString(16):'ok')}</td></tr>`}).join('')+'<tr><td colspan="11" class="dim">* = range not frozen · hover a row for its error text</td></tr>'}
 else{$('grip').innerHTML='<span class="dim">no data for this bus at this time</span>';$('joints').innerHTML=''}
 let ci=-1;E.forEach((e,i)=>{if(e.t<=cur)ci=i});
 document.querySelectorAll('#events .ev').forEach((d,i)=>{d.classList.toggle('past',i<=ci);d.classList.toggle('cur',i===ci)});
 cursor();
}
$('pos').addEventListener('input',ev=>seek(ev.target.value/1000*DUR));
$('bus').addEventListener('change',ev=>{bus=ev.target.value;draw();update()});
function setPlay(p){playing=p;if(playing&&cur>=DUR)cur=0;$('play').innerHTML=playing?'&#10074;&#10074; pause':'&#9654; play';last=performance.now()}
$('play').onclick=()=>setPlay(!playing);
[0.25,0.5,1,2,4].forEach(r=>{const b=document.createElement('button');b.textContent=r+'×';b.onclick=()=>{speed=r;document.querySelectorAll('#speeds button').forEach(x=>x.classList.toggle('on',x===b))};if(r===1)b.classList.add('on');$('speeds').append(b)});
document.addEventListener('keydown',ev=>{if(ev.target.tagName==='INPUT'&&ev.target.type!=='range')return;if(ev.code==='Space'){ev.preventDefault();setPlay(!playing)}else if(ev.code==='ArrowRight')seek(cur+(ev.shiftKey?1:0.1));else if(ev.code==='ArrowLeft')seek(cur-(ev.shiftKey?1:0.1))});
(function loop(now){const dt=(now-last)/1000;last=now;if(playing){cur+=dt*speed;if(cur>=DUR){cur=DUR;setPlay(false)}update()}requestAnimationFrame(loop)})(performance.now());
window.addEventListener('resize',draw);
load().catch(e=>{document.body.insertAdjacentHTML('afterbegin',`<div style="background:#5a1f24;padding:8px">failed to load recording: ${esc(e.message)}</div>`)});
</script></body></html>"""

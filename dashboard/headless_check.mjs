// Headless check that the page renders and updates live (Node >= 22, Google Chrome).
//   node dashboard/headless_check.mjs http://127.0.0.1:8921 /path/to/learn_dir
// Loads the page, reads the attempt count, appends a new episode_begin to episodes.jsonl, and waits for the
// page (without reloading) to show it. Exits non-zero on failure. Writes a screenshot to /tmp/r3dash-check.png.
import { spawn } from 'node:child_process';
import { appendFileSync, writeFileSync, readFileSync } from 'node:fs';

const [url = 'http://127.0.0.1:8920', learnDir] = process.argv.slice(2);
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, ['--headless=new', '--disable-gpu', `--remote-debugging-port=${port}`, '--window-size=1440,1800',
  `--user-data-dir=/tmp/r3dash-chrome-${port}`, 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const fail = m => { console.error('FAIL:', m); chrome.kill(); process.exit(1); };

let target;
for (let i = 0; i < 150 && !target; i++) {
  await sleep(200);
  try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find(t => t.type === 'page'); } catch { }
}
if (!target) fail('chrome did not start');
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r));
let id = 0; const pending = new Map(); const errors = [];
ws.addEventListener('message', e => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
});
const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async expr => (await send('Runtime.evaluate', { expression: expr, returnByValue: true })).result?.result?.value;

await send('Runtime.enable');
await send('Page.enable');
await send('Page.navigate', { url });
await sleep(4000);
const q = `({att: document.querySelectorAll('.att').length, cur: document.querySelector('.cur-n')?.textContent || '',
  charts: document.querySelectorAll('#ch-rot rect').length, calls: document.querySelectorAll('#calls .call').length,
  joints: document.querySelector('#j7 .pos-v')?.textContent, doc: document.querySelector('#doc')?.textContent.slice(0, 40)})`;
const before = await evaluate(q);
console.log('before', JSON.stringify(before));
if (errors.length) fail('page errors: ' + errors.join(' | '));
if (learnDir) {
  const ep = `${learnDir}/episodes.jsonl`;
  const lines = readFileSync(ep, 'utf8').split('\n').filter(l => { try { JSON.parse(l); return true; } catch { return false; } });
  const n = Math.max(0, ...lines.map(l => JSON.parse(l).attempt || 0)) + 1;
  writeFileSync(ep, lines.join('\n') + '\n');
  appendFileSync(ep, JSON.stringify({ type: 'episode_begin', attempt: n, t: Date.now() / 1000, params: { strokes: 8, grip_close: 0.9 },
    hypothesis: 'headless check: live update', based_on: [n - 1], recording_id: null }) + '\n');
  appendFileSync(ep, JSON.stringify({ type: 'phase', attempt: n, t: Date.now() / 1000, phase: 'twist', status: 'ok', measurements: {}, note: 'turning' }) + '\n');
  let after;
  for (let i = 0; i < 20; i++) { await sleep(500); after = await evaluate(q); if (after.cur === `#${n}`) break; }
  console.log('after ', JSON.stringify(after));
  if (after.cur !== `#${n}` || after.att !== before.att + 1) fail(`page did not update to attempt ${n}`);
}
const shot = await send('Page.captureScreenshot', { format: 'png' });
writeFileSync('/tmp/r3dash-check.png', Buffer.from(shot.result.data, 'base64'));
if (errors.length) fail('page errors: ' + errors.join(' | '));
console.log('OK');
ws.close(); chrome.kill(); process.exit(0);

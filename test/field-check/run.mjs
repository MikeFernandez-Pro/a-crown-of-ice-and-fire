// Opens the field check in headless Chrome (WebGPU on) and prints what it logs.
//   npm run dev                      (in another terminal)
//   node test/field-check/run.mjs [--model <name>] [--url https://localhost:5190/]
// playwright-core comes from PLAYWRIGHT_DIR (default ~/.cache/crown-recorder), as record.mjs.
import { createRequire } from 'node:module';
import { homedir } from 'node:os';
import path from 'node:path';

const toolDir = process.env.PLAYWRIGHT_DIR ?? path.join(homedir(), '.cache', 'crown-recorder');
const { chromium } = createRequire(path.join(toolDir, 'package.json'))('playwright-core');

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i < 0 ? fallback : args.splice(i, 2)[1];
};
const base = flag('--url', 'https://localhost:5190/');
const model = flag('--model', 'crown');

const browser = await chromium.launch({
  channel: 'chrome',
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist'],
});
const context = await browser.newContext({ ignoreHTTPSErrors: true });
const page = await context.newPage();
page.on('console', (m) => console.log(`[page:${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
await page.goto(`${base}test/field-check/?model=${model}`);
await page.waitForFunction(() => document.getElementById('out').textContent.endsWith('done'), null, { timeout: 120000 }).catch((e) => console.log(`timed out: ${e.message}`));
await browser.close();

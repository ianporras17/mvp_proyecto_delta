// Disposable real PocketBase + production web. No fixtures in the operator's DB.
const { execFileSync, spawnSync } = require('node:child_process');
const { randomBytes } = require('node:crypto');
const path = require('node:path');

const suffix = randomBytes(5).toString('hex');
const containers = [];
const pbName = `delta-auth-check-${suffix}`, webName = `delta-web-check-${suffix}`;
// 5173 is an existing permitted development origin for all three MPC parties.
const pbUrl = 'http://127.0.0.1:18090', webUrl = 'http://127.0.0.1:5173';
const secret = () => `${randomBytes(24).toString('base64url')}aA1!`;
const docker = (...args) => execFileSync('docker', args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

async function ready(url) {
  for (let i = 0; i < 60; i++) {
    try { if ((await fetch(url)).ok) return; } catch {}
    await pause(500);
  }
  throw new Error(`El servicio no respondió: ${url}`);
}

async function json(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${new URL(url).pathname}: HTTP ${response.status}`);
  return response.json();
}

async function main() {
  const superEmail = `bootstrap.${suffix}@example.test`, superPassword = secret();
  const adminEmail = `admin.${suffix}@example.test`, adminPassword = secret();
  const userEmail = `user.${suffix}@example.test`, userPassword = secret();
  try {
    containers.push(pbName);
    docker('run', '-d', '--name', pbName, '-p', '127.0.0.1:18090:8090', process.env.E2E_POCKETBASE_IMAGE ?? 'mvp_proyecto_delta-pocketbase:latest');
    await ready(`${pbUrl}/api/health`);
    docker('exec', pbName, '/pb/pocketbase', 'superuser', 'upsert', superEmail, superPassword);
    const auth = await json(`${pbUrl}/api/collections/_superusers/auth-with-password`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ identity: superEmail, password: superPassword }),
    });
    for (const [email, password, role] of [[adminEmail, adminPassword, 'admin'], [userEmail, userPassword, 'user']]) {
      await json(`${pbUrl}/api/collections/users/records`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: auth.token },
        body: JSON.stringify({ name: `Prueba ${role}`, email, password, passwordConfirm: password, role }),
      });
    }
    // Persisted SQLite users must survive a real container restart.
    docker('restart', pbName);
    await ready(`${pbUrl}/api/health`);
    containers.push(webName);
    docker('run', '-d', '--name', webName, '-p', '127.0.0.1:5173:80', '-e', `DELTA_POCKETBASE_URL=${pbUrl}`, process.env.E2E_WEB_IMAGE ?? 'mvp-proyecto-delta-web:local');
    await ready(webUrl);
    const runtime = await (await fetch(`${webUrl}/config.js`)).text();
    if (!runtime.includes(pbUrl)) throw new Error('Falta la URL PocketBase en la configuración Docker');
    console.log('PocketBase temporal: migración, cuentas, reinicio y URL en runtime preparados.');
    const env = { ...process.env, PLAYWRIGHT_BASE_URL: webUrl, E2E_POCKETBASE_URL: pbUrl,
      E2E_ADMIN_EMAIL: adminEmail, E2E_ADMIN_PASSWORD: adminPassword, E2E_USER_EMAIL: userEmail, E2E_USER_PASSWORD: userPassword };
    const tests = ['frontend.spec.ts', 'pocketbase.spec.ts'];
    if (process.argv.includes('--with-mpc')) {
      if (!env.E2E_VIDEO_PATH) throw new Error('--with-mpc requiere E2E_VIDEO_PATH (MP4 corto) y el Compose MPC saludable.');
      env.E2E_VIDEO_PATH = path.resolve(env.E2E_VIDEO_PATH);
      env.RUN_WEB_MPC_E2E = '1';
      tests.push('t49-real.spec.ts');
    }
    const args = ['playwright', 'test', ...tests, '--reporter=list'];
    const command = process.platform === 'win32' ? process.env.ComSpec ?? 'cmd.exe' : 'npx';
    const commandArgs = process.platform === 'win32' ? ['/d', '/s', '/c', `npx.cmd ${args.join(' ')}`] : args;
    const result = spawnSync(command, commandArgs, { cwd: path.resolve(__dirname, '..'), env, stdio: 'inherit' });
    if (result.error) throw result.error;
    process.exitCode = result.status ?? 1;
  } finally {
    for (const name of containers.reverse()) {
      try { docker('rm', '--force', '--volumes', name); }
      catch { console.error(`No se pudo retirar el contenedor de prueba ${name}.`); process.exitCode = 1; }
    }
    console.log('Contenedores y volúmenes temporales de autenticación retirados.');
  }
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });

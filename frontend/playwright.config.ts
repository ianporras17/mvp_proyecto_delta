import { defineConfig, devices } from '@playwright/test';

const baseURL = process.env.PLAYWRIGHT_BASE_URL ?? 'http://127.0.0.1:5173';
const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const executablePath = process.env.PLAYWRIGHT_BROWSER_PATH;

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  expect: { timeout: 5_000 },
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: process.env.PLAYWRIGHT_BROWSER_NAME ?? 'chrome',
      use: { ...devices['Desktop Chrome'], channel: executablePath ? undefined : 'chrome',
        ...(executablePath ? { launchOptions: { executablePath } } : {}) },
    },
  ],
  webServer: process.env.PLAYWRIGHT_BASE_URL
    ? undefined
    : {
        command: `${npmCommand} run dev -- --port 5173 --strictPort`,
        url: baseURL,
        reuseExistingServer: !process.env.CI,
        timeout: 30_000,
      },
});

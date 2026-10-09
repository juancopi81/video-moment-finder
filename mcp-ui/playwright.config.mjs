import {defineConfig} from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  fullyParallel: true,
  workers: 2,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:6285',
    viewport: {width: 1440, height: 1050},
    ...(process.env.VMF_BROWSER_EXECUTABLE ? {launchOptions: {executablePath: process.env.VMF_BROWSER_EXECUTABLE}} : {}),
  },
  webServer: {command: 'node tests/serve.mjs', url: 'http://127.0.0.1:6285', reuseExistingServer: !process.env.CI},
});

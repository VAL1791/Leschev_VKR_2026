const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

const ROOT = path.resolve(__dirname, '..');
const OUTPUT_DIR = path.join(ROOT, 'docs', 'media', 'screens');
const BASE_URL = process.env.DASHBOARD_URL || 'http://localhost:8779';
const CHROME_PATH = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

async function waitForStreamlit(page) {
  await page.waitForLoadState('domcontentloaded');
  // Streamlit finishes rendering widgets after the shell is visible, so a short
  // additional wait keeps the captures stable and avoids half-painted controls.
  await page.waitForTimeout(4500);
}

async function selectOption(page, label, optionText) {
  const control = page.getByLabel(label);
  if (await control.count()) {
    await control.click();
    await page.waitForTimeout(300);
    await page.getByRole('option', { name: optionText, exact: true }).click();
    await page.waitForTimeout(1200);
  }
}

async function capture(page, relativePath, configure = async () => {}) {
  await configure();
  await page.waitForTimeout(600);
  await page.screenshot({
    path: path.join(OUTPUT_DIR, relativePath),
    fullPage: false,
  });
}

async function main() {
  ensureDir(OUTPUT_DIR);

  const browser = await chromium.launch({
    executablePath: CHROME_PATH,
    headless: true,
  });

  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
    deviceScaleFactor: 1,
  });
  const page = await context.newPage();

  try {
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await waitForStreamlit(page);
    await capture(page, '01_home.png');

    await page.goto(`${BASE_URL}/Data_Audit`, { waitUntil: 'domcontentloaded' });
    await waitForStreamlit(page);
    await capture(page, '02_data_audit_google_local.png', async () => {
      await selectOption(page, 'Dataset', 'Google Local South Carolina');
    });

    await page.goto(`${BASE_URL}/Model_Benchmark`, { waitUntil: 'domcontentloaded' });
    await waitForStreamlit(page);
    await capture(page, '03_model_benchmark_google_local.png', async () => {
      await selectOption(page, 'Dataset', 'Google Local South Carolina');
    });

    await page.goto(`${BASE_URL}/Robustness_and_Business`, { waitUntil: 'domcontentloaded' });
    await waitForStreamlit(page);
    await capture(page, '04_robustness_google_local.png', async () => {
      await selectOption(page, 'Dataset', 'Google Local South Carolina');
    });

    await page.goto(`${BASE_URL}/Recommendation_Sandbox`, { waitUntil: 'domcontentloaded' });
    await waitForStreamlit(page);
    await capture(page, '05_sandbox_google_local.png', async () => {
      await selectOption(page, 'Dataset', 'Google Local South Carolina');
    });

    await page.goto(`${BASE_URL}/Supervisor_Brief`, { waitUntil: 'domcontentloaded' });
    await waitForStreamlit(page);
    await capture(page, '06_supervisor_brief.png');

    console.log(`Saved dashboard tour screens to ${OUTPUT_DIR}`);
  } finally {
    await context.close();
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('fs')
const os = require('os')
const path = require('path')
const { loadRepoEnv } = require('../../js/sense/loadEnv')

test('loadRepoEnv reads WANDB_API_KEY from .env', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'fly-env-'))
  fs.writeFileSync(path.join(dir, '.env'), 'WANDB_API_KEY=test-key-from-dotenv\n')
  delete process.env.WANDB_API_KEY
  const loaded = loadRepoEnv(dir)
  assert.equal(loaded, path.join(dir, '.env'))
  assert.equal(process.env.WANDB_API_KEY, 'test-key-from-dotenv')
})

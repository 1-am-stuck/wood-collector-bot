const { test } = require('node:test')
const assert = require('node:assert/strict')
const { hopDistance, withTimeout } = require('../../js/sense/actions')

test('hopDistance stays long in open ground and shrinks next to a log', () => {
  assert.equal(hopDistance(10, 40), 10)
  assert.equal(hopDistance(10, null), 10)
  assert.ok(hopDistance(10, 0.4) < 4)
  assert.ok(hopDistance(10, 0.4) >= 2)
})

test('withTimeout returns even if the other promise never settles', async () => {
  const hung = new Promise(() => {})
  const t0 = Date.now()
  await withTimeout(hung, 30)
  assert.ok(Date.now() - t0 < 200)
})

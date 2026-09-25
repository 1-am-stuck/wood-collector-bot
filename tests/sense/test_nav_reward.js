const test = require('node:test')
const assert = require('node:assert')
const { navReward, newVisitSet, cellKey, weights } = require('../../js/sense/nav_reward')

const at = (x, y, z, extra = {}) => ({ x, y, z, ...extra })

test('the first observation registers the start cell without paying for it', () => {
  const seen = newVisitSet()
  const first = navReward(null, at(0, 64, 0), seen)
  assert.equal(first.reward, 0)
  assert.equal(seen.size, 1)
  // Standing still on the start cell must not earn the new-ground bonus.
  const again = navReward(at(0, 64, 0), at(0.01, 64, 0), seen)
  assert.ok(again.reward < 0, 'staying put should cost, not pay')
})

test('new ground pays, revisited ground does not', () => {
  const seen = newVisitSet()
  navReward(null, at(0, 64, 0), seen)
  const fresh = navReward(at(0, 64, 0), at(0, 64, 1.2), seen)
  assert.ok(fresh.newCell)
  assert.ok(fresh.reward > weights.NEW_CELL * 0.9)
  const back = navReward(at(0, 64, 1.2), at(0, 64, 0.4), seen)
  assert.equal(back.newCell, false)
  assert.ok(back.reward < fresh.reward)
})

test('pacing between two cells earns far less than walking a line', () => {
  // This is the property the reward exists for: distance alone would score these the
  // same, and a policy that paces would never learn to travel.
  const pacing = newVisitSet()
  navReward(null, at(0, 64, 0), pacing)
  let paced = 0
  let prev = at(0, 64, 0)
  for (let i = 0; i < 10; i++) {
    const next = at(0, 64, i % 2 === 0 ? 1.2 : 0.2)
    paced += navReward(prev, next, pacing).reward
    prev = next
  }

  const line = newVisitSet()
  navReward(null, at(0, 64, 0), line)
  let walked = 0
  prev = at(0, 64, 0)
  for (let i = 1; i <= 10; i++) {
    const next = at(0, 64, i * 1.1)
    walked += navReward(prev, next, line).reward
    prev = next
  }
  assert.ok(walked > paced * 3, `walking ${walked} should beat pacing ${paced}`)
})

test('spinning on the spot earns nothing at all', () => {
  // Rotation does not change position, so there is no displacement to pay for. The
  // policy cannot farm reward by turning, which is how earlier runs got stuck.
  const seen = newVisitSet()
  navReward(null, at(10, 64, 10), seen)
  let total = 0
  for (let i = 0; i < 20; i++) total += navReward(at(10, 64, 10), at(10, 64, 10), seen).reward
  assert.ok(total < 0, `spinning should be a net cost, got ${total}`)
})

test('walking into a wall costs more than walking in the open', () => {
  const open = newVisitSet()
  navReward(null, at(0, 64, 0), open)
  const clear = navReward(at(0, 64, 0), at(0, 64, 1.2), open)

  const walled = newVisitSet()
  navReward(null, at(0, 64, 0), walled)
  const bumped = navReward(at(0, 64, 0), at(0, 64, 1.2, { touchHead: true, touchLegs: true }), walled)

  assert.ok(bumped.reward < clear.reward)
  assert.ok(Math.abs((clear.reward - bumped.reward) - 2 * -weights.CONTACT) < 1e-9)
})

test('standing on the ground is not a collision', () => {
  const seen = newVisitSet()
  navReward(null, at(0, 64, 0), seen)
  const walking = navReward(at(0, 64, 0), at(0, 64, 1.2, { onGround: true }), seen)
  const flying = newVisitSet()
  navReward(null, at(0, 64, 0), flying)
  const airborne = navReward(at(0, 64, 0), at(0, 64, 1.2), flying)
  assert.equal(walking.reward, airborne.reward)
})

test('getting hurt outweighs the ground it uncovers', () => {
  // If damage only cancelled the new-cell bonus the fly would be indifferent to
  // walking into lava, so it has to be strictly worse than the best a tick can pay.
  assert.ok(Math.abs(weights.DAMAGE) > weights.NEW_CELL)
  const seen = newVisitSet()
  navReward(null, at(0, 64, 0), seen)
  const hurt = navReward(at(0, 64, 0), at(0, 64, 1.2, { damage: 1 }), seen)
  assert.ok(hurt.newCell, 'this tick did find new ground')
  assert.ok(hurt.reward < 0, `still a net loss, got ${hurt.reward}`)
})

test('death ends the episode and costs more than any single hit', () => {
  const seen = newVisitSet()
  navReward(null, at(0, 64, 0), seen)
  const hurt = navReward(at(0, 64, 0), at(0, 64, 1.2, { damage: 1 }), seen)
  const died = navReward(at(0, 64, 1.2), at(0, 64, 2.4, { dead: true }), seen)
  assert.ok(died.done)
  assert.ok(died.reward < hurt.reward)
})

test('stalling costs much less than new ground pays', () => {
  // Otherwise a bump or two makes standing still the rational policy.
  assert.ok(Math.abs(weights.STALL) < weights.NEW_CELL / 10)
  assert.ok(Math.abs(weights.CONTACT) < weights.NEW_CELL / 10)
})

test('cells are block-sized and include height, so climbing counts', () => {
  assert.equal(cellKey({ x: 0.2, y: 64.9, z: -0.1 }), '0,64,-1')
  assert.notEqual(cellKey({ x: 0, y: 64, z: 0 }), cellKey({ x: 0, y: 65, z: 0 }))
})

test('a reset forgets the ground already covered', () => {
  const first = newVisitSet()
  navReward(null, at(0, 64, 0), first)
  navReward(at(0, 64, 0), at(0, 64, 1.2), first)
  const second = newVisitSet()
  navReward(null, at(0, 64, 0), second)
  const again = navReward(at(0, 64, 0), at(0, 64, 1.2), second)
  assert.ok(again.newCell, 'a new episode must be able to earn the same ground again')
})

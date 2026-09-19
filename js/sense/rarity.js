const WOOD_TYPES = [
  'oak', 'spruce', 'birch', 'jungle', 'acacia',
  'dark_oak', 'mangrove', 'cherry', 'pale_oak',
]
const WOOD_LOGS = WOOD_TYPES.map(t => `${t}_log`)

function stripNs (name) {
  return String(name || '').replace(/^minecraft:/, '')
}

function blockInExplored (block, visited, radius) {
  if (!visited || visited.length === 0) return false
  const r2 = radius * radius
  for (const [vx, vz] of visited) {
    const dx = block.x - vx
    const dz = block.z - vz
    if (dx * dx + dz * dz <= r2) return true
  }
  return false
}

function censusLogs (blocks, visited = null, radius = 8) {
  const counts = {}
  for (const b of blocks || []) {
    const name = stripNs(b.name)
    if (!WOOD_LOGS.includes(name)) continue
    if (visited && !blockInExplored(b, visited, radius)) continue
    counts[name] = (counts[name] || 0) + 1
  }
  return counts
}

function rarestLog (counts) {
  const seen = Object.entries(counts || {}).filter(([, n]) => n > 0)
  if (seen.length === 0) return null
  seen.sort((a, b) => a[1] - b[1] || (a[0] < b[0] ? -1 : 1))
  return seen[0][0]
}

module.exports = { WOOD_TYPES, WOOD_LOGS, censusLogs, rarestLog, stripNs }

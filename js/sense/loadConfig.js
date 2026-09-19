const fs = require('fs')
const path = require('path')

const ROOT = path.join(__dirname, '..', '..')

function readJson (rel) {
  return JSON.parse(fs.readFileSync(path.join(ROOT, rel), 'utf8'))
}

function loadSenseConfig (root = ROOT) {
  const join = (...p) => path.join(root, ...p)
  return {
    root,
    retina: JSON.parse(fs.readFileSync(join('configs/sense/retina.json'), 'utf8')),
    blockOdor: JSON.parse(fs.readFileSync(join('configs/sense/block_odor.json'), 'utf8')),
    itemOdor: JSON.parse(fs.readFileSync(join('configs/sense/item_odor.json'), 'utf8')),
    entityOdor: JSON.parse(fs.readFileSync(join('configs/sense/entity_odor.json'), 'utf8')),
    taste: JSON.parse(fs.readFileSync(join('configs/sense/taste_table.json'), 'utf8')),
    mechano: JSON.parse(fs.readFileSync(join('configs/sense/mechano.json'), 'utf8')),
    populations: JSON.parse(fs.readFileSync(join('configs/sense/population_index.json'), 'utf8')),
    actions: JSON.parse(fs.readFileSync(join('configs/action_space.json'), 'utf8')),
  }
}

function loadGoalSpec (idOrPath, root = ROOT) {
  if (idOrPath.endsWith('.json')) {
    return JSON.parse(fs.readFileSync(path.isAbsolute(idOrPath) ? idOrPath : path.join(root, idOrPath), 'utf8'))
  }
  const file = path.join(root, 'configs', 'goals', `${idOrPath.replace(/[:/]/g, '_')}.json`)
  if (fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, 'utf8'))
  const alt = path.join(root, 'configs', 'goals', `${idOrPath}.json`)
  return JSON.parse(fs.readFileSync(alt, 'utf8'))
}

function listGoalSpecs (root = ROOT) {
  const dir = path.join(root, 'configs', 'goals')
  return fs.readdirSync(dir).filter(f => f.endsWith('.json')).map(f => JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8')))
}

module.exports = { ROOT, readJson, loadSenseConfig, loadGoalSpec, listGoalSpecs }

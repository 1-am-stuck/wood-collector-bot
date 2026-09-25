const fs = require('fs')
const path = require('path')

const ROOT = path.join(__dirname, '..', '..')

function readJson (rel) {
  return JSON.parse(fs.readFileSync(path.join(ROOT, rel), 'utf8'))
}

/**
 * Retina config with the measured ommatidial directions spliced in.
 *
 * `retina.json` holds the photoreceptor response parameters; the viewing directions
 * live in the generated `retina_columns.json` because they are derived data, not
 * settings -- `python/sense/retina_map.py` writes them from the male-cns eye map and
 * they must stay byte-identical to the ray order the connectome's lamina nodes are
 * routed to. Keeping them in a separate generated file is what stops someone editing
 * a viewing direction by hand and silently decorrelating the two languages.
 */
function loadRetinaConfig (root = ROOT) {
  const base = JSON.parse(fs.readFileSync(path.join(root, 'configs/sense/retina.json'), 'utf8'))
  const generated = path.join(root, 'configs/sense/retina_columns.json')
  if (!fs.existsSync(generated)) {
    throw new Error(
      'configs/sense/retina_columns.json is missing. Generate it with ' +
      '`uv run python python/sense/retina_map.py`.'
    )
  }
  const columns = JSON.parse(fs.readFileSync(generated, 'utf8'))
  return {
    ...base,
    rays: columns.rays,
    raysPerEye: columns.raysPerEye,
    columns: columns.count,
    eyes: ['L', 'R'],
  }
}

function loadSenseConfig (root = ROOT) {
  const join = (...p) => path.join(root, ...p)
  return {
    root,
    retina: loadRetinaConfig(root),
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

module.exports = { ROOT, readJson, loadSenseConfig, loadRetinaConfig, loadGoalSpec, listGoalSpecs }

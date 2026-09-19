const path = require('path')
const dotenv = require('dotenv')

const ROOT = path.join(__dirname, '..', '..')

function loadRepoEnv (root = ROOT) {
  const result = dotenv.config({ path: path.join(root, '.env'), override: false, quiet: true })
  return result.error ? null : path.join(root, '.env')
}

module.exports = { loadRepoEnv, ROOT }

/** Map the latest flybody Drosophila onto the Mineflayer player.

 * TuragaLab/flybody (DeepMind + Janelia, Nature 2025): body length 0.297 cm,
 * wingspan 0.604 cm, mass 0.983 mg. Minecraft's block is 1 m, and the bot is
 * still a player entity, so the honest size is `length / 1.8 m`, then clamped
 * to vanilla `minecraft:scale` min 0.0625.
 */

const spec = require('../../configs/sense/fly_body.json')

const FLYBODY_LENGTH_M = spec.bodyLengthM
const VANILLA_SCALE_MIN = spec.vanillaScaleMin
const PLAYER_HEIGHT_M = spec.playerHeightM
const PLAYER_WIDTH_M = spec.playerWidthM
const PLAYER_EYE_M = spec.playerEyeHeightM

function targetScale (lengthM = FLYBODY_LENGTH_M) {
  return lengthM / PLAYER_HEIGHT_M
}

function appliedScale (lengthM = FLYBODY_LENGTH_M) {
  return Math.max(VANILLA_SCALE_MIN, targetScale(lengthM))
}

function standingHeightM (lengthM = FLYBODY_LENGTH_M) {
  return PLAYER_HEIGHT_M * appliedScale(lengthM)
}

function standingWidthM (lengthM = FLYBODY_LENGTH_M) {
  return PLAYER_WIDTH_M * appliedScale(lengthM)
}

function eyeHeightM (lengthM = FLYBODY_LENGTH_M) {
  return PLAYER_EYE_M * appliedScale(lengthM)
}

function reachBlocks (baseReach = 5, lengthM = FLYBODY_LENGTH_M) {
  return Math.max(spec.minReachBlocks, baseReach * appliedScale(lengthM))
}

function attributeCommands (lengthM = FLYBODY_LENGTH_M) {
  const scale = appliedScale(lengthM)
  const reach = reachBlocks(5, lengthM)
  return [
    `/attribute @s minecraft:scale base set ${scale}`,
    `/attribute @s minecraft:block_interaction_range base set ${reach}`,
    `/attribute @s minecraft:entity_interaction_range base set ${reach}`,
  ]
}

function sleep (ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function applyFlyBody (bot) {
  if (!bot || typeof bot.chat !== 'function') return
  for (const cmd of attributeCommands()) bot.chat(cmd)
  await sleep(50)
}

module.exports = {
  spec,
  FLYBODY_LENGTH_M,
  VANILLA_SCALE_MIN,
  targetScale,
  appliedScale,
  standingHeightM,
  standingWidthM,
  eyeHeightM,
  reachBlocks,
  attributeCommands,
  applyFlyBody,
}

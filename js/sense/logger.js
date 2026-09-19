const fs = require('fs')
const path = require('path')

function openJsonl (filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true })
  const stream = fs.createWriteStream(filePath, { flags: 'a' })
  return {
    write (row) {
      stream.write(JSON.stringify(row) + '\n')
    },
    close () {
      return new Promise((resolve, reject) => {
        stream.end(() => resolve())
        stream.on('error', reject)
      })
    },
    path: filePath,
  }
}

function compactFrame (frame) {
  const lum = (frame && frame.luminance) || []
  return {
    luminance: lum.map(v => (Number.isNaN(v) ? null : Math.round(v * 1000) / 1000)),
    objects: frame.objects,
    yawRateDegPerS: frame.yawRateDegPerS,
    pitchRateDegPerS: frame.pitchRateDegPerS,
    odor: frame.odor,
    odorBearingDeg: frame.odorBearingDeg,
    taste: frame.taste,
    windLeft: frame.windLeft,
    windRight: frame.windRight,
    soundLow: frame.soundLow,
    soundHigh: frame.soundHigh,
    touchHead: frame.touchHead,
    touchLegs: frame.touchLegs,
    groomDust: frame.groomDust,
    damage: frame.damage,
    hot: frame.hot,
    cold: frame.cold,
    dry: frame.dry,
    moist: frame.moist,
    airborne: frame.airborne,
    legsOnGround: frame.legsOnGround,
    objectChannels: frame.objectChannels,
  }
}

module.exports = { openJsonl, compactFrame }

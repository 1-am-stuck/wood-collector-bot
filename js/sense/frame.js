/**
 * Engine-independent tick of fly senses.
 * Field names match fly-brain-minecraft SensoryFrame.java
 * (head frame: azimuth 0 = ahead, positive = fly's right).
 */
function emptyFrame (nColumns) {
  return {
    luminance: Array(nColumns).fill(NaN),
    objects: [],
    yawRateDegPerS: 0,
    pitchRateDegPerS: 0,
    rollRateDegPerS: 0,
    odor: {},
    odorBearingDeg: null,
    taste: {},
    windLeft: 0,
    windRight: 0,
    tilt: 0,
    soundLow: 0,
    soundHigh: 0,
    song: 0,
    touchHead: 0,
    touchWing: 0,
    touchLegs: 0,
    touchNotum: 0,
    touchAbdomen: 0,
    groomDust: 0,
    damage: 0,
    hot: 0,
    cold: 0,
    dry: 0,
    moist: 0,
    airborne: false,
    legsOnGround: true,
    wingbeat: 0,
    objectChannels: { LC4: 0, LPLC2: 0, LC11: 0, LC18: 0, LC10a: 0, LC15: 0, HS: 0, VS: 0 },
  }
}

function addOdor (frame, glom, drive) {
  frame.odor[glom] = (frame.odor[glom] || 0) + drive
}

function addTaste (frame, grn, drive) {
  frame.taste[grn] = Math.max(frame.taste[grn] || 0, drive)
}

function clampOdor (frame, max = 1.5) {
  for (const k of Object.keys(frame.odor)) {
    frame.odor[k] = Math.min(max, frame.odor[k])
  }
}

function hill (s, n = 1.5, k = 0.2) {
  if (s <= 0) return 0
  const sn = s ** n
  return sn / (k ** n + sn)
}

module.exports = { emptyFrame, addOdor, addTaste, clampOdor, hill }

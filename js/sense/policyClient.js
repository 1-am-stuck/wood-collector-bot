const net = require('net')

function connectPolicy (host = '127.0.0.1', port = 8765) {
  let sock = null
  let buf = ''
  const pending = []

  function ensure () {
    if (sock && !sock.destroyed) return Promise.resolve()
    return new Promise((resolve, reject) => {
      sock = net.connect(port, host, resolve)
      sock.setEncoding('utf8')
      sock.on('data', chunk => {
        buf += chunk
        let idx
        while ((idx = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, idx)
          buf = buf.slice(idx + 1)
          const waiter = pending.shift()
          if (!waiter) continue
          try { waiter.resolve(JSON.parse(line)) } catch (err) { waiter.reject(err) }
        }
      })
      sock.on('error', err => {
        const w = pending.shift()
        if (w) w.reject(err)
        else reject(err)
      })
    })
  }

  async function request (msg) {
    await ensure()
    return new Promise((resolve, reject) => {
      pending.push({ resolve, reject })
      sock.write(JSON.stringify(msg) + '\n')
    })
  }

  return {
    async act (frame, extra = {}) {
      const res = await request({ type: 'act', frame, ...extra })
      return res
    },
    async ping () {
      return request({ type: 'ping' })
    },
    close () {
      if (sock) sock.destroy()
    },
  }
}

module.exports = { connectPolicy }

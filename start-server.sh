#!/bin/sh
# Real Paper 1.21.11 on localhost:25565. No synthetic stand-in.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/server"
mkdir -p "$ROOT/server"
# Vanilla player is 1.8 m; flybody Drosophila is 2.97 mm. This datapack applies
# minecraft:scale 0.0625 (the engine floor) so the bot is insect-sized vs blocks.
mkdir -p "$ROOT/server/world/datapacks"
if [ -d "$ROOT/configs/minecraft/datapacks/flybody_scale" ]; then
  rm -rf "$ROOT/server/world/datapacks/flybody_scale"
  cp -R "$ROOT/configs/minecraft/datapacks/flybody_scale" "$ROOT/server/world/datapacks/flybody_scale"
fi

if [ ! -f eula.txt ]; then
  printf '%s\n' 'eula=true' > eula.txt
fi

have_java() {
  if [ -x "$ROOT/server/jre/Contents/Home/bin/java" ]; then
    JAVA="$ROOT/server/jre/Contents/Home/bin/java"
    return 0
  fi
  if command -v java >/dev/null 2>&1 && java -version 2>&1 | grep -qi 'version'; then
    JAVA=java
    return 0
  fi
  for c in \
    /opt/homebrew/opt/openjdk@21/bin/java \
    /opt/homebrew/opt/openjdk/bin/java \
    /usr/local/opt/openjdk@21/bin/java
  do
    if [ -x "$c" ]; then
      JAVA="$c"
      return 0
    fi
  done
  return 1
}

fetch_paper() {
  if [ -f paper.jar ]; then
    return 0
  fi
  echo "downloading Paper 1.21.11…"
  url="$(curl -fsSL -A 'wood-collector-bot/0.1 (https://github.com/agniva/wood-collector-bot)' \
    'https://fill.papermc.io/v3/projects/paper/versions/1.21.11/builds/latest' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["downloads"]["server:default"]["url"])')"
  curl -fL -A 'wood-collector-bot/0.1 (https://github.com/agniva/wood-collector-bot)' \
    -o paper.jar "$url"
}

if have_java; then
  fetch_paper
  echo "starting Paper with $JAVA"
  exec "$JAVA" -Xms1G -Xmx2G -jar paper.jar nogui
fi

if command -v docker >/dev/null 2>&1; then
  echo "no local JRE — starting Paper 1.21.11 in Docker (still Minecraft on :25565)"
  docker rm -f fly-paper >/dev/null 2>&1 || true
  exec docker run --rm --name fly-paper \
    -p 127.0.0.1:25565:25565 \
    -e EULA=TRUE \
    -e TYPE=PAPER \
    -e VERSION=1.21.11 \
    -e MODE=creative \
    -e ONLINE_MODE=false \
    -e ALLOW_FLIGHT=true \
    -e DIFFICULTY=peaceful \
    -e SPAWN_PROTECTION=0 \
    -e OPS=FruitFly \
    -e OVERRIDE_SERVER_PROPERTIES=true \
    -e MEMORY=2G \
    -v "$ROOT/server:/data" \
    itzg/minecraft-server
fi

echo "Need Java 21 or Docker to run Paper." >&2
echo "  brew install openjdk@21" >&2
exit 1

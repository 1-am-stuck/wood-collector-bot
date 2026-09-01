#!/bin/sh
# Starts the local Paper dev server (Minecraft 1.21.11) on localhost:25565
cd "$(dirname "$0")/server"
exec ./jre/Contents/Home/bin/java -Xms1G -Xmx2G -jar paper.jar nogui

#!/bin/sh
set -eu
config_file=/usr/share/nginx/html/config.js
printf 'window.DELTA_CONFIG = ' > "$config_file"
jq -n --arg coordinator "${DELTA_COORDINATOR_URL:-http://localhost:8000}" --arg parties "${DELTA_PARTY_URLS:-http://localhost:8100,http://localhost:8101,http://localhost:8102}" --arg storage "${DELTA_STORAGE_URL:-http://localhost:8200}" --arg pocketbase "${DELTA_POCKETBASE_URL:-http://127.0.0.1:8090}" '{coordinatorUrl:$coordinator,partyUrls:($parties|split(",")),storageUrl:$storage,pocketbaseUrl:$pocketbase}' >> "$config_file"
printf ';\n' >> "$config_file"
if [ "${DELTA_WEB_MODE:-processing}" = catalog ]; then
    cp /usr/share/nginx/html/catalog.html /usr/share/nginx/html/index.html
fi

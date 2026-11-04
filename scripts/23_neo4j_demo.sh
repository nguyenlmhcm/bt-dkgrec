#!/usr/bin/env bash
# Bat Neo4j cho demo va nap do thi nhom Active. Chi nghe tren 127.0.0.1. Chay lai thi bat lai, khong nap trung.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/neo4j/active_bt_dkgrec_l05_evaluated"
NAME=bt_neo4j_demo
if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker run -d --name "$NAME" -p 127.0.0.1:7687:7687 -e NEO4J_AUTH=none \
    -e NEO4J_server_memory_heap_max__size=512M -e NEO4J_server_memory_pagecache_size=256M \
    -v "$DATA":/import:ro neo4j:5.26-community >/dev/null
fi
docker start "$NAME" >/dev/null
until docker exec "$NAME" cypher-shell "RETURN 1" >/dev/null 2>&1; do sleep 2; done
N=$(docker exec "$NAME" cypher-shell --format plain "MATCH (e:Event) RETURN count(e)" | tail -1)
if [ "$N" = "0" ]; then docker exec "$NAME" cypher-shell -f /import/import.cypher; fi
docker exec "$NAME" cypher-shell --format plain "MATCH (e:Event) RETURN count(e) AS su_kien"

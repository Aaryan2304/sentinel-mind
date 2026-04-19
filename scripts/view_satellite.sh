#!/bin/bash
# View current satellite image from SimSat
# Usage: bash scripts/view_satellite.sh [band1] [band2] [band3]
# Example: bash scripts/view_satellite.sh nir red green  (false color)

BAND1=${1:-red}
BAND2=${2:-green}
BAND3=${3:-blue}
OUTPUT="/tmp/satellite_view.png"

# Check if SimSat is running
if ! curl -s http://localhost:9005/data/current/position > /dev/null 2>&1; then
    echo "ERROR: SimSat API not reachable at localhost:9005"
    echo "Start SimSat: cd simsat-reference && docker compose up"
    exit 1
fi

# Get satellite position
POS=$(curl -s http://localhost:9005/data/current/position)
LON=$(echo "$POS" | python3 -c "import sys,json; print(json.load(sys.stdin)['lon-lat-alt'][0])")
LAT=$(echo "$POS" | python3 -c "import sys,json; print(json.load(sys.stdin)['lon-lat-alt'][1])")
ALT=$(echo "$POS" | python3 -c "import sys,json; print(json.load(sys.stdin)['lon-lat-alt'][2])")
TS=$(echo "$POS" | python3 -c "import sys,json; print(json.load(sys.stdin)['timestamp'])")

echo "Satellite: lon=$LON, lat=$LAT, alt=${ALT}km"
echo "Time: $TS"
echo "Bands: $BAND1, $BAND2, $BAND3"

# Fetch image (with headers to check metadata)
HEADERS=$(curl -s -D - -o "$OUTPUT" \
  "http://localhost:9005/data/image/sentinel?lon=$LON&lat=$LAT&timestamp=$TS&spectral_bands=$BAND1&spectral_bands=$BAND2&spectral_bands=$BAND3&size_km=5.0&return_type=png" 2>&1)

# Check if image is actually available
AVAILABLE=$(echo "$HEADERS" | grep -o '"image_available": [a-z]*' | head -1 | cut -d' ' -f2)
CLOUD=$(echo "$HEADERS" | grep -o '"cloud_cover": [0-9.]*' | head -1 | cut -d' ' -f2)

if [ "$AVAILABLE" = "false" ]; then
    echo "No image available at this position (ocean, polar, or no data)"
    rm -f "$OUTPUT"
    exit 0
fi

echo "Cloud cover: ${CLOUD}%"
SIZE=$(stat -c%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT" 2>/dev/null)
echo "Image saved: $OUTPUT ($SIZE bytes)"

# Try to open with default viewer
if command -v xdg-open &> /dev/null; then
    xdg-open "$OUTPUT" 2>/dev/null &
elif command -v wslview &> /dev/null; then
    wslview "$OUTPUT" 2>/dev/null &
else
    echo "Open $OUTPUT manually to view"
fi

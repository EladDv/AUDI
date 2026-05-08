#!/bin/bash
# Launch Chromium in kiosk mode once audi-app web UI is ready.
# Place in ~/.config/autostart/audi-kiosk.desktop to run at boot.

URL="${1:-http://localhost:8080}"
MAX_WAIT=90

echo "Waiting for audi-app at $URL ..."
for i in $(seq 1 "$MAX_WAIT"); do
    if curl -s -o /dev/null -w "%{http_code}" "$URL" | grep -q 200; then
        echo "Ready after ${i}s"
        break
    fi
    sleep 1
done

if command -v unclutter >/dev/null 2>&1; then
    # Hide cursor after 3s of inactivity.
    unclutter -idle 3 -root &
fi

if command -v chromium-browser >/dev/null 2>&1; then
    BROWSER="chromium-browser"
elif command -v chromium >/dev/null 2>&1; then
    BROWSER="chromium"
else
    echo "ERROR: Chromium is not installed. Run scripts/install-kiosk.sh." >&2
    exit 1
fi

exec "$BROWSER" \
    --kiosk \
    --noerrdialogs \
    --disable-infobars \
    --no-first-run \
    --disable-translate \
    --disable-features=TranslateUI \
    --overscroll-history-navigation=0 \
    --disable-pinch \
    "$URL"

#!/bin/sh
set -eu

password_file=${REDIS_HEALTHCHECK_PASSWORD_FILE:-/run/redis/healthcheck.password}
password=
response_file=

cleanup() {
    unset REDISCLI_AUTH password
    if [ -n "$response_file" ]; then
        rm -f "$response_file"
    fi
}

trap cleanup EXIT
trap 'exit 1' HUP INT TERM

unset REDISCLI_AUTH
if [ ! -f "$password_file" ] || [ -L "$password_file" ] || [ ! -r "$password_file" ]; then
    exit 1
fi

byte_count=$(wc -c < "$password_file" | tr -d '[:space:]')
allowed_count=$(LC_ALL=C tr -cd 'A-Za-z0-9_-' < "$password_file" | wc -c | tr -d '[:space:]')
if [ "$byte_count" -lt 32 ] || [ "$byte_count" -ne "$allowed_count" ]; then
    exit 1
fi

password=$(cat "$password_file") || exit 1
response_file=$(mktemp) || exit 1

if ! REDISCLI_AUTH="$password" REDISCLI_HISTFILE=/dev/null \
    redis-cli --raw --user healthcheck ping > "$response_file" 2>/dev/null
then
    exit 1
fi

printf 'PONG\n' | cmp -s - "$response_file"

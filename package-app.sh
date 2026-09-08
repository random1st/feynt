#!/bin/bash
# Package the SwiftPM QwenLocal executable into a drag-and-drop QwenLocal.app.
#
# SwiftPM only produces a bare executable; macOS needs a bundle with an Info.plist
# (LSUIElement, so the app is menu-bar-first) and a signature carrying the JIT
# entitlement MLX's runtime Metal codegen requires.
#
# Ad-hoc signing (`-`) is the default so the script works on any machine without a
# Developer ID. Override with QWENLOCAL_SIGN_IDENTITY for a distributable build.
#
# Usage: ./package-app.sh [debug|release]
set -euo pipefail

# Resolve paths relative to this script — no hardcoded home directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PKG="$ROOT/packaging"
CONFIG="${1:-debug}"
APP="$ROOT/build/QwenLocal.app"
IDENTITY="${QWENLOCAL_SIGN_IDENTITY:--}"

if [ "$CONFIG" != "debug" ] && [ "$CONFIG" != "release" ]; then
    echo "usage: $(basename "$0") [debug|release]" >&2
    exit 2
fi

echo "1. swift build ($CONFIG)..."
cd "$ROOT"
if [ "$CONFIG" = "release" ]; then
    swift build -c release
    BIN_DIR="$ROOT/.build/release"
else
    swift build
    BIN_DIR="$ROOT/.build/debug"
fi

echo "2. Assembling $APP ..."
# Idempotent: a previous bundle is replaced wholesale, never merged into.
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$PKG/Info.plist" "$APP/Contents/Info.plist"
cp "$BIN_DIR/QwenLocal" "$APP/Contents/MacOS/QwenLocal"

# SwiftPM resource bundles (Bundle.module) are resolved next to the executable. They are
# emitted as FLAT directories with no Info.plist, which codesign rejects with "bundle format
# unrecognized" — give each one a minimal plist so it becomes a signable flat bundle.
RES_BUNDLES=()
set +u  # bash 3.2: empty-array expansion trips `set -u` in the loops below
for bundle in "$BIN_DIR"/*.bundle; do
    [ -d "$bundle" ] || continue
    name="$(basename "$bundle" .bundle)"
    cp -R "$bundle" "$APP/Contents/MacOS/"
    target="$APP/Contents/MacOS/$name.bundle"
    if [ ! -f "$target/Info.plist" ]; then
        cat > "$target/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>com.random1st.qwenlocal.resources.$name</string>
    <key>CFBundleName</key><string>$name</string>
    <key>CFBundlePackageType</key><string>BNDL</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
</dict>
</plist>
PLIST
    fi
    RES_BUNDLES+=("$target")
done
echo "   resource bundles: ${#RES_BUNDLES[@]}"

# Metal shader libraries ship next to the binary and must travel with the app.
for lib in "$BIN_DIR"/*.metallib; do
    [ -f "$lib" ] || continue
    cp "$lib" "$APP/Contents/MacOS/"
    echo "   metallib: $(basename "$lib")"
done

if [ "$IDENTITY" = "-" ]; then
    echo "3. Codesigning (ad-hoc; set QWENLOCAL_SIGN_IDENTITY for a Developer ID)..."
else
    echo "3. Codesigning as '$IDENTITY' (hardened runtime + timestamp)..."
fi

# Hardened runtime and a secure timestamp are required for notarization and meaningless
# for an ad-hoc signature. Wrapped in a function because macOS ships bash 3.2, where
# expanding an empty array under `set -u` injects a bogus empty argument.
sign() {
    local target="$1"
    shift
    if [ "$IDENTITY" = "-" ]; then
        codesign --force "$@" --sign "$IDENTITY" "$target"
    else
        codesign --force --options runtime --timestamp "$@" --sign "$IDENTITY" "$target"
    fi
}

# Inside-out: nested bundles first, then the executable, then the app wrapper.
if [ "${#RES_BUNDLES[@]}" -gt 0 ]; then
    for bundle in "${RES_BUNDLES[@]}"; do
        sign "$bundle"
    done
fi
sign "$APP/Contents/MacOS/QwenLocal" --entitlements "$PKG/QwenLocal.entitlements"
sign "$APP" --entitlements "$PKG/QwenLocal.entitlements"

echo "4. Verifying..."
codesign --verify --strict --verbose=2 "$APP"
codesign -d --entitlements - --xml "$APP" >/dev/null 2>&1 && echo "   entitlements embedded"

SIZE="$(du -sh "$APP" | cut -f1)"
echo
echo "Done: $APP ($CONFIG, $SIZE, identity '$IDENTITY')"
echo "Drag it to /Applications. The endpoint is http://127.0.0.1:19234/v1 once a model is loaded."

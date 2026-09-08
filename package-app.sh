#!/bin/bash
# Package the SwiftPM Feynt executable into a drag-and-drop Feynt.app.
#
# SwiftPM only produces a bare executable; macOS needs a bundle with an Info.plist
# (LSUIElement, so the app is menu-bar-first) and a signature carrying the JIT
# entitlement MLX's runtime Metal codegen requires.
#
# Ad-hoc signing (`-`) is the default so the script works on any machine without a
# Developer ID. Override with FEYNT_SIGN_IDENTITY for a distributable build.
#
# Usage: ./package-app.sh [debug|release]
set -euo pipefail

# Resolve paths relative to this script — no hardcoded home directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PKG="$ROOT/packaging"
CONFIG="${1:-debug}"
APP="$ROOT/build/Feynt.app"
IDENTITY="${FEYNT_SIGN_IDENTITY:--}"

if [ "$CONFIG" != "debug" ] && [ "$CONFIG" != "release" ]; then
    echo "usage: $(basename "$0") [debug|release]" >&2
    exit 2
fi

# Built with xcodebuild rather than `swift build`, and not for taste: SwiftPM does not
# emit mlx-swift's `mlx-swift_Cmlx.bundle`, which carries `default.metallib`. Without it
# the app launches, shows its window, and then dies the moment it touches MLX —
# "Failed to load the default metallib" on stdout, no crash report, no log line.
XCCONFIG=$([ "$CONFIG" = "release" ] && echo Release || echo Debug)
echo "1. xcodebuild ($XCCONFIG)..."
cd "$ROOT"
xcodebuild -scheme Feynt -configuration "$XCCONFIG" \
    -destination 'platform=macOS,arch=arm64' \
    -derivedDataPath "$ROOT/.build/xcode" \
    -skipPackagePluginValidation -skipMacroValidation \
    build > "$ROOT/.build/xcodebuild.log" 2>&1 || {
        echo "xcodebuild failed; see .build/xcodebuild.log" >&2
        grep -E "error:" "$ROOT/.build/xcodebuild.log" | head -5 >&2
        exit 1
    }
BIN_DIR="$ROOT/.build/xcode/Build/Products/$XCCONFIG"
[ -d "$BIN_DIR/mlx-swift_Cmlx.bundle" ] || {
    echo "the Metal library bundle is missing from $BIN_DIR — the app would die on first use" >&2
    exit 1
}

echo "2. Assembling $APP ..."
# Idempotent: a previous bundle is replaced wholesale, never merged into.
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$PKG/Info.plist" "$APP/Contents/Info.plist"
cp "$PKG/icon.icns" "$APP/Contents/Resources/icon.icns"
cp "$BIN_DIR/Feynt" "$APP/Contents/MacOS/Feynt"

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
    # Xcode emits some dependencies as proper bundles with Contents/; dropping a plist in
    # their root makes codesign refuse them as "unsealed contents present in the bundle
    # root". Only the flat SwiftPM ones need the shim.
    if [ -d "$target/Contents" ]; then
        RES_BUNDLES+=("$target")
        continue
    fi
    if [ ! -f "$target/Info.plist" ]; then
        cat > "$target/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>com.random1st.feynt.resources.$name</string>
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

# MLX looks for its Metal library in a fixed order, and the first place it checks is
# `mlx.metallib` sitting next to the binary (mlx/backend/metal/device.cpp,
# load_default_library). Shipping the SwiftPM bundle alone was not enough — inside an .app
# the bundle lookup does not resolve, and MLX aborts the process the moment it is touched:
# "Failed to load the default metallib", with no crash report and no log line.
METALLIB="$(find "$BIN_DIR" -name default.metallib -print -quit 2>/dev/null || true)"
if [ -n "$METALLIB" ]; then
    cp "$METALLIB" "$APP/Contents/MacOS/mlx.metallib"
    echo "   metallib: mlx.metallib"
else
    echo "no default.metallib under $BIN_DIR — the app would die on first use" >&2
    exit 1
fi

if [ "$IDENTITY" = "-" ]; then
    echo "3. Codesigning (ad-hoc; set FEYNT_SIGN_IDENTITY for a Developer ID)..."
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
sign "$APP/Contents/MacOS/Feynt" --entitlements "$PKG/Feynt.entitlements"
sign "$APP" --entitlements "$PKG/Feynt.entitlements"

echo "4. Verifying..."
codesign --verify --strict --verbose=2 "$APP"
codesign -d --entitlements - --xml "$APP" >/dev/null 2>&1 && echo "   entitlements embedded"

SIZE="$(du -sh "$APP" | cut -f1)"
echo
echo "Done: $APP ($CONFIG, $SIZE, identity '$IDENTITY')"
echo "Drag it to /Applications. The endpoint is http://127.0.0.1:19234/v1 once a model is loaded."

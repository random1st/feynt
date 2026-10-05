#!/bin/bash
# Cut a distributable release: signed app, DMG, notarised by Apple, ticket stapled.
#
# The steps are in one script because they only work in this order and each one fails
# quietly if skipped. An unsigned DMG notarises fine and then Gatekeeper blocks the app
# inside it; an unstapled DMG passes on this machine, where the notarisation is cached,
# and shows "damaged and cannot be opened" on the first machine that opens it offline.
#
# Credentials are never handled here: the Developer ID is a public name that appears in
# every signed binary, and notarisation goes through the `notarize-diana` keychain
# profile, which holds the app-specific password.
#
# Usage: ./release.sh 0.2.0
set -euo pipefail

VERSION="${1:?usage: release.sh <version>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
APP="$ROOT/build/Feynt.app"
DMG="$ROOT/build/Feynt-$VERSION.dmg"
IDENTITY="${FEYNT_SIGN_IDENTITY:-Developer ID Application: Roman Voronin (56LH34GFLK)}"
PROFILE="${FEYNT_NOTARY_PROFILE:-notarize-diana}"

echo "== version $VERSION"
# The bundle is what a user's "About" reads, and Homebrew compares it against the cask -
# a mismatch makes `brew upgrade` reinstall the same build forever.
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$ROOT/packaging/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $VERSION" "$ROOT/packaging/Info.plist"

FEYNT_SIGN_IDENTITY="$IDENTITY" "$ROOT/package-app.sh" release

# Debug symbols, kept beside the DMG and never shipped in it.
#
# A release binary is stripped and Xcode's DerivedData is overwritten by the next
# build, so a crash report from a released version is a list of offsets nobody can
# resolve. That is not hypothetical: 0.7.0 segfaulted in the generation thread on
# 2026-10-04 and the frames could not be named, because by then the only matching
# dSYM had been rebuilt away. Keeping it costs a copy.
DSYM="$ROOT/.build/xcode/Build/Products/Release/Feynt.dSYM"
if [ -d "$DSYM" ]; then
    rm -rf "$ROOT/build/Feynt-$VERSION.dSYM"
    cp -R "$DSYM" "$ROOT/build/Feynt-$VERSION.dSYM"
    echo "== symbols kept: build/Feynt-$VERSION.dSYM"
else
    echo "WARNING: no dSYM at $DSYM - a crash in $VERSION will not be symbolicatable" >&2
fi

# Приложение получает свой билет ДО того, как попадёт в DMG.
echo "== app ticket"
ZIP="$ROOT/build/Feynt-app.zip"
rm -f "$ZIP"
ditto -c -k --keepParent "$APP" "$ZIP"
xcrun notarytool submit "$ZIP" --keychain-profile "$PROFILE" --wait
xcrun stapler staple "$APP"
rm -f "$ZIP"

echo "== DMG"
rm -f "$DMG"
STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "Feynt" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$STAGE"
codesign --force --sign "$IDENTITY" --timestamp "$DMG"

echo "== notarising (a few minutes)"
xcrun notarytool submit "$DMG" --keychain-profile "$PROFILE" --wait
xcrun stapler staple "$DMG"
xcrun stapler validate "$DMG"

echo
echo "$DMG"
shasum -a 256 "$DMG"

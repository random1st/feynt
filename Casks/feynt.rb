cask "feynt" do
  version "0.7.3"
  sha256 "5e9c36259b18efa8c69cd4c2f82c33c3866b37aa12d17f6cf30e0f40b7a51fd6"

  url "https://github.com/random1st/feynt/releases/download/v#{version}/Feynt-#{version}.dmg"
  name "Feynt"
  desc "Local LLM on Apple Silicon with DFlash 2 speculative decoding"
  homepage "https://github.com/random1st/feynt"

  depends_on macos: :sonoma
  depends_on arch: :arm64

  app "Feynt.app"

  # Models and the prefix cache live here; a plain uninstall would strip the
  # download too, which is not what someone reinstalling wants.
  zap trash: [
    "~/Library/Application Support/Feynt",
    "~/Library/Logs/Feynt",
    "~/Library/Preferences/com.random1st.feynt.plist",
  ]

  # Signed with a Developer ID and notarised by Apple. Both the app and the DMG
  # carry their own ticket, so the first launch needs no network round trip.
end

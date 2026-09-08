cask "feynt" do
  version "0.2.1"
  sha256 "314756e60eda040c2557c336c3447dd7fba96723a82332ebfa9c93f4a230cf12"

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

  # Signed with a Developer ID and notarised by Apple; the ticket is attached to
  # the DMG, so Gatekeeper accepts the app without a manual quarantine dance.
end

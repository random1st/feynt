cask "feynt" do
  version "0.3.0"
  sha256 "749026e3d7cbb1ff635775869790fa93f29ffbe840a87c3da4ee9f6a07537edf"

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

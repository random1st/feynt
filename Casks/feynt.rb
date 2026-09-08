cask "feynt" do
  version "0.1.0"
  sha256 "edcdcfd48b3f5ed5b3f5a51a403bdf4086bda5e41b04c23cc826e09d89665b31"

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

  # Signed with a Developer ID and notarised by Apple, so the first launch needs
  # no quarantine dance. The ticket is not stapled into the disk image, so that
  # first launch does check with Apple and therefore wants a network.
end

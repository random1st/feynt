cask "feynt" do
  version "0.1.0"
  sha256 "50495e1677054f611a0ef399cfa0222622cc2dee875ecd956a7915906f4644fb"

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

  caveats <<~EOS
    The app is not notarised yet, so macOS blocks the first launch. Either run

      xattr -dr com.apple.quarantine /Applications/Feynt.app

    once, or allow it under System Settings > Privacy & Security after the
    first attempt.
  EOS
end

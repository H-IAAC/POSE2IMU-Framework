REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

rm -rf "$REPO_ROOT/checkpoints"
mkdir "$REPO_ROOT/checkpoints"
cd "$REPO_ROOT/checkpoints"

echo "Downloading"
gdown "https://drive.google.com/uc?id=1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8"
echo "Extracting"
tar xfzv t2m.tar.gz
echo "Cleaning"
rm t2m.tar.gz

echo -e "Downloading done!"

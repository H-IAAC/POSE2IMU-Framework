REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

mkdir -p "$REPO_ROOT/body_models"
cd "$REPO_ROOT/body_models"

echo -e "The smpl files will be stored in the 'body_models/smpl/' folder\n"
gdown 1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2
rm -rf smpl

unzip smpl.zip
echo -e "Cleaning\n"
rm smpl.zip

echo -e "Downloading done!"

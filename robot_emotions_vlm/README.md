# robot_emotions_vlm

Standalone RobotEmotions video-description module powered by `Qwen/Qwen3-VL-8B-Instruct`.

It scans `data/RobotEmotions`, generates one movement description per video capture, and exports a Kimodo-ready prompt catalog.

## Requirements

Run it directly in the Conda environment `kimodo`:

```bash
conda activate kimodo
python -c "from transformers import AutoProcessor, Qwen3VLForConditionalGeneration; import av; print('qwen3_vl_import_ok')"
```

## Run

Process the full dataset:

```bash
conda activate kimodo
python -m robot_emotions_vlm describe-videos \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_qwen
```

Process only specific domains:

```bash
conda activate kimodo
python -m robot_emotions_vlm describe-videos \
  --dataset-root data/RobotEmotions \
  --domains 10ms 30ms \
  --output-dir output/robot_emotions_qwen
```

Process a single clip:

```bash
conda activate kimodo
python -m robot_emotions_vlm describe-videos \
  --dataset-root data/RobotEmotions \
  --clip-id robot_emotions_10ms_u02_tag11 \
  --output-dir output/robot_emotions_qwen_single
```

Use only local Hugging Face files:

```bash
conda activate kimodo
python -m robot_emotions_vlm describe-videos \
  --dataset-root data/RobotEmotions \
  --output-dir output/robot_emotions_qwen \
  --local-files-only
```

Generate Kimodo motions for all catalog entries:

```bash
conda activate kimodo
python -m robot_emotions_vlm generate-kimodo \
  --catalog-path output/robot_emotions_qwen/kimodo_prompt_catalog.jsonl \
  --output-dir output/robot_emotions_kimodo
```

By default, `generate-kimodo` uses `Kimodo-SMPLX-RP-v1` so that an AMASS file is exported next to each generated NPZ.

Generate only selected clips from the catalog:

```bash
conda activate kimodo
python -m robot_emotions_vlm generate-kimodo \
  --catalog-path output/robot_emotions_qwen/kimodo_prompt_catalog.jsonl \
  --clip-id robot_emotions_10ms_u02_tag11 \
  --output-dir output/robot_emotions_kimodo_single
```

## Main options

- `--model-id`: defaults to `Qwen/Qwen3-VL-8B-Instruct`
- `--num-video-frames`: defaults to `32`
- `--max-new-tokens`: defaults to `384`
- `--temperature`: defaults to `0.2`
- `--top-p`: defaults to `0.9`
- `--system-prompt-path`: override the system prompt template
- `--user-prompt-path`: override the user prompt template
- `--catalog-output-path`: write the Kimodo catalog to a custom path
- `generate-kimodo --duration-sec`: fallback duration when the catalog has no `duration_hint_sec`
- `generate-kimodo --model`: choose the Kimodo model to run; default is `Kimodo-SMPLX-RP-v1`
- `generate-kimodo --bvh`: also export BVH for SOMA models

## Outputs

Root files:

- `video_description_manifest.jsonl`
- `video_description_summary.json`
- `kimodo_prompt_catalog.jsonl`
- `kimodo_generation_manifest.jsonl`
- `kimodo_generation_summary.json`

Per-clip files:

- `description.json`
- `raw_response.txt`
- `prompt_context.json`
- `quality_report.json`

Per-generated clip:

- `prompt_entry.json`
- `generation_config.json`
- `motion.npz` or a `motion/` folder for multiple samples
- `motion_amass.npz` next to each NPZ when using the SMPL-X model
- optional `motion.bvh` or `motion.csv` depending on the Kimodo model and flags

## Notes

- The first real run may download model weights from Hugging Face.
- The module is independent from `pose_module`.
- Prompt templates are editable in `robot_emotions_vlm/prompt_templates/`.

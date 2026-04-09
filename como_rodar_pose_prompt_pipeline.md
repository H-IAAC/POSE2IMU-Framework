# Como rodar cada parte do pipeline

Este guia resume como executar as partes disponíveis do pipeline atual no repositório, com foco em:

- caminho por vídeo já integrado
- caminho por prompt implementado na fase 1
- preparação do ambiente e dos artefatos necessários

## 1. Pré-requisitos

Use sempre a `venv` do projeto:

```bash
source .venv/bin/activate
```

Os artefatos mínimos para o backend por prompt já devem existir nestes caminhos:

```text
pretrained/VQVAE/net_last.pth
pretrained/VQTransformer_corruption05/net_best_fid.pth
checkpoints/t2m/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy
checkpoints/t2m/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy
ViT-B-32.pt
```

Dependências importantes na `.venv`:

- `torch`
- `clip`
- `gdown`

Se quiser baixar novamente os pacotes legados:

```bash
PATH="$PWD/.venv/bin:$PATH" bash dataset/prepare/download_model.sh
PATH="$PWD/.venv/bin:$PATH" bash dataset/prepare/download_extractor.sh
```

Observação:

- para a fase 1 do ramo por prompt, `glove/` não é obrigatório
- o backend por prompt já foi validado localmente com geração real de pose 3D

## 2. Entry point da CLI

Todos os comandos abaixo usam:

```bash
.venv/bin/python -m pose_module.robot_emotions
```

Lista completa de subcomandos:

- `scan`
- `export-imu`
- `export-pose2d`
- `export-pose3d`
- `build-prompt-catalog`
- `export-prompt-pose3d`
- `export-virtual-imu`

## 3. Parte A: inspecionar o dataset

Para listar rapidamente os registros detectados no dataset:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  scan \
  --dataset-root data/RobotEmotions \
  --domains 10ms 30ms
```

Use `--clip-id ...` para limitar a alguns clipes específicos.

## 4. Parte B: exportar o IMU real

Esse comando só extrai e normaliza os artefatos reais do dataset:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-imu \
  --dataset-root data/RobotEmotions \
  --domains 10ms 30ms \
  --output-dir output/robot_emotions_imu
```

Saídas principais:

- `manifest.jsonl`
- `summary.json`
- por clipe: `imu.npz` e `metadata.json`

## 5. Parte C: exportar pose 2D a partir de vídeo

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-pose2d \
  --dataset-root data/RobotEmotions \
  --domains 10ms 30ms \
  --output-dir output/robot_emotions_pose2d \
  --fps-target 20
```

Flags úteis:

- `--clip-id ...`
- `--env-name openmmlab`
- `--no-debug`

Saídas principais:

- `pose_manifest.jsonl`
- `pose2d_summary.json`
- por clipe: `pose/pose2d/pose2d.npz`, `quality_report.json` e overlays de debug

## 6. Parte D: exportar pose 3D real a partir de vídeo

Esse é o caminho por vídeo que já estava integrado:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-pose3d \
  --dataset-root data/RobotEmotions \
  --domains 10ms 30ms \
  --output-dir output/robot_emotions_pose3d \
  --fps-target 20
```

Flags úteis:

- `--clip-id ...`
- `--debug-2d` ou `--no-debug-2d`
- `--debug-3d` ou `--no-debug-3d`
- `--motionbert-env-name ...`
- `--motionbert-device auto`
- `--allow-motionbert-fallback-backend`

Saídas principais:

- `pose3d_manifest.jsonl`
- `pose3d_summary.json`
- por clipe:
  - `pose3d_motionbert17.npz`
  - `pose3d_metric_local.npz`
  - `pose3d.npz`
  - `pose3d.bvh`
  - `quality_report.json`

## 7. Parte E: montar o catálogo de prompts canônico

Esse é o primeiro passo do ramo por prompt da fase 1.

Sem enriquecimento por pose real:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  build-prompt-catalog \
  --dataset-root data/RobotEmotions \
  --domains 10ms 30ms \
  --output-path data/prompts/robot_emotions_prompts.jsonl
```

Com enriquecimento usando estatísticas de pose real:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  build-prompt-catalog \
  --dataset-root data/RobotEmotions \
  --domains 10ms 30ms \
  --output-path data/prompts/robot_emotions_prompts.jsonl \
  --real-pose3d-manifest-path output/robot_emotions_virtual_imu_v2_all_dataset/virtual_imu_manifest.jsonl
```

Flags úteis:

- `--num-samples 1`
- `--seed 123`
- `--fps 20`

Saídas principais:

- `data/prompts/robot_emotions_prompts.jsonl`
- `data/prompts/robot_emotions_prompts.summary.json`

Observações:

- o catálogo cobre as 18 condições canônicas do protocolo
- se o manifest real for passado, os prompts são enriquecidos com traços cinemáticos
- sem o manifest real, o sistema gera prompts base curados por condição

## 8. Parte F: exportar pose 3D sintética a partir do catálogo

Esse é o segundo passo do ramo por prompt da fase 1.

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-prompt-pose3d \
  --prompt-catalog data/prompts/robot_emotions_prompts.jsonl \
  --output-dir output/robot_emotions_virtual_imu_v2_all_dataset/virtual_pose3d
```

Para desabilitar o BVH opcional:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-prompt-pose3d \
  --prompt-catalog data/prompts/robot_emotions_prompts.jsonl \
  --output-dir output/robot_emotions_prompt_pose3d \
  --no-bvh
```

Saídas principais:

- `prompt_pose3d_manifest.jsonl`
- `prompt_pose3d_summary.json`
- por amostra sintética em `synthetic/<sample_id>/pose/`:
  - `prompt_metadata.json`
  - `prompt_backend_report.json`
  - `pose3d_prompt_raw.npz`
  - `pose3d_metric_local.npz`
  - `pose3d.npz`
  - `pose3d.bvh`
  - `quality_report.json`

Observações:

- o fluxo interno é `prompt -> T2M-GPT -> PoseSequence3D -> metric_normalizer -> root_estimator`
- o BVH é apenas artefato opcional de inspeção
- nesta fase ainda não existe comando público para `prompt -> virtual IMU`

## 9. Parte G: exportar o pipeline completo por vídeo até virtual IMU

Esse comando continua disponível para o ramo por vídeo:

```bash
.venv/bin/python -m pose_module.robot_emotions \
  export-virtual-imu \
  --dataset-root data/RobotEmotions \
  --domains 10ms 30ms \
  --output-dir output/robot_emotions_virtual_imu \
  --fps-target 20
```

Flags úteis:

- `--clip-id ...`
- `--sensor-layout-path ...`
- `--imu-random-seed 0`
- `--real-imu-reference-path ...`
- `--real-imu-label-key emotion`
- `--real-imu-signal-mode acc`
- `--estimate-sensor-frame`
- `--estimate-sensor-names waist head left_forearm right_forearm`

Saídas principais:

- `virtual_imu_manifest.jsonl`
- `virtual_imu_summary.json`
- por clipe: `ik_sequence.npz`, `virtual_imu.npz`, relatórios de qualidade e artefatos auxiliares

## 10. Smoke tests úteis

### 10.1. Validar que o backend por prompt consegue gerar pose

```bash
.venv/bin/python -c "from pose_module.prompt_source.backend import LegacyT2MGPTBackend; backend=LegacyT2MGPTBackend(); result=backend.generate(prompt_text='a person walks with light arm swings', seed=0, fps=20.0); print(result['joint_positions_xyz'].shape)"
```

### 10.2. Validar os testes da fase 1

```bash
.venv/bin/python -m pytest tests/test_prompt_pose_pipeline.py tests/test_robot_emotions_prompt_exports.py -q
```

## 11. Ordem recomendada para usar a feature nova

Se o objetivo for usar o ramo por prompt da fase 1:

1. garantir que a `.venv` e os artefatos estejam prontos
2. opcionalmente gerar `pose3d_manifest.jsonl` real com `export-pose3d`
3. rodar `build-prompt-catalog`
4. rodar `export-prompt-pose3d`

Se o objetivo for usar o pipeline por vídeo completo:

1. `export-imu`
2. `export-pose2d`
3. `export-pose3d`
4. `export-virtual-imu`

## 12. Limites atuais

No estado atual do código:

- `prompt -> pose3d` está implementado e validado
- `prompt -> virtual_imu` ainda não foi exposto na CLI
- a avaliação multisource da fase 3 ainda não está pronta

Em outras palavras:

- a fase 1 está operacional
- o caminho por vídeo segue operacional
- o próximo passo natural é ligar o `PoseSequence3D` sintético ao downstream de `IK` e `IMUSim`

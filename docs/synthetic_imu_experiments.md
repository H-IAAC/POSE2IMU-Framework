# Synthetic IMU Experiments

Três modos de experimento para gerar IMU sintético:

| Modo | Fonte de pose | Comando |
|------|--------------|---------|
| Apenas real | Vídeo → MotionBERT | `export-virtual-imu` (já existia) |
| Apenas sintético | Kimodo → pipeline | `export-kimodo-virtual-imu` |
| Real + sintético | Merge dos dois manifestos | `export-mixed-virtual-imu` |

---

## Pré-requisitos

- **Ambiente real** (`.venv`): `source .venv/bin/activate`
- **Ambiente Kimodo** (`kimodo`): `conda activate kimodo`

---

## Modo 1 — Apenas pose real

> Ambiente: `.venv`

```bash
.venv/bin/python -m pose_module.robot_emotions export-virtual-imu \
  --dataset-root data/RobotEmotions \
  --domains 10ms 30ms \
  --output-dir output/exp_real_pose \
  --no-debug-2d --no-debug-3d
```

Produz `output/exp_real/virtual_imu_manifest.jsonl`.

---

## Modo 2 — Apenas pose sintética (Kimodo)

> Ambiente: `kimodo`

Requer um manifesto de geração Kimodo (saída de `generate-kimodo`).

```bash
python -m robot_emotions_vlm export-kimodo-virtual-imu \
  --kimodo-manifest output/robot_emotions_kimodo_generated_single_5/kimodo_generation_manifest.jsonl \
  --output-dir output/exp_synthetic_pose_5
```

Produz `output/exp_synthetic/virtual_imu_manifest.jsonl`.

**Recomendado:** use `--real-imu-root` para aplicar os mesmos alinhamento geométrico e
calibração por percentil do ramo real. Sem isso, o IMU sintético fica no frame do SMPL-X
(gravity em Y), enquanto o sensor físico tem montagem com inclinação diferente — gerando
offsets DC grotescos na comparação direta.

```bash
python -m robot_emotions_vlm export-kimodo-virtual-imu \
  --kimodo-manifest output/robot_emotions_kimodo_generated_5/kimodo_generation_manifest.jsonl \
  --output-dir output/exp_synthetic_pose \
  --real-imu-root output/exp_real_pose \
  --real-imu-signal-mode acc \
  --real-imu-label-key emotion
```

Opções úteis:

```
--real-imu-root DIR              Raiz do IMU real; ativa alinhamento + calibração por clipe
--real-imu-signal-mode acc|gyro|both  Sinal calibrado (padrão: acc)
--real-imu-percentile-resolution N    Bins de percentil (padrão: 100)
--no-real-imu-per-class-calibration   Usa distribuição global em vez de por classe
--real-imu-label-key FIELD            Campo para calibração por classe (ex: emotion)
--clip-id CLIP_ID                Processar apenas um clipe específico
--export-bvh                     Exportar também pose3d.bvh
--no-skip-existing               Reprocessar mesmo se virtual_imu.npz já existir
--imu-acc-noise-std-m-s2 N       Ruído gaussiano no acelerômetro (m/s²)
--imu-gyro-noise-std-rad-s N     Ruído gaussiano no giroscópio (rad/s)
--imu-random-seed N              Semente para ruído
```

Artefatos adicionais por clipe quando `--real-imu-root` é usado:

```
virtual_imu/
├── virtual_imu.npz                         # sinal final alinhado + calibrado
├── virtual_imu_geometric_aligned.npz       # após alinhamento, antes da calibração
├── imu_alignment_transforms.json
├── imu_alignment_metrics.json
└── virtual_imu_calibration_report.json
```

---

## Modo 3 — Real + sintético

> Ambiente: `kimodo` (só para o merge; cada manifesto pode ter sido gerado separadamente)

Requer os dois manifestos já gerados pelos modos 1 e 2.

```bash
python -m robot_emotions_vlm export-mixed-virtual-imu \
  --real-manifest      output/exp_real_pose/virtual_imu_manifest.jsonl \
  --synthetic-manifest output/exp_synthetic_pose/virtual_imu_manifest.jsonl \
  --output-dir         output/exp_mixed_pose
```

Produz `output/exp_mixed/mixed_virtual_imu_manifest.jsonl` com campo `"pose_kind": "real"|"synthetic"` em cada entrada.

---

## Suavização temporal aplicada ao Kimodo

O pipeline Kimodo aplica as mesmas etapas do pipeline de vídeo antes do IK + IMUSim:

1. `run_metric_normalizer` — escala métrica + **Savitzky-Golay** (window=9, poly=2) nas posições locais
2. `run_root_trajectory_estimator` — **Savitzky-Golay** (window=9, poly=2) na trajetória global da raiz

---

## Artefatos por clipe

```
<output-dir>/<prompt_id>/
├── pose3d_kimodo_raw.npz       # pose bruta do Kimodo
├── pose3d_metric_local.npz     # após normalização métrica
├── pose3d.npz                  # pose final com raiz global
├── quality_report.json
├── ik/                         # rotações locais + offsets
└── virtual_imu/
    └── virtual_imu.npz         # sinal IMU sintético (acc + gyro)
```

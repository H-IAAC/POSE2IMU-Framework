import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pose_module.model_registry import (
    DEFAULT_MOTIONBERT_MODEL_ID,
    DEFAULT_PERSON_DETECTOR_MODEL_ID,
    DEFAULT_VITPOSE_MODEL_ID,
    required_download_specs,
    resolve_local_motionbert_artifacts,
    resolve_local_pose2d_backend_artifacts,
)


class ModelRegistryTests(unittest.TestCase):
    def test_resolve_local_pose2d_backend_artifacts_uses_repo_checkpoint_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoints_dir = Path(tmp_dir)
            (checkpoints_dir / f"{DEFAULT_VITPOSE_MODEL_ID}.py").write_text("# vitpose\n", encoding="utf-8")
            (checkpoints_dir / f"{DEFAULT_VITPOSE_MODEL_ID}-weights.pth").touch()
            (checkpoints_dir / f"{DEFAULT_PERSON_DETECTOR_MODEL_ID}.py").write_text("# detector\n", encoding="utf-8")
            (checkpoints_dir / f"{DEFAULT_PERSON_DETECTOR_MODEL_ID}_weights.pth").touch()

            with patch("pose_module.model_registry.CHECKPOINTS_DIR", checkpoints_dir):
                artifacts = resolve_local_pose2d_backend_artifacts("vitpose-b")

            self.assertEqual(artifacts.pose2d_model_id, DEFAULT_VITPOSE_MODEL_ID)
            self.assertEqual(artifacts.detector_model_id, DEFAULT_PERSON_DETECTOR_MODEL_ID)
            self.assertEqual(artifacts.pose2d_config_path, (checkpoints_dir / f"{DEFAULT_VITPOSE_MODEL_ID}.py").resolve())
            self.assertEqual(
                artifacts.detector_config_path,
                (checkpoints_dir / f"{DEFAULT_PERSON_DETECTOR_MODEL_ID}.py").resolve(),
            )

    def test_resolve_local_motionbert_artifacts_accepts_checkpoint_filename_not_matching_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoints_dir = Path(tmp_dir)
            (checkpoints_dir / f"{DEFAULT_MOTIONBERT_MODEL_ID}.py").write_text("# motionbert\n", encoding="utf-8")
            expected_checkpoint = checkpoints_dir / "motionbert_ft_h36m-d80af323_20230531.pth"
            expected_checkpoint.touch()

            with patch("pose_module.model_registry.CHECKPOINTS_DIR", checkpoints_dir):
                artifacts = resolve_local_motionbert_artifacts()

            self.assertEqual(artifacts.model_id, DEFAULT_MOTIONBERT_MODEL_ID)
            self.assertEqual(artifacts.config_path, (checkpoints_dir / f"{DEFAULT_MOTIONBERT_MODEL_ID}.py").resolve())
            self.assertEqual(artifacts.checkpoint_path, expected_checkpoint.resolve())

    def test_required_download_specs_validate_manifest_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "mmpose_models.txt"
            manifest_path.write_text(
                "\n".join(
                    [
                        DEFAULT_VITPOSE_MODEL_ID,
                        DEFAULT_MOTIONBERT_MODEL_ID,
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("pose_module.model_registry.MMPOSE_MODELS_FILE", manifest_path):
                specs = required_download_specs()

            self.assertEqual(
                [(spec.package_name, spec.model_id) for spec in specs],
                [
                    ("mmpose", DEFAULT_VITPOSE_MODEL_ID),
                    ("mmpose", DEFAULT_MOTIONBERT_MODEL_ID),
                    ("mmdet", DEFAULT_PERSON_DETECTOR_MODEL_ID),
                ],
            )


if __name__ == "__main__":
    unittest.main()

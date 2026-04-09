import unittest

from pose_module.prompt_source.backend import LegacyT2MGPTBackendConfig, _resolve_target_length


class PromptBackendDurationTests(unittest.TestCase):
    def test_resolve_target_length_uses_duration_hint_with_cap(self) -> None:
        config = LegacyT2MGPTBackendConfig()

        target = _resolve_target_length(
            fps=20.0,
            duration_hint_sec=20.0,
            config=config,
        )

        self.assertEqual(target["frame_factor"], 4)
        self.assertEqual(target["max_supported_frames"], 196)
        self.assertEqual(target["target_frames"], 196)
        self.assertEqual(target["target_tokens"], 49)

    def test_resolve_target_length_rounds_and_ceilings_tokens(self) -> None:
        config = LegacyT2MGPTBackendConfig()

        target = _resolve_target_length(
            fps=20.0,
            duration_hint_sec=5.05,
            config=config,
        )

        self.assertEqual(target["target_frames"], 101)
        self.assertEqual(target["target_tokens"], 26)

    def test_resolve_target_length_skips_for_missing_hint(self) -> None:
        config = LegacyT2MGPTBackendConfig()

        target = _resolve_target_length(
            fps=20.0,
            duration_hint_sec=None,
            config=config,
        )

        self.assertIsNone(target["target_frames"])
        self.assertEqual(target["target_tokens"], 0)
        self.assertEqual(target["max_supported_frames"], 196)

    def test_resolve_target_length_rejects_non_positive_values(self) -> None:
        config = LegacyT2MGPTBackendConfig()

        with self.assertRaisesRegex(ValueError, "fps > 0"):
            _resolve_target_length(
                fps=0.0,
                duration_hint_sec=5.0,
                config=config,
            )

        with self.assertRaisesRegex(ValueError, "duration_hint_sec > 0"):
            _resolve_target_length(
                fps=20.0,
                duration_hint_sec=0.0,
                config=config,
            )

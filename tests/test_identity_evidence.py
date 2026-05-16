import base64
import copy
import hashlib
import hmac
import os
import sys
import unittest
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "go2control"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from go2_client import (  # noqa: E402
    AGiXTVoiceClient,
    DEFAULT_CONFIG,
    _evidence_envelope_binding,
    _sha256_hex,
    _utc_now_rfc3339,
    load_config,
    sign_evidence_hmac,
)


class DummyRobot:
    robot_config = {"connection": "dds"}


class IdentityEvidenceTests(unittest.TestCase):
    def test_binding_matches_workconductor_v3_order(self):
        request = {
            "machine_id": "machine-1",
            "key_id": "key-1",
            "conversation_id": "conversation-1",
            "stream_id": "go2-audio",
            "sequence_number": 7,
            "captured_at": "2026-05-15T10:00:00+00:00",
            "sent_at": "2026-05-15T10:00:01+00:00",
            "previous_payload_sha256": "previous",
            "transport_format": "pcm_audio",
            "content_type": "audio/wav",
            "codec": "wav",
            "sample_rate_hz": 16000,
            "channels": 1,
            "duration_ms": 1200,
            "evidence_profile": "normal",
            "metadata": {
                "challenge_id": "challenge-1",
                "nonce": "nonce-1",
                "command_id": "command-1",
                "action_type": "robot_action",
                "command_name": "dance1",
            },
        }
        payload_hash = _sha256_hex(b"voice-bytes")

        expected = (
            "v3|company-1|machine-1|key-1|conversation-1|go2-audio|7|"
            "2026-05-15T10:00:00+00:00|2026-05-15T10:00:01+00:00|"
            f"{payload_hash}|previous|pcm_audio|audio/wav|wav|||16000|1|1200|"
            "normal|||challenge-1|nonce-1|command-1|robot_action|dance1"
        )
        binding = _evidence_envelope_binding("company-1", request, payload_hash)
        self.assertEqual(binding, expected)

        expected_signature = hmac.new(
            b"shared-secret", expected.encode(), hashlib.sha256
        ).hexdigest()
        self.assertEqual(
            sign_evidence_hmac("company-1", request, payload_hash, "shared-secret"),
            expected_signature,
        )

    def test_build_identity_evidence_sequences_and_chains_hashes(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["agixt"]["conversation_id"] = "conversation-1"
        config["identity_evidence"].update(
            {
                "company_id": "company-1",
                "machine_id": "machine-1",
                "key_id": "key-1",
                "signing_secret": "shared-secret",
            }
        )
        client = AGiXTVoiceClient(config, DummyRobot())

        first_payload = b"first-frame"
        first = client._build_identity_evidence(
            method_type="face",
            stream_id="go2-camera",
            transport_format="jpeg_frame",
            payload=first_payload,
            content_type="image/jpeg",
            codec="jpeg",
            captured_at="2026-05-15T10:00:00+00:00",
            width=640,
            height=480,
        )

        self.assertEqual(first["sequence_number"], 1)
        self.assertIsNone(first["previous_payload_sha256"])
        self.assertEqual(first["payload_sha256"], _sha256_hex(first_payload))
        self.assertEqual(first["data_base64"], base64.b64encode(first_payload).decode())
        self.assertEqual(first["device_class"], "go2_robot_camera")
        self.assertEqual(first["metadata"]["capture_client"], "go2control")
        self.assertEqual(first["metadata"]["robot_connection"], "dds")
        self.assertIn("resource_pressure", first["metadata"])
        self.assertIn("transport_pressure", first["metadata"])
        self.assertEqual(
            first["signature"],
            sign_evidence_hmac(
                "company-1", first, first["payload_sha256"], "shared-secret"
            ),
        )

        second_payload = b"second-frame"
        second = client._build_identity_evidence(
            method_type="face",
            stream_id="go2-camera",
            transport_format="jpeg_frame",
            payload=second_payload,
            content_type="image/jpeg",
            codec="jpeg",
            captured_at="2026-05-15T10:00:02+00:00",
        )

        self.assertEqual(second["sequence_number"], 2)
        self.assertEqual(second["previous_payload_sha256"], _sha256_hex(first_payload))
        self.assertEqual(
            second["signature"],
            sign_evidence_hmac(
                "company-1", second, second["payload_sha256"], "shared-secret"
            ),
        )

    def test_rfc3339_helper_uses_server_canonical_utc_offset(self):
        timestamp = _utc_now_rfc3339()
        self.assertTrue(timestamp.endswith("+00:00"))
        self.assertNotIn("Z", timestamp)

    def test_load_config_env_overrides_do_not_mutate_defaults(self):
        with mock.patch.dict(
            os.environ,
            {
                "AGIXT_EVIDENCE_SIGNING_SECRET": "temporary-secret",
                "AGIXT_COMPANY_ID": "company-1",
            },
            clear=True,
        ):
            config = load_config()

        self.assertEqual(
            config["identity_evidence"]["signing_secret"], "temporary-secret"
        )
        self.assertEqual(DEFAULT_CONFIG["identity_evidence"]["signing_secret"], "")

        with mock.patch.dict(os.environ, {}, clear=True):
            fresh_config = load_config()
        self.assertEqual(fresh_config["identity_evidence"]["signing_secret"], "")


if __name__ == "__main__":
    unittest.main()

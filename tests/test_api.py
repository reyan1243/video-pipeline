"""API tests against the real pipeline (real Grounding DINO + SAM2 models,
no fakes/mocks) — run against a 29-frame (1s) trimmed fixture clip so a real
end-to-end run finishes in a reasonable time. Requires the models to be
downloaded/cached locally (see README.md) and takes real
minutes, not milliseconds — this is an integration test, not a unit test.
"""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import app, store

FIXTURE_CLIP = Path(__file__).parent / "fixtures" / "short_clip.mp4"


class TestApi(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

        data_dir_patcher = patch("api.app.DATA_DIR", Path(self._tmpdir.name))
        data_dir_patcher.start()
        self.addCleanup(data_dir_patcher.stop)

        self.client = TestClient(app)

    def _submit(self, prompt="person", person_format="fill_matte", num_seed_candidates=3):
        with FIXTURE_CLIP.open("rb") as video_file:
            return self.client.post(
                "/jobs",
                data={
                    "prompt": prompt,
                    "person_format": person_format,
                    "num_seed_candidates": num_seed_candidates,
                },
                files={"video": ("short_clip.mp4", video_file, "video/mp4")},
            )

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_full_job_lifecycle_and_downloads(self):
        create_response = self._submit()
        self.assertEqual(create_response.status_code, 200)
        job_id = create_response.json()["job_id"]

        status_response = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(status_response.status_code, 200)
        body = status_response.json()
        self.assertEqual(body["status"], "done", msg=body.get("error"))
        self.assertEqual(body["person_url"], f"/jobs/{job_id}/download/person")
        self.assertEqual(body["matte_url"], f"/jobs/{job_id}/download/matte")
        self.assertEqual(body["background_url"], f"/jobs/{job_id}/download/background")

        person_download = self.client.get(body["person_url"])
        self.assertEqual(person_download.status_code, 200)
        self.assertGreater(len(person_download.content), 0)

        matte_download = self.client.get(body["matte_url"])
        self.assertEqual(matte_download.status_code, 200)
        self.assertGreater(len(matte_download.content), 0)

        background_download = self.client.get(body["background_url"])
        self.assertEqual(background_download.status_code, 200)
        # background is a byte-identical copy of the uploaded fixture
        self.assertEqual(background_download.content, FIXTURE_CLIP.read_bytes())

        unknown_artifact = self.client.get(f"/jobs/{job_id}/download/nonsense")
        self.assertEqual(unknown_artifact.status_code, 404)

    def test_failed_job_reports_error(self):
        # An unreadable "video" fails fast in VideoSource.load_metadata(),
        # before any model inference runs — deterministic, unlike trying to
        # pick a prompt with "nothing to detect": box_threshold=0.2 turned
        # out lenient enough that even "a flying saucer" still produced some
        # low-confidence match somewhere in the frame, so the job completed
        # instead of failing. A corrupt upload is a real failure mode anyway.
        create_response = self.client.post(
            "/jobs",
            data={"prompt": "person", "person_format": "fill_matte", "num_seed_candidates": 3},
            files={"video": ("corrupt.mp4", io.BytesIO(b"not a real video"), "video/mp4")},
        )
        job_id = create_response.json()["job_id"]

        status_response = self.client.get(f"/jobs/{job_id}")
        body = status_response.json()
        self.assertEqual(body["status"], "failed")
        self.assertIn("could not open video", body["error"])

    def test_unknown_job_404(self):
        response = self.client.get("/jobs/does-not-exist")
        self.assertEqual(response.status_code, 404)

    def test_download_before_done_is_409(self):
        # Pure API-layer routing check — construct the job state directly
        # rather than pay for a real pipeline run just to see it mid-flight.
        job = store.create(
            prompt="person",
            person_format="fill_matte",
            num_seed_candidates=3,
            mask_close_kernel_size=45,
            video_path=FIXTURE_CLIP,
            output_dir=Path(self._tmpdir.name) / "unused",
        )
        response = self.client.get(f"/jobs/{job.id}/download/person")
        self.assertEqual(response.status_code, 409)

    def test_invalid_person_format_rejected(self):
        response = self.client.post(
            "/jobs",
            data={"prompt": "person", "person_format": "bogus"},
            files={"video": ("clip.mp4", io.BytesIO(b"x"), "video/mp4")},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()

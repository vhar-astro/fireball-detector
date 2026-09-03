from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fireball_edge.bundles import EventBundle, discover_bundle
from fireball_edge.contracts import CANDIDATE_EXTRACTOR, SCHEMA_VERSION
from fireball_edge.offline.manifest import build_records, load_expert_labels, write_manifest
from fireball_edge.offline.ufocapture import (
    BundlePreflightError,
    MediaInfo,
    PreflightBundle,
    UfoCaptureRecord,
    UfoCaptureMetadataError,
    parse_ufocapture_record,
    preflight_bundle,
    validate_xml_provenance,
)


def _xml(*, timestamp: str = "2026-09-03T11:59:59.500", width: int = 1920, height: int = 1080) -> str:
    return (
        '<ufocapture_record timestamp="' + timestamp + '" tz="10800" '
        f'width="{width}" height="{height}" fps="30" frame_count="100" '
        'codec="MJPG" dropped_frames="0" countrycode="PL" lid="12" sid="34" '
        'cam="cam-a" lens="f12" />'
    )


class UfoCaptureXmlV2Tests(unittest.TestCase):
    def test_decimal_seconds_timezone_and_noon_boundary_are_grouping_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "event.xml"
            path.write_text(_xml(), encoding="utf-8")
            record = parse_ufocapture_record(path)
        self.assertEqual("2026-09-03T11:59:59.500000+03:00", record.timestamp)
        self.assertEqual("PL:12:34", record.station)
        self.assertEqual("PL:12:34:cam-a:f12", record.camera)
        self.assertEqual("2026-09-02", record.camera_night)
        from fireball_edge.inference import CALIBRATION_FEATURES

        self.assertTrue(
            {"station", "camera", "countrycode", "lid", "sid", "cam", "lens", "timezone_seconds"}
            .isdisjoint(CALIBRATION_FEATURES)
        )

    def test_malformed_or_unbounded_xml_is_rejected_and_runtime_only_marks_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad.xml"
            path.write_text("<ufo>", encoding="utf-8")
            with self.assertRaisesRegex(UfoCaptureMetadataError, "malformed XML"):
                parse_ufocapture_record(path)
            self.assertEqual("malformed", validate_xml_provenance(path))
            path.write_bytes(b"<!DOCTYPE ufo [<!ENTITY x 'x'>]><ufocapture_record />")
            with self.assertRaisesRegex(UfoCaptureMetadataError, "DTD"):
                parse_ufocapture_record(path)

    def test_conventional_component_time_and_cx_cy_attributes_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "event.xml"
            path.write_text(
                '<ufocapture_record y="2026" mo="9" d="3" h="12" m="0" '
                's="00.250000" tz="10800" cx="1920" cy="1080" fps="25" '
                'frames="61" fourcc="MJPG" drop="-1" countrycode="PL" '
                'lid="12" sid="34" cam="cam-a" lens="f12"/>',
                encoding="utf-8",
            )
            record = parse_ufocapture_record(path)
        self.assertEqual("2026-09-03T12:00:00.250000+03:00", record.timestamp)
        self.assertEqual("2026-09-03", record.camera_night)
        self.assertEqual((1920, 1080, 61, -1), (
            record.width,
            record.height,
            record.frame_count,
            record.dropped_frames,
        ))


class BundlePreflightV2Tests(unittest.TestCase):
    def _bundle_files(self, root: Path) -> Path:
        base = root / "M20260903_010203_CAM01"
        base.with_suffix(".avi").write_bytes(b"not decoded in mocked test")
        (root / f"{base.name}P.bmp").write_bytes(b"corrupt bmp")
        (root / f"{base.name}P.jpg").write_bytes(b"valid jpg in mocked test")
        (root / f"{base.name}M.bmp").write_bytes(b"provenance only")
        base.with_suffix(".xml").write_text(_xml(), encoding="utf-8")
        return base

    def test_bmp_is_preferred_but_corrupt_bmp_falls_back_to_jpg(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = self._bundle_files(root)
            bundle = discover_bundle(base)
            self.assertEqual(f"{base.name}P.bmp", bundle.stack_image.name)
            with (
                mock.patch(
                    "fireball_edge.offline.ufocapture._avi_info",
                    return_value=MediaInfo(1920, 1080, 30.0, 100, "MJPG"),
                ),
                mock.patch(
                    "fireball_edge.offline.ufocapture._stack_info",
                    side_effect=[
                        BundlePreflightError("cannot read stack image: corrupt"),
                        MediaInfo(1920, 1080, None, None, None),
                    ],
                ),
            ):
                checked = preflight_bundle(bundle)
        self.assertEqual(f"{base.name}P.jpg", checked.bundle.stack_image.name)
        self.assertEqual("provenance_only", checked.as_dict()["star_mask_role"])

    def test_training_rejects_a_missing_p_stack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = self._bundle_files(root)
            (root / f"{base.name}P.bmp").unlink()
            (root / f"{base.name}P.jpg").unlink()
            with self.assertRaisesRegex(BundlePreflightError, "P stack"):
                preflight_bundle(discover_bundle(base))

    def test_wrong_geometry_bmp_falls_back_to_valid_jpg(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = self._bundle_files(root)
            with (
                mock.patch(
                    "fireball_edge.offline.ufocapture._avi_info",
                    return_value=MediaInfo(1920, 1080, 30.0, 100, "MJPG"),
                ),
                mock.patch(
                    "fireball_edge.offline.ufocapture._stack_info",
                    side_effect=[
                        MediaInfo(100, 100, None, None, None),
                        MediaInfo(1920, 1080, None, None, None),
                    ],
                ),
            ):
                checked = preflight_bundle(discover_bundle(base))
        self.assertEqual(f"{base.name}P.jpg", checked.bundle.stack_image.name)

    def test_geometry_mismatch_is_fatal_but_metadata_disagreement_is_a_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = self._bundle_files(root)
            bundle = discover_bundle(base)
            with (
                mock.patch(
                    "fireball_edge.offline.ufocapture._avi_info",
                    return_value=MediaInfo(1919, 1080, 25.0, 99, "XVID"),
                ),
                mock.patch(
                    "fireball_edge.offline.ufocapture._stack_info",
                    return_value=MediaInfo(1920, 1080, None, None, None),
                ),
            ):
                with self.assertRaisesRegex(BundlePreflightError, "geometry"):
                    preflight_bundle(bundle)

            with (
                mock.patch(
                    "fireball_edge.offline.ufocapture._avi_info",
                    return_value=MediaInfo(1920, 1080, 30.0, 100, "MJPG"),
                ),
                mock.patch(
                    "fireball_edge.offline.ufocapture._stack_info",
                    return_value=MediaInfo(1920, 1079, None, None, None),
                ),
            ):
                with self.assertRaisesRegex(BundlePreflightError, "stack"):
                    preflight_bundle(bundle)

    def test_non_geometric_xml_media_differences_are_recorded_as_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = self._bundle_files(root)
            base.with_suffix(".xml").write_text(
                _xml().replace('dropped_frames="0"', 'dropped_frames="2"'),
                encoding="utf-8",
            )
            bundle = discover_bundle(base)
            with (
                mock.patch(
                    "fireball_edge.offline.ufocapture._avi_info",
                    return_value=MediaInfo(1920, 1080, 25.0, 99, "XVID"),
                ),
                mock.patch(
                    "fireball_edge.offline.ufocapture._stack_info",
                    side_effect=[
                        BundlePreflightError("cannot read stack image: corrupt"),
                        MediaInfo(1920, 1080, None, None, None),
                    ],
                ),
            ):
                checked = preflight_bundle(bundle)
        self.assertEqual(4, len(checked.warnings))
        self.assertTrue(any(item.startswith("frame_count differs") for item in checked.warnings))
        self.assertTrue(any(item.startswith("fps differs") for item in checked.warnings))
        self.assertTrue(any(item.startswith("codec differs") for item in checked.warnings))
        self.assertTrue(any("dropped frame" in item for item in checked.warnings))


class ManifestV2Tests(unittest.TestCase):
    def test_minimal_label_csv_and_optional_derived_assertions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            labels = root / "labels.csv"
            labels.write_text(
                "clip_base,label,physical_event_id,nuisance_tags,station,camera,night\n"
                "event,fireball,physical-1,moon_only,PL:12:34,PL:12:34:cam-a:f12,2026-09-02\n",
                encoding="utf-8",
            )
            (root / "event.avi").write_bytes(b"avi")
            (root / "eventP.bmp").write_bytes(b"stack")
            (root / "event.xml").write_bytes(b"xml")
            metadata = UfoCaptureRecord(
                timestamp="2026-09-03T11:59:59+03:00",
                timezone_seconds=10800,
                width=1920,
                height=1080,
                fps=30.0,
                frame_count=100,
                codec="MJPG",
                dropped_frames=0,
                countrycode="PL",
                lid="12",
                sid="34",
                cam="cam-a",
                lens="f12",
                station="PL:12:34",
                camera="PL:12:34:cam-a:f12",
                camera_night="2026-09-02",
            )
            expected = PreflightBundle(
                bundle=EventBundle(
                    clip_base=root / "event",
                    avi=root / "event.avi",
                    stack_image=root / "eventP.bmp",
                    star_mask=None,
                    xml=root / "event.xml",
                ),
                metadata=metadata,
                avi=MediaInfo(1920, 1080, 30.0, 100, "MJPG"),
                stack=MediaInfo(1920, 1080, None, None, None),
                warnings=(),
            )
            with mock.patch("fireball_edge.offline.manifest.preflight_bundle", return_value=expected):
                records = build_records(load_expert_labels(labels))
            destination = root / "manifest.json"
            write_manifest(destination, records)
            document = __import__("json").loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(SCHEMA_VERSION, document["schema_version"])
        self.assertEqual(CANDIDATE_EXTRACTOR, document["candidate_extractor"])
        self.assertEqual("provenance_only", document["records"][0]["star_mask_role"])

    def test_optional_csv_assertion_cannot_override_xml_grouping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            labels = root / "labels.csv"
            labels.write_text(
                "clip_base,label,physical_event_id,nuisance_tags,station\n"
                "event,fireball,physical-1,,wrong-station\n",
                encoding="utf-8",
            )
            metadata = UfoCaptureRecord(
                "2026-09-03T11:59:59+03:00", 10800, 1920, 1080, None, None, None, None,
                "PL", "12", "34", "cam-a", "f12", "PL:12:34", "PL:12:34:cam-a:f12", "2026-09-02",
            )
            checked = PreflightBundle(
                EventBundle(root / "event", root / "event.avi", root / "eventP.bmp", None, root / "event.xml"),
                metadata,
                MediaInfo(1920, 1080, None, None, None),
                MediaInfo(1920, 1080, None, None, None),
                (),
            )
            with mock.patch("fireball_edge.offline.manifest.preflight_bundle", return_value=checked):
                with self.assertRaisesRegex(ValueError, "station assertion"):
                    build_records(load_expert_labels(labels))

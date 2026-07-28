"""Tests for mantpy.fetch — all HTTP calls mocked via `responses`."""

from __future__ import annotations

import hashlib
import io
import tarfile
import textwrap

import pytest
import requests
import responses as resp_lib

import mantpy.fetch as fetch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MATRISOME_CSV = textwrap.dedent("""\
    Matrisome.Division,Matrisome.Category,Gene.Symbol,Gene.Name
    ECM Glycoproteins,Core matrisome,Fn1,Fibronectin
    ECM Glycoproteins,Core matrisome,Lamc1,Laminin subunit gamma-1
    Collagens,Core matrisome,Col1a1,Collagen type I alpha 1
    Collagens,Core matrisome,Col3a1,Collagen type III alpha 1
    Proteoglycans,Core matrisome,Vcan,Versican
    ECM Regulators,Matrisome-associated,Mmp2,Matrix metallopeptidase 2
    Secreted Factors,Matrisome-associated,Tgfb1,Transforming growth factor beta 1
""")


@pytest.fixture
def tmp_cache(tmp_path):
    return tmp_path


@pytest.fixture(autouse=True)
def _pin_synthetic_matrisome_export(monkeypatch):
    digest = hashlib.sha256(MATRISOME_CSV.encode()).hexdigest()
    for species in fetch._MATRISOME_SHA256:
        monkeypatch.setitem(fetch._MATRISOME_SHA256, species, digest)


class TestSecureFileHandling:
    @resp_lib.activate
    def test_download_does_not_forward_ambient_credentials(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ZENODO_TOKEN", "synthetic-secret")
        url = "https://zenodo.org/api/records/1/files/example/content"
        resp_lib.add(resp_lib.GET, url, body=b"ok", status=200)

        fetch._download_with_cache(url, tmp_path / "example", min_bytes=1)

        assert "Authorization" not in resp_lib.calls[0].request.headers

    def test_tar_extraction_rejects_parent_traversal_before_writing(self, tmp_path):
        archive = tmp_path / "unsafe.tar.gz"
        payload = b"not allowed"
        with tarfile.open(archive, "w:gz") as tar:
            member = tarfile.TarInfo("../escaped.txt")
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))

        with pytest.raises(RuntimeError, match="Unsafe member"):
            fetch._safe_extract_tar_archive(archive, tmp_path / "extract")
        assert not (tmp_path / "escaped.txt").exists()

    def test_tar_extraction_rejects_links(self, tmp_path):
        archive = tmp_path / "link.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            member = tarfile.TarInfo("spatial/link")
            member.type = tarfile.SYMTYPE
            member.linkname = "outside"
            tar.addfile(member)

        with pytest.raises(RuntimeError, match="Unsafe non-data member"):
            fetch._safe_extract_tar_archive(archive, tmp_path / "extract")

    @pytest.mark.parametrize("filename", ["../escape", "nested/file", r"nested\file"])
    def test_remote_manifest_filename_must_be_flat(self, tmp_path, filename):
        with pytest.raises(RuntimeError, match="Unsafe dataset filename"):
            fetch._bundle_cache_path(tmp_path, filename)


# ---------------------------------------------------------------------------
# TestFetchMatrisome
# ---------------------------------------------------------------------------


class TestFetchMatrisome:
    @resp_lib.activate
    def test_columns(self, tmp_cache):
        resp_lib.add(
            resp_lib.GET,
            fetch._MATRISOME_URLS["mouse"][0],
            body=MATRISOME_CSV,
            status=200,
        )
        df = fetch.fetch_matrisome("mouse", cache_dir=tmp_cache)
        assert list(df.columns) == ["gene_symbol", "category", "division"]

    @resp_lib.activate
    def test_uppercase(self, tmp_cache):
        resp_lib.add(resp_lib.GET, fetch._MATRISOME_URLS["mouse"][0], body=MATRISOME_CSV, status=200)
        df = fetch.fetch_matrisome("mouse", cache_dir=tmp_cache)
        assert (df["gene_symbol"] == df["gene_symbol"].str.upper()).all()

    @resp_lib.activate
    def test_no_nulls(self, tmp_cache):
        resp_lib.add(resp_lib.GET, fetch._MATRISOME_URLS["mouse"][0], body=MATRISOME_CSV, status=200)
        df = fetch.fetch_matrisome("mouse", cache_dir=tmp_cache)
        assert df["gene_symbol"].notna().all()
        assert (df["gene_symbol"] != "NAN").all()

    @resp_lib.activate
    def test_uses_cache_on_second_call(self, tmp_cache):
        resp_lib.add(resp_lib.GET, fetch._MATRISOME_URLS["mouse"][0], body=MATRISOME_CSV, status=200)
        fetch.fetch_matrisome("mouse", cache_dir=tmp_cache)
        # Second call: no mock registered — should use cached file without network
        df = fetch.fetch_matrisome("mouse", cache_dir=tmp_cache)
        assert len(df) == 7

    def test_invalid_species(self, tmp_cache):
        with pytest.raises(ValueError, match="must be one of"):
            fetch.fetch_matrisome("rat", cache_dir=tmp_cache)

    @resp_lib.activate
    def test_network_failure_message(self, tmp_cache):
        resp_lib.add(
            resp_lib.GET, fetch._MATRISOME_URLS["mouse"][0], body=requests.exceptions.ConnectionError("timeout")
        )
        with pytest.raises(RuntimeError, match="Manual"):
            fetch.fetch_matrisome("mouse", cache_dir=tmp_cache)

    @resp_lib.activate
    def test_checksum_mismatch_is_not_cached(self, tmp_cache):
        resp_lib.add(
            resp_lib.GET,
            fetch._MATRISOME_URLS["mouse"][0],
            body=MATRISOME_CSV + "unexpected,source,change\n",
            status=200,
        )

        with pytest.raises(RuntimeError, match="no verified cache"):
            fetch.fetch_matrisome("mouse", cache_dir=tmp_cache)

        assert not (tmp_cache / "mouse_matrisome_masterlist.csv").exists()

    @resp_lib.activate
    def test_human_uses_human_url(self, tmp_cache):
        # Register human URL — requesting mouse URL would raise
        resp_lib.add(resp_lib.GET, fetch._MATRISOME_URLS["human"][0], body=MATRISOME_CSV, status=200)
        df = fetch.fetch_matrisome("human", cache_dir=tmp_cache)
        assert len(df) > 0

    @resp_lib.activate
    def test_row_count(self, tmp_cache):
        resp_lib.add(resp_lib.GET, fetch._MATRISOME_URLS["mouse"][0], body=MATRISOME_CSV, status=200)
        df = fetch.fetch_matrisome("mouse", cache_dir=tmp_cache)
        assert len(df) == 7

    @resp_lib.activate
    def test_excel_style_columns(self, tmp_cache, monkeypatch):
        """CSV with Excel-style column names (spaces instead of dots) also parses."""
        csv_excel = MATRISOME_CSV.replace(
            "Matrisome.Division,Matrisome.Category,Gene.Symbol",
            "Matrisome Division,Matrisome Category,Gene Symbol",
        )
        monkeypatch.setitem(fetch._MATRISOME_SHA256, "mouse", hashlib.sha256(csv_excel.encode()).hexdigest())
        resp_lib.add(resp_lib.GET, fetch._MATRISOME_URLS["mouse"][0], body=csv_excel, status=200)
        df = fetch.fetch_matrisome("mouse", cache_dir=tmp_cache)
        assert "gene_symbol" in df.columns

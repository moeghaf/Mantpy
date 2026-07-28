"""Tests for mantpy._utils."""

from __future__ import annotations

import pytest

from mantpy._utils import contrast_text_color


class TestContrastTextColor:
    def test_dark_background_returns_white(self):
        # Deep blue Okabe-Ito anchor: luminance ~67 < threshold 140.
        assert contrast_text_color("#0072B2") == "white"

    def test_light_background_returns_black(self):
        # Tol-bright amber: luminance ~177 > threshold 140.
        assert contrast_text_color("#DDAA33") == "black"

    def test_accepts_leading_hash_or_not(self):
        assert contrast_text_color("0072B2") == contrast_text_color("#0072B2")

    def test_black_returns_white(self):
        assert contrast_text_color("#000000") == "white"

    def test_white_returns_black(self):
        assert contrast_text_color("#FFFFFF") == "black"

    def test_threshold_override(self):
        # Raise the threshold above the colour's luminance, flipping the answer.
        # #DDAA33 has Y ~177; with threshold=200, white wins.
        assert contrast_text_color("#DDAA33", threshold=200) == "white"

    def test_lowercase_hex_accepted(self):
        assert contrast_text_color("#0072b2") == "white"

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError, match="6-digit"):
            contrast_text_color("#FFF")

    def test_invalid_chars_raise(self):
        with pytest.raises(ValueError, match="invalid literal"):
            contrast_text_color("#ZZZZZZ")

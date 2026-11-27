import pytest

from scripts.t11b_pilot_russia5 import decide


@pytest.mark.parametrize("share_all, share_after, qualified, expected", [
    (0.35, 0.40, 2, "tai_tiep"),
    (0.20, 0.20, 1, "tai_tiep"),  # biên đúng 0.20 dùng >=
    (0.10, 0.25, 3, "bao_GVHD"),
    (0.19999999, 0.20, 1, "bao_GVHD"),  # biên của share_after_first_control, không làm tròn share_all
    (0.10, 0.19999999, 1, "dung"),
    (0.05, 0.05, 0, "dung"),
    (0.30, 0.30, 0, "dung"),  # share_all >= 0.20 nhưng không có đợt đạt
    (0.20, 0.50, 0, "dung"),
    (None, None, 0, "dung"),  # io_total == 0
    (0.10, None, 1, "dung"),
])
def test_decide(share_all, share_after, qualified, expected):
    assert decide(share_all, share_after, qualified) == expected

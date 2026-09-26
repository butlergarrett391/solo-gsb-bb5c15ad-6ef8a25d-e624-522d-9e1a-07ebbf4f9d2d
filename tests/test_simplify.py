"""轨迹精简（Douglas-Peucker）的单元测试。

可直接运行：python3 tests/test_simplify.py
或使用 pytest：python3 -m pytest tests/
"""
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.geo import EARTH_RADIUS_M, TrackPoint
from app.gpxutil import build_gpx, parse_gpx
from app.simplify import (
    DEFAULT_TOLERANCE_M,
    MAX_TOLERANCE_M,
    MIN_TOLERANCE_M,
    simplify_track,
)

T0 = datetime(2026, 5, 1, 8, 0, 0, tzinfo=timezone.utc)

# 测试轨迹围绕 (31.0, 121.0)：0.001° 纬度 ≈ 111.19 m，0.001° 经度 ≈ 95.32 m
A = (31.0, 121.0)


def make_seg(points):
    """points: [(lat, lon, 秒偏移), ...]，海拔固定 500"""
    return [
        TrackPoint(lat=la, lon=lo, ele=500.0, time=T0 + timedelta(seconds=s))
        for la, lo, s in points
    ]


def lon_offset_for_meters(meters, lat=31.0):
    """在纬度 lat 处，水平偏移 meters 米对应的经度增量（与后端投影公式一致）。"""
    return math.degrees(meters / (math.cos(math.radians(lat)) * EARTH_RADIUS_M))


def test_straight_line_reduced_to_endpoints():
    """共线点全部落在阈值内：只保留首尾两点，且为原始点对象。"""
    seg = make_seg([(31.0 + i * 0.0001, 121.0, i * 10) for i in range(11)])
    res = simplify_track([seg], 10.0)
    out = res.simplified_segments[0]
    assert len(out) == 2
    assert out[0] is seg[0] and out[-1] is seg[-1]
    assert res.points_before == 11 and res.points_after == 2


def test_max_deviation_point_kept():
    """中间点横向偏移约 95 m 超过阈值：选中该点，结果为 首-偏差点-尾。"""
    seg = make_seg([
        (A[0], A[1], 0),
        (31.0005, 121.001, 5),   # 距经线约 95.3 m
        (31.001, 121.0, 10),
    ])
    res = simplify_track([seg], 10.0)
    out = res.simplified_segments[0]
    assert len(out) == 3
    assert out[0] is seg[0] and out[1] is seg[1] and out[2] is seg[2]


def test_below_threshold_not_split():
    """最大偏差 9.5 m 不超过阈值 10 m：不拆分，只留首尾。"""
    dlon = lon_offset_for_meters(9.5)
    seg = make_seg([
        (A[0], A[1], 0),
        (31.0005, 121.0 + dlon, 5),
        (31.001, 121.0, 10),
    ])
    res = simplify_track([seg], 10.0)
    assert len(res.simplified_segments[0]) == 2


def test_above_threshold_split():
    """最大偏差 10.5 m 超过阈值 10 m：拆分为 首-偏差点-尾。"""
    dlon = lon_offset_for_meters(10.5)
    seg = make_seg([
        (A[0], A[1], 0),
        (31.0005, 121.0 + dlon, 5),
        (31.001, 121.0, 10),
    ])
    res = simplify_track([seg], 10.0)
    out = res.simplified_segments[0]
    assert len(out) == 3
    assert out[1] is seg[1]


def test_closed_loop_point_to_endpoint_distance():
    """首尾重合（环线）：偏差按点到端点距离处理，距端点最远的点被选中保留。"""
    loop = [
        (A[0], A[1], 0),          # 首
        (31.0, 121.001, 10),      # 距首点约 95.3 m
        (31.001, 121.001, 20),    # 距首点约 146.5 m（最远）
        (31.001, 121.0, 30),      # 距首点约 111.2 m
        (A[0], A[1], 40),         # 尾（与首重合）
    ]
    seg = make_seg(loop)
    res = simplify_track([seg], 10.0)
    out = res.simplified_segments[0]
    # 所有中间点到重合端点的距离都超过 10 m，且拆分后子段偏差仍超阈值：全部保留
    assert out == seg
    assert out[0] is seg[0] and out[-1] is seg[-1]


def test_closed_loop_far_threshold_keeps_only_endpoints():
    """首尾重合且所有点到端点距离都在阈值内：只保留重合的首尾两点。"""
    dlon = lon_offset_for_meters(50.0)
    seg = make_seg([
        (A[0], A[1], 0),
        (31.0, 121.0 + dlon, 10),          # 距首点 50 m
        (31.0 + 0.0001, 121.0 + dlon, 20), # 距首点约 51.2 m
        (A[0], A[1], 30),
    ])
    res = simplify_track([seg], 60.0)
    out = res.simplified_segments[0]
    assert len(out) == 2
    assert out[0] is seg[0] and out[-1] is seg[-1]


def test_single_point_segment():
    """单点段：原样保留。"""
    seg = make_seg([(A[0], A[1], 0)])
    res = simplify_track([seg], 10.0)
    out = res.simplified_segments[0]
    assert len(out) == 1 and out[0] is seg[0]
    assert res.points_before == 1 and res.points_after == 1


def test_two_point_segment():
    """两点段：没有中间点，原样保留。"""
    seg = make_seg([(A[0], A[1], 0), (31.001, 121.0, 10)])
    res = simplify_track([seg], 10.0)
    assert res.simplified_segments[0] == seg


def test_duplicate_points():
    """重复点（零长度弦）：不崩溃，偏差为 0 时只留首尾两个重合点。"""
    seg = make_seg([(A[0], A[1], i * 10) for i in range(5)])
    res = simplify_track([seg], 10.0)
    out = res.simplified_segments[0]
    assert len(out) == 2
    assert out[0] is seg[0] and out[-1] is seg[-1]


def test_kept_points_preserve_original_fields():
    """保留点直接引用原始点对象：经纬度、海拔、时间原样不变。"""
    seg = make_seg([
        (A[0], A[1], 0),
        (31.0005, 121.001, 5),
        (31.001, 121.0, 10),
    ])
    res = simplify_track([seg], 10.0)
    out = res.simplified_segments[0]
    for orig, kept in zip(seg, out):
        assert kept is orig
        assert kept.lat == orig.lat and kept.lon == orig.lon
        assert kept.ele == 500.0
        assert kept.time == orig.time


def test_segments_independent_no_cross_connection():
    """不跨段连线：两个 trkseg 各自精简，结果分段一一对应、互不合并。"""
    seg1 = make_seg([(31.0 + i * 0.0001, 121.0, i * 10) for i in range(5)])
    seg2 = make_seg([
        (31.01, 121.0, 100),
        (31.0105, 121.001, 110),  # 横向偏移约 95.3 m
        (31.011, 121.0, 120),
    ])
    res = simplify_track([seg1, seg2], 10.0)
    assert len(res.simplified_segments) == 2
    out1, out2 = res.simplified_segments
    assert len(out1) == 2            # 直线段精简为首尾
    assert len(out2) == 3            # 含跳变段保留中间点
    assert out1[-1] is seg1[-1]      # 第一段末尾不与第二段拼合
    assert out2[0] is seg2[0]
    assert res.points_before == 8 and res.points_after == 5


def test_original_segments_not_mutated():
    """精简不修改上传的原始轨迹：输入分段长度与内容保持不变。"""
    seg = make_seg([
        (A[0], A[1], 0),
        (31.0005, 121.001, 5),
        (31.001, 121.0, 10),
    ])
    snapshot = list(seg)
    simplify_track([seg], 10.0)
    assert seg == snapshot
    assert len(seg) == 3


def test_tolerance_validation():
    """阈值须在 1～100 米之间：越界抛 ValueError，边界值可用。"""
    seg = make_seg([(A[0], A[1], 0), (31.001, 121.0, 10), (31.002, 121.0, 20)])
    for bad in (0.0, 0.5, 100.5, 101.0):
        try:
            simplify_track([seg], bad)
        except ValueError as exc:
            assert "1～100" in str(exc)
        else:
            raise AssertionError(f"阈值 {bad} 应报明确错误")
    assert MIN_TOLERANCE_M == 1.0 and MAX_TOLERANCE_M == 100.0
    assert DEFAULT_TOLERANCE_M == 10.0
    simplify_track([seg], MIN_TOLERANCE_M)
    simplify_track([seg], MAX_TOLERANCE_M)


def test_large_straight_input_no_recursion_error():
    """十万点共线输入：迭代实现不递归，精简为两点且不栈溢出。"""
    n = 100_000
    seg = make_seg([(31.0 + i * 1e-6, 121.0, i) for i in range(n)])
    res = simplify_track([seg], 10.0)
    out = res.simplified_segments[0]
    assert len(out) == 2
    assert out[0] is seg[0] and out[-1] is seg[-1]
    assert res.points_before == n and res.points_after == 2


def test_deep_split_no_recursion_error():
    """锯齿轨迹每层只剥离一个点（拆分深度约 1500，超过默认递归上限）：
    迭代实现全部保留且不栈溢出。"""
    n = 3000
    dlon = lon_offset_for_meters(60.0)  # 横向 ±60 m，偏差约 120 m > 阈值 10 m
    seg = make_seg([
        (31.0 + i * 0.0001, 121.0 + (dlon if i % 2 == 0 else -dlon), i * 10)
        for i in range(n)
    ])
    res = simplify_track([seg], 10.0)
    out = res.simplified_segments[0]
    assert len(out) == n
    assert out[0] is seg[0] and out[-1] is seg[-1]


def test_large_wavy_input_kept_points_subset():
    """两万点正弦轨迹：精简后点数远小于原始，首尾保留，保留点为原始序列子序列。"""
    n = 20_000
    amp = lon_offset_for_meters(50.0)
    seg = make_seg([
        (31.0 + i * 1e-5, 121.0 + amp * math.sin(2 * math.pi * i / 100), i)
        for i in range(n)
    ])
    res = simplify_track([seg], 10.0)
    out = res.simplified_segments[0]
    assert 2 < len(out) < n // 10
    assert out[0] is seg[0] and out[-1] is seg[-1]
    # 子序列单调性：保留点在原始序列中的下标严格递增
    idx = [i for i, p in enumerate(seg) if any(p is q for q in out)]
    assert idx == sorted(idx)


def test_simplified_gpx_roundtrip():
    """精简结果导出 GPX 后可解析：分段数、点数、时间与海拔均与预览数据一致。"""
    seg1 = make_seg([(31.0 + i * 0.0001, 121.0, i * 10) for i in range(5)])
    seg2 = make_seg([
        (31.01, 121.0, 100),
        (31.0105, 121.001, 110),
        (31.011, 121.0, 120),
    ])
    res = simplify_track([seg1, seg2], 10.0)
    gpx_bytes = build_gpx(res.simplified_segments, "demo-simplified")
    segments, name = parse_gpx(gpx_bytes)
    assert name == "demo-simplified"
    assert len(segments) == 2
    assert [len(s) for s in segments] == [2, 3]
    assert segments[0][0].time == seg1[0].time
    assert segments[0][-1].time == seg1[-1].time
    assert segments[1][1].time == seg2[1].time
    assert segments[1][1].ele == 500.0


def _run_all():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())

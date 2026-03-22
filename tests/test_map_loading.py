from __future__ import annotations

from pathlib import Path

from ha_lmapf.io.movingai_map import load_movingai_map


def test_map_loading_tmp(tmp_path: Path) -> None:
    # 4x3 map: blocked '@' and 'T', free '.' and 'G' and 'S'
    content = "\n".join(
        [
            "type octile",
            "height 3",
            "width 4",
            "map",
            ".@..",
            ".T.G",
            "S...",
            "",
        ]
    )
    p = tmp_path / "toy.map"
    p.write_text(content, encoding="utf-8")

    md = load_movingai_map(str(p))
    assert md.width == 4
    assert md.height == 3

    # blocked at (0,1) '@' and (1,1) 'T'
    assert (0, 1) in md.blocked
    assert (1, 1) in md.blocked
    assert len(md.blocked) == 2

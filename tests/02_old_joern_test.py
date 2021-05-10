from glob import glob

import gnnproject as gp
import gnnproject.helpers.old_joern as gpj


def test_old_joern_output():
    """Test whether run_joern returns a reasonable output."""
    files = glob(str(gp.external_dir() / "devign_ffmpeg_qemu/functions/*"))
    output = gpj.run_joern_old(
        files[0],
        "devign_ffmpeg_qemu",
        "old-joern-parse",
        save=False,
    )
    assert output is not None
    assert len(output) == 2
    assert "code" in output[0].columns
    assert "start" in output[1].columns
    assert "end" in output[1].columns
    assert len(output[0]) > 3
    assert len(output[1]) > 3

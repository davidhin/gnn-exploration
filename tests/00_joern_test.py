import gnnproject as gp
import gnnproject.helpers.joern as gpj


def test_joern_output():
    """Test whether run_joern returns a reasonable output."""
    output = gpj.run_joern(
        filepath=gp.external_dir()
        / "devign_ffmpeg_qemu/functions/1_FFmpeg_973b1a6b9070e2bf17d17568cbaf4043ce931f51_0.c",
        dataset_name="devign_ffmpeg_qemu",
        save=False,
        joern_parse="joern-parse",
        joern_export="joern-export",
    )
    assert len(output) > 0, "Joern output shouldn't be zero for this file."
    assert output.count("AST:") > 20, "Should be more than 20 AST labels"
    assert output.count("DDG:") > 20, "Should be more than 20 DDG labels"
    assert output.count("CFG:") > 20, "Should be more than 20 CFG labels"

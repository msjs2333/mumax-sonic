import json

import pytest

from mumax_sonic.sources.discovery import DiscoveryResult, SimulationCandidate


def test_discover_cli_prints_json_and_passes_search_roots(monkeypatch, capsys, tmp_path):
    import mumax_sonic.__main__ as cli
    from mumax_sonic.sources import discovery

    roots = []
    candidate = SimulationCandidate(tmp_path / "demo.out", pids=(42,), evidence="running process",
                                    has_frames=True)

    def discover(search_roots):
        roots.extend(search_roots)
        return DiscoveryResult((candidate,), ("permission warning",))

    monkeypatch.setattr(discovery, "discover_simulations", discover)
    monkeypatch.setattr("sys.argv", ["mumax-sonic", "--discover", "--search-root", str(tmp_path)])
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert roots == [tmp_path]
    assert report["candidates"] == [{"directory": str(candidate.directory), "pids": [42],
                                     "evidence": "running process", "has_frames": True}]
    assert report["warnings"] == ["permission warning"]


@pytest.mark.parametrize("args", [
    ["--discover", "--attach"],
    ["--discover", "--doctor"],
])
def test_discover_cli_rejects_conflicting_modes(monkeypatch, args):
    import mumax_sonic.__main__ as cli

    monkeypatch.setattr("sys.argv", ["mumax-sonic", *args])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2

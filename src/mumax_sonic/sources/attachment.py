"""Read-only attachment to an already-producing MuMax3 directory."""
from heapq import nlargest
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread
from uuid import uuid4

from .live import LiveConfig, LiveFollower
from .mumax3_bridge import MuMax3Bridge


class AttachedSimulation:
    """Own a private bridge and follower; never write into the simulation.

    Start at a recent tail, then append. Each 4096-frame publication is rotated
    into a fresh observation segment, preserving bounded reader metadata.
    """

    def __init__(self, directory, config=LiveConfig(), *, all_material=False,
                 mask_path=None, z_index=None, time_kind='dynamics', pattern='m[0-9]*.ovf'):
        if bool(all_material) == (mask_path is not None):
            raise ValueError('Declare the material domain: all_material or mask_path')
        if z_index is not None and (type(z_index) is not int or z_index < 0):
            raise ValueError('z_index must be a nonnegative integer')
        if time_kind not in ('dynamics', 'static', 'relaxation'):
            raise ValueError('invalid time_kind')
        if '/' in pattern or '\\' in pattern or '**' in pattern:
            raise ValueError('pattern must select files directly in the source directory')
        self.directory = Path(directory).resolve()
        self.config = config
        self._options = dict(all_material=all_material, mask_path=mask_path,
                             z_index=z_index, time_kind=time_kind, pattern=pattern)
        self._stop = Event()
        self._lock = Lock()
        self._thread = None
        self._follower = None
        self._bridge_state = dict(state='waiting', reason='waiting for complete OVF files')
        self._attention = None
        self._focus = False

    def start(self):
        if self._stop.is_set():
            raise RuntimeError('attachment is closed')
        if self._thread is None:
            self._thread = Thread(target=self._run, name='sonic-attach', daemon=True)
            self._thread.start()
        return self

    def close(self):
        self._stop.set()
        with self._lock:
            if self._follower is not None:
                self._follower.close()

    def set_attention(self, attention, enabled=False):
        with self._lock:
            self._attention, self._focus = attention, enabled
            if self._follower is not None:
                self._follower.set_attention(attention, enabled)

    def with_config(self, config):
        """Create a fresh observation session; the caller closes the old one."""
        return AttachedSimulation(self.directory, config, **self._options)

    @property
    def options(self):
        return dict(self._options)

    def snapshot(self):
        with self._lock:
            follower = self._follower
            bridge = dict(self._bridge_state)
            result = follower.snapshot() if follower is not None else dict(
                view=None, state='waiting', reason='waiting for complete OVF files',
                age_s=None, updates=0, last_error=None, processing_ms=None)
        if self._stop.is_set():
            result.update(state='closed', reason='attachment closed')
        elif bridge['state'] == 'invalid':
            result.update(state='invalid', reason=bridge['reason'], last_error=bridge['reason'])
        elif not bridge.get('published_frames'):
            result.update(state='waiting', reason=bridge['reason'])
        result['attachment'] = dict(directory=str(self.directory), **bridge)
        return result

    def _start_sequence(self):
        keep = self.config.band_config.window_samples if (
            self.config.recipe == 'band' and self.config.band_config is not None) else 2
        if self._options['time_kind'] != 'dynamics':
            keep = 1  # static/relaxation snapshots have no physical dt history
        # Only filenames are inspected here; no historical frame decode or hash.
        sequences = (int(match.group(1)) for path in self.directory.glob(self._options['pattern'])
                     if (match := re.search(r'(\d+)$', path.stem)))
        recent = nlargest(keep, sequences)
        return min(recent, default=0)

    def _run(self):
        while not self._stop.is_set():
            follower = None
            try:
                with TemporaryDirectory(prefix='mumax-sonic-attach-') as scratch:
                    manifest = Path(scratch) / 'live.json'
                    with MuMax3Bridge(self.directory, manifest, entity_id='m',
                                      segment_id=uuid4().hex, start_sequence=self._start_sequence(),
                                      **self._options) as bridge:
                        follower = LiveFollower(manifest, self.config).start()
                        with self._lock:
                            self._follower = follower
                            if self._attention is not None:
                                follower.set_attention(self._attention, self._focus)
                        try:
                            while not self._stop.is_set():
                                result = bridge.poll() if self.directory.is_dir() else dict(
                                    state='invalid', reason='Output directory unavailable; waiting to reconnect',
                                    published_frames=0)
                                if result.get('reason') == 'OVF series exceeds 4096 frames':
                                    break  # new bounded observation segment, not a solver restart
                                with self._lock:
                                    self._bridge_state = result
                                if self._stop.wait(self.config.poll_interval_s):
                                    break
                        finally:
                            follower.close()
                            # The source worker must finish before its private files are removed.
                            if follower._thread is not None:
                                follower._thread.join()
            except Exception as exc:
                with self._lock:
                    self._bridge_state = dict(state='invalid', reason=str(exc), published_frames=0)
                self._stop.wait(max(.2, self.config.poll_interval_s))
            finally:
                if follower is not None:
                    follower.close()
                with self._lock:
                    if self._follower is follower:
                        self._follower = None

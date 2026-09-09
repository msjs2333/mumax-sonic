"""Auditory voice continuity, independent of physical partition identities."""
from dataclasses import replace


class AdaptiveVoiceTracker:
    def __init__(self):
        self._previous = []
        self._serial = 0

    @staticmethod
    def _role(source):
        parts = source.source_id.split(':')
        if len(parts) < 3 or parts[0] != 'adaptive' or parts[2] not in {'focus', 'background', 'overview'}:
            return None
        return source.sign, parts[2]

    def map(self, scene):
        roles = [self._role(source) for source in scene.sources]
        if scene.validity != 'valid' or not roles or any(role is None for role in roles):
            self._previous = []
            return scene
        # Greedy nearest matching is bounded by the 16-voice input contract.
        # A tree path may represent a different region after a split, so it is
        # deliberately not given precedence over geometric continuity.
        pairs = sorted(
            (sum((a-b)**2 for a, b in zip(old.position, source.position)), old.source_id, j, i)
            for j, source in enumerate(scene.sources)
            for i, (old_role, old) in enumerate(self._previous)
            if roles[j] == old_role)
        assigned, used = {}, set()
        for _, voice_id, j, i in pairs:
            if j not in assigned and i not in used:
                assigned[j] = voice_id
                used.add(i)
        rendered = []
        for j, source in enumerate(scene.sources):
            if j not in assigned:
                self._serial += 1
                assigned[j] = f'sonic-adaptive:{source.sign}:{roles[j][1]}:{self._serial}'
            rendered.append(replace(source, source_id=assigned[j]))
        self._previous = list(zip(roles, rendered))
        return replace(scene, sources=tuple(rendered))

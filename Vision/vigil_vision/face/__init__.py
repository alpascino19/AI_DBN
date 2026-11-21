"""CV1 - fața și ochii.

Hook-ul cerut de pipeline.py (CV2):
    create(cfg) -> obiect cu .process(frame) -> context.FaceOutput
Importul e leneș, ca `import vigil_vision.face.eyes` să nu încarce MediaPipe.
"""


def create(cfg=None):
    from .stack import FaceStack
    return FaceStack(cfg)

# src/utils/numpy_patch.py
"""
Compatibilidad NumPy ≥2.0 para librerías que aún llaman np.obj2sctype.
Debe importarse *antes* de importar esas librerías (p.ej. shap).
"""
import numpy as np

if not hasattr(np, "obj2sctype"):
    def _obj2sctype(obj):
        """Retorna dtype equivalente; fallback a object_."""
        try:
            return np.dtype(obj).type
        except Exception:
            return np.object_
    np.obj2sctype = _obj2sctype

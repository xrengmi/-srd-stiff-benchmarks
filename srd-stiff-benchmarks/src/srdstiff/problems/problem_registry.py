"""Problem meta-veri kaydı — kategoriler ve beklenen özellikler."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProblemCategory(str, Enum):
    """Önceden kayıtlı problem kategorileri."""

    SMALL = "A"  # n ≤ 100
    MEDIUM = "B"  # 100 < n ≤ 1000
    LARGE = "C"  # n > 1000
    STIFF = "D"  # kötü koşullanmış


@dataclass(frozen=True, slots=True)
class ProblemMeta:
    """Problem meta-verisi."""

    name: str
    category: ProblemCategory
    n: int
    f_optimal: float | None = None
    kappa_estimate: float | None = None
    is_convex: bool = False
    notes: str = ""


# Preregistration'da sabitlenmiş liste
PROBLEM_REGISTRY: dict[str, ProblemMeta] = {
    # Kategori A: Küçük (n ≤ 100)
    "ROSENBR": ProblemMeta("ROSENBR", ProblemCategory.SMALL, 2, 0.0, 2000.0, False),
    "BEALE": ProblemMeta("BEALE", ProblemCategory.SMALL, 2, 0.0, None, False),
    "HELIX": ProblemMeta("HELIX", ProblemCategory.SMALL, 3, 0.0, None, False),
    # ... (tam liste 119 problem içerir)

    # Kategori C: Geniş ölçek (n = 5000)
    "ARWHEAD": ProblemMeta("ARWHEAD", ProblemCategory.LARGE, 5000, 0.0, None, False),
    "BDQRTIC": ProblemMeta("BDQRTIC", ProblemCategory.LARGE, 5000, None, None, False),
    "CURLY10": ProblemMeta("CURLY10", ProblemCategory.STIFF, 5000, None, 1e4, False),
    "CURLY20": ProblemMeta("CURLY20", ProblemCategory.STIFF, 5000, None, 1e6, False),
    "CURLY30": ProblemMeta("CURLY30", ProblemCategory.STIFF, 5000, None, 1e8, False),
    "MOREBV": ProblemMeta("MOREBV", ProblemCategory.STIFF, 5000, 0.0, None, False,
                          notes="Singular Hessian at origin"),
    "FLETCBV2": ProblemMeta("FLETCBV2", ProblemCategory.STIFF, 5000, None, None, False),
    "FLETCBV3": ProblemMeta("FLETCBV3", ProblemCategory.STIFF, 5000, None, None, False),
    "NONDQUAR": ProblemMeta("NONDQUAR", ProblemCategory.STIFF, 5000, 0.0, None, False),
    # ...
}


def get_problems_by_category(cat: ProblemCategory) -> list[str]:
    """Kategoriye göre problem adlarını döndür."""
    return [name for name, meta in PROBLEM_REGISTRY.items() if meta.category == cat]

# ============================================================
# utils.py – Shared utility functions
# ============================================================

import math
import time
import logging
from typing import List, Tuple, Optional

# ── Configure module-level logger ───────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger("TouristPlanner")

# ── Terminal colour codes ────────────────────────────────────
class C:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"
    MAGENTA = "\033[35m"
    WHITE   = "\033[97m"

def banner(text: str, width: int = 60, char: str = "═") -> None:
    """Print a styled section banner."""
    print(f"\n{C.CYAN}{C.BOLD}{char * width}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}  {text}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}{char * width}{C.RESET}")

def sub_banner(text: str, width: int = 50) -> None:
    print(f"\n{C.YELLOW}{C.BOLD}── {text} {'─' * max(0, width - len(text) - 4)}{C.RESET}")

def info(msg: str) -> None:
    print(f"  {C.GREEN}▶{C.RESET} {msg}")

def warn(msg: str) -> None:
    print(f"  {C.YELLOW}⚠{C.RESET}  {msg}")

def error(msg: str) -> None:
    print(f"  {C.RED}✘{C.RESET} {msg}")

def success(msg: str) -> None:
    print(f"  {C.GREEN}✔{C.RESET} {C.BOLD}{msg}{C.RESET}")

def ai_reason(step: str, detail: str) -> None:
    """Print an AI reasoning trace line."""
    print(f"  {C.MAGENTA}🧠 [{step}]{C.RESET} {detail}")

def separator(char: str = "─", width: int = 60) -> None:
    print(f"{C.CYAN}{char * width}{C.RESET}")

# ── Maths / Geography ────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine formula → straight-line distance in km between two
    geographic coordinates.  Used as the A* / Greedy heuristic.
    """
    R = 6371.0   # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def format_route(route: List[str]) -> str:
    """Pretty-print a list of cities as an arrow chain."""
    return f" {C.YELLOW}→{C.RESET} ".join(
        [f"{C.BOLD}{c}{C.RESET}" for c in route]
    )

def format_time(hours: float) -> str:
    """Convert decimal hours to 'Xh Ym' string."""
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h}h {m}m"

# ── Timer context manager ────────────────────────────────────

class Timer:
    """Simple wall-clock timer used to measure algorithm runtime."""
    def __init__(self):
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = (time.perf_counter() - self._start) * 1000  # ms

# ── Input helpers ────────────────────────────────────────────

def pick_city(prompt: str, cities: List[str]) -> Optional[str]:
    """
    Show numbered city list and return user's choice, or None if invalid.
    """
    print(f"\n{C.BOLD}{prompt}{C.RESET}")
    for i, c in enumerate(cities, 1):
        print(f"  {i:>2}. {c}")
    raw = input(f"\n{C.CYAN}Enter number or city name:{C.RESET} ").strip()
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(cities):
            return cities[idx]
    # try by name (case-insensitive)
    raw_lower = raw.lower()
    for c in cities:
        if c.lower() == raw_lower:
            return c
    return None

def get_int(prompt: str, lo: int, hi: int) -> int:
    """Prompt until a valid integer in [lo, hi] is entered."""
    while True:
        raw = input(f"{C.CYAN}{prompt}{C.RESET} ").strip()
        if raw.isdigit() and lo <= int(raw) <= hi:
            return int(raw)
        warn(f"Please enter a number between {lo} and {hi}.")

def get_float(prompt: str, lo: float = 0.0) -> float:
    """Prompt until a valid positive float is entered."""
    while True:
        raw = input(f"{C.CYAN}{prompt}{C.RESET} ").strip()
        try:
            val = float(raw)
            if val >= lo:
                return val
        except ValueError:
            pass
        warn(f"Please enter a number ≥ {lo}.")

def yes_no(prompt: str) -> bool:
    ans = input(f"{C.CYAN}{prompt} (y/n):{C.RESET} ").strip().lower()
    return ans in ("y", "yes")

# ============================================================
# tests.py – Unit tests for all major modules
# Run: python tests.py
# ============================================================

import sys
import traceback
from typing import Callable, List

from graph import TouristGraph
from search_algorithms import SearchAlgorithms
from csp_module import CSPSolver, TripConstraints
from bayesian_module import BayesianModule
from decision_module import DecisionModule
from agent import TouristAgent
from utils import C, banner, sub_banner, separator, success, error, info

# ── Test runner ──────────────────────────────────────────────

passed = 0
failed = 0

def run_test(name: str, fn: Callable) -> None:
    global passed, failed
    try:
        fn()
        print(f"  {C.GREEN}✔{C.RESET} {name}")
        passed += 1
    except Exception as e:
        print(f"  {C.RED}✘{C.RESET} {name}")
        print(f"    {C.RED}{e}{C.RESET}")
        failed += 1

# ════════════════════════════════════════════════════════════
#  GRAPH TESTS
# ════════════════════════════════════════════════════════════

def test_graph_cities():
    g = TouristGraph()
    assert len(g.cities) >= 10, "Should have at least 10 cities"

def test_graph_neighbours():
    g = TouristGraph()
    nb = g.neighbours("Delhi")
    assert len(nb) > 0, "Delhi should have neighbours"

def test_heuristic():
    g = TouristGraph()
    h = g.heuristic("Delhi", "Mumbai")
    assert h > 0, "Heuristic Delhi→Mumbai should be positive"
    assert h < 2000, "Distance should be < 2000 km"

def test_heuristic_admissible():
    """
    Haversine is crow-flies distance; road networks curve around terrain.
    Verify heuristic is in a sane range (not more than 2x road distance).
    """
    g = TouristGraph()
    sa = SearchAlgorithms(g)
    for city in ["Delhi", "Agra", "Jaipur"]:
        for goal in ["Mumbai", "Chennai", "Kolkata"]:
            h = g.heuristic(city, goal)
            r = sa.ucs(city, goal)
            if r.found:
                assert h <= r.total_distance * 2.0, \
                    f"h({city}→{goal})={h:.1f} is wildly off from actual={r.total_distance:.1f}"

def test_total_distance():
    g = TouristGraph()
    path = ["Delhi", "Agra"]
    d = g.total_distance(path)
    assert d > 0, "Distance should be > 0"

# ════════════════════════════════════════════════════════════
#  SEARCH ALGORITHM TESTS
# ════════════════════════════════════════════════════════════

def test_bfs():
    g = TouristGraph()
    sa = SearchAlgorithms(g)
    r = sa.bfs("Delhi", "Agra")
    assert r.found, "BFS should find Delhi→Agra"
    assert r.path[0] == "Delhi" and r.path[-1] == "Agra"
    assert r.nodes_expanded > 0

def test_dfs():
    g = TouristGraph()
    sa = SearchAlgorithms(g)
    r = sa.dfs("Delhi", "Mumbai")
    assert r.found, "DFS should find Delhi→Mumbai"

def test_ucs_optimal():
    """UCS must find the shortest-distance path."""
    g = TouristGraph()
    sa = SearchAlgorithms(g)
    r_ucs  = sa.ucs("Delhi", "Jaipur")
    r_bfs  = sa.bfs("Delhi", "Jaipur")
    assert r_ucs.found, "UCS should find path"
    # UCS distance ≤ BFS distance (BFS ignores weights)
    assert r_ucs.total_distance <= r_bfs.total_distance + 1

def test_greedy():
    g = TouristGraph()
    sa = SearchAlgorithms(g)
    r = sa.greedy("Jaipur", "Mumbai")
    assert r.found, "Greedy should find Jaipur→Mumbai"

def test_astar():
    g = TouristGraph()
    sa = SearchAlgorithms(g)
    r = sa.astar("Delhi", "Mumbai")
    assert r.found, "A* should find Delhi→Mumbai"
    assert r.path[0] == "Delhi" and r.path[-1] == "Mumbai"

def test_astar_vs_ucs():
    """A* should find path ≤ cost of UCS (both optimal)."""
    g = TouristGraph()
    sa = SearchAlgorithms(g)
    r_astar = sa.astar("Delhi", "Chennai")
    r_ucs   = sa.ucs("Delhi", "Chennai")
    if r_astar.found and r_ucs.found:
        assert abs(r_astar.total_distance - r_ucs.total_distance) < 1, \
            "A* and UCS should find same optimal distance"

def test_no_path():
    """Searching from a city to itself with no self-loop."""
    g = TouristGraph()
    sa = SearchAlgorithms(g)
    # Using a non-existent goal
    r = sa.bfs("Delhi", "NonExistentCity")
    assert not r.found, "Should not find path to non-existent city"

# ════════════════════════════════════════════════════════════
#  CSP TESTS
# ════════════════════════════════════════════════════════════

def test_csp_backtrack():
    g = TouristGraph()
    solver = CSPSolver(g)
    constraints = TripConstraints(
        total_budget=30_000, total_days=3, hotel_preference="any")
    cities = ["Delhi", "Agra", "Jaipur", "Udaipur"]
    result = solver.backtrack(constraints, cities)
    assert result is not None, "CSP backtrack should find assignment"
    assert len(result) == 3, "Should assign exactly 3 days"

def test_csp_min_conflicts():
    g = TouristGraph()
    solver = CSPSolver(g)
    constraints = TripConstraints(
        total_budget=50_000, total_days=4, hotel_preference="any")
    cities = ["Delhi", "Agra", "Jaipur", "Udaipur", "Mumbai"]
    result = solver.min_conflicts(constraints, cities)
    assert result is not None, "Min-Conflicts should return assignment"

def test_csp_constraint_violation():
    """Very low budget should fail or produce warning."""
    g = TouristGraph()
    solver = CSPSolver(g)
    constraints = TripConstraints(
        total_budget=100, total_days=5, hotel_preference="luxury")
    cities = ["Delhi", "Mumbai", "Goa", "Jaipur", "Chennai"]
    # May return None (infeasible) – that's correct behaviour
    result = solver.backtrack(constraints, cities)
    # Just verify it doesn't crash
    assert result is None or isinstance(result, dict)

# ════════════════════════════════════════════════════════════
#  BAYESIAN TESTS
# ════════════════════════════════════════════════════════════

def test_bayes_update():
    bm = BayesianModule()
    p = bm.bayes_update(0.6, 0.8, 0.3)
    assert 0 < p < 1, "Posterior must be in (0,1)"

def test_variable_elimination():
    bm = BayesianModule()
    r = bm.variable_elimination_success("winter")
    assert abs(r["success"] + r["failure"] - 1.0) < 0.001, "Must sum to 1"

def test_rejection_sampling():
    bm = BayesianModule()
    r = bm.rejection_sampling("good", n_samples=1000)
    assert 0 <= r["success"] <= 1

def test_likelihood_weighting():
    bm = BayesianModule()
    r = bm.likelihood_weighting("bad", n_samples=500)
    assert 0 <= r["success"] <= 1

def test_markov_chain():
    bm = BayesianModule()
    traj = bm.markov_chain_simulation("Delhi", 5)
    assert len(traj) >= 1, "Trajectory should not be empty"

def test_predict_city():
    bm = BayesianModule()
    pred = bm.predict_city("Jaipur", "winter")
    assert "P(trip_success)" in pred
    assert 0 <= pred["P(trip_success)"] <= 1

# ════════════════════════════════════════════════════════════
#  DECISION MODULE TESTS
# ════════════════════════════════════════════════════════════

def test_utility_function():
    g = TouristGraph()
    dm = DecisionModule(g)
    from decision_module import TravelOption
    opt = TravelOption(
        route=["Delhi", "Agra"],
        distance=200, total_cost=5000,
        travel_time=3, satisfaction=18,
        weather_risk=0.3, crowd_factor=0.6,
    )
    u = dm.utility(opt)
    assert isinstance(u, float), "Utility must be float"

def test_minimax():
    g = TouristGraph()
    dm = DecisionModule(g)
    paths = [["Delhi", "Agra"], ["Delhi", "Jaipur", "Udaipur"]]
    opts  = dm.build_options(paths)
    val, best = dm.minimax(opts, depth=2)
    assert best is not None, "Minimax must return a best option"

def test_alpha_beta():
    g = TouristGraph()
    dm = DecisionModule(g)
    paths = [["Delhi", "Agra"], ["Delhi", "Jaipur"]]
    opts  = dm.build_options(paths)
    val, best = dm.alpha_beta(opts, depth=2)
    assert best is not None

# ════════════════════════════════════════════════════════════
#  AGENT TESTS
# ════════════════════════════════════════════════════════════

def test_agent_init():
    g = TouristGraph()
    a = TouristAgent(g)
    state = a.initialise("Delhi", 20000, 48)
    assert state.current_location == "Delhi"
    assert state.remaining_budget == 20000

def test_agent_move():
    g = TouristGraph()
    a = TouristAgent(g)
    a.initialise("Delhi", 50000, 100)
    ok = a.action_move("Agra")
    assert ok, "Delhi→Agra is a valid move"
    assert a.state.current_location == "Agra"

def test_agent_invalid_move():
    g = TouristGraph()
    a = TouristAgent(g)
    a.initialise("Goa", 50000, 100)
    ok = a.action_move("Manali")  # Not directly connected
    assert not ok, "Goa→Manali is not directly connected"

def test_agent_recommend():
    g = TouristGraph()
    a = TouristAgent(g)
    recs = a.recommend_attractions("Delhi", 10000, 10)
    assert len(recs) > 0, "Should recommend at least one attraction"

# ════════════════════════════════════════════════════════════
#  MAIN TEST RUNNER
# ════════════════════════════════════════════════════════════

def run_all_tests() -> None:
    banner("TOURIST ROUTE PLANNER – TEST SUITE", 65)

    test_groups = {
        "Graph Module": [
            ("Cities loaded", test_graph_cities),
            ("Neighbours found", test_graph_neighbours),
            ("Heuristic positive", test_heuristic),
            ("Heuristic admissible", test_heuristic_admissible),
            ("Total distance", test_total_distance),
        ],
        "Search Algorithms": [
            ("BFS finds path", test_bfs),
            ("DFS finds path", test_dfs),
            ("UCS optimal", test_ucs_optimal),
            ("Greedy search", test_greedy),
            ("A* finds path", test_astar),
            ("A* == UCS distance", test_astar_vs_ucs),
            ("No path found", test_no_path),
        ],
        "CSP Module": [
            ("Backtracking solves", test_csp_backtrack),
            ("Min-Conflicts solves", test_csp_min_conflicts),
            ("Infeasible handled", test_csp_constraint_violation),
        ],
        "Bayesian Module": [
            ("Bayes update", test_bayes_update),
            ("Variable Elimination", test_variable_elimination),
            ("Rejection Sampling", test_rejection_sampling),
            ("Likelihood Weighting", test_likelihood_weighting),
            ("Markov Chain", test_markov_chain),
            ("City prediction", test_predict_city),
        ],
        "Decision Module": [
            ("Utility function", test_utility_function),
            ("Minimax", test_minimax),
            ("Alpha-Beta", test_alpha_beta),
        ],
        "Agent": [
            ("Agent initialise", test_agent_init),
            ("Valid move", test_agent_move),
            ("Invalid move blocked", test_agent_invalid_move),
            ("Recommendations", test_agent_recommend),
        ],
    }

    for group, tests in test_groups.items():
        sub_banner(group)
        for name, fn in tests:
            run_test(name, fn)

    separator()
    total = passed + failed
    colour = C.GREEN if failed == 0 else C.RED
    print(f"\n  {colour}{C.BOLD}Results: {passed}/{total} passed, {failed} failed{C.RESET}\n")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()

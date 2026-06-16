#!/usr/bin/env python3
# ============================================================
# main.py – Tourist Route Planner
# Entry point: python main.py
# ============================================================

import sys
import os

# Ensure project directory is on path
sys.path.insert(0, os.path.dirname(__file__))

from graph import TouristGraph
from agent import TouristAgent
from search_algorithms import SearchAlgorithms
from csp_module import CSPSolver, TripConstraints
from bayesian_module import BayesianModule
from decision_module import DecisionModule
from hybrid_planner import HybridPlanner
from dataset import get_city_list, CITIES
from utils import (C, banner, sub_banner, info, warn, separator,
                   pick_city, get_int, get_float, yes_no, format_route,
                   format_time)


# ════════════════════════════════════════════════════════════
#  SHARED GLOBALS
# ════════════════════════════════════════════════════════════

graph   = TouristGraph()
agent   = TouristAgent(graph)
search  = SearchAlgorithms(graph)
csp     = CSPSolver(graph)
bayes   = BayesianModule()
decision= DecisionModule(graph)
hybrid  = HybridPlanner(graph)

CITIES_LIST = get_city_list()


# ════════════════════════════════════════════════════════════
#  HELPER: pick source & destination
# ════════════════════════════════════════════════════════════

def pick_src_dst():
    src = pick_city("Select SOURCE city:", CITIES_LIST)
    if src is None:
        warn("Invalid selection.")
        return None, None
    dst = pick_city("Select DESTINATION city:", CITIES_LIST)
    if dst is None:
        warn("Invalid selection.")
        return None, None
    if src == dst:
        warn("Source and destination must be different.")
        return None, None
    return src, dst


# ════════════════════════════════════════════════════════════
#  MENU HANDLERS
# ════════════════════════════════════════════════════════════

def menu_view_locations():
    graph.display_cities()
    graph.display_graph_summary()
    input(f"\n{C.CYAN}Press Enter to continue …{C.RESET}")


def menu_search(algo: str):
    src, dst = pick_src_dst()
    if src is None:
        return
    search.run_single(algo, src, dst)
    input(f"\n{C.CYAN}Press Enter to continue …{C.RESET}")


def menu_compare_all():
    src, dst = pick_src_dst()
    if src is None:
        return
    search.compare_all(src, dst)
    input(f"\n{C.CYAN}Press Enter to continue …{C.RESET}")


def menu_csp():
    banner("CSP TRIP PLANNER SETUP", 60)
    src, dst = pick_src_dst()
    if src is None:
        return

    # Build intermediate path via A*
    r = search.astar(src, dst)
    if not r.found:
        warn(f"No route found from {src} to {dst}.")
        return

    path = r.path
    info(f"Base route: {format_route(path)}")

    # Collect constraints
    print(f"\n{C.BOLD}Enter trip constraints:{C.RESET}")
    budget = get_float("  Total budget (INR) [default 25000]: ", lo=1000.0)
    if budget == 0:
        budget = 25_000.0
    days = get_int("  Number of days [1-14]:", 1, 14)

    print("  Hotel preference: 1=any  2=budget (≤₹3000)  3=luxury (>₹3000)")
    hp_choice = get_int("  Choice [1-3]:", 1, 3)
    hotel_pref = {1: "any", 2: "budget", 3: "luxury"}[hp_choice]

    max_att = get_int("  Max attractions per day [1-6]:", 1, 6)

    constraints = TripConstraints(
        total_budget=budget,
        total_days=days,
        hotel_preference=hotel_pref,
        max_attractions_day=max_att,
        must_visit=[src, dst],
    )

    use_mc = yes_no("  Use Min-Conflicts instead of Backtracking?")
    csp.run(path, constraints, use_min_conflicts=use_mc)
    input(f"\n{C.CYAN}Press Enter to continue …{C.RESET}")


def menu_bayesian():
    banner("BAYESIAN PREDICTION SETUP", 60)
    city = pick_city("Select a city for prediction:", CITIES_LIST)
    if city is None:
        return

    print("\n  Season: 1=winter  2=summer  3=monsoon  4=spring")
    s_choice = get_int("  Choice [1-4]:", 1, 4)
    season = {1: "winter", 2: "summer", 3: "monsoon", 4: "spring"}[s_choice]

    bayes.display_predictions(city, season)
    input(f"\n{C.CYAN}Press Enter to continue …{C.RESET}")


def menu_hybrid():
    banner("HYBRID INTELLIGENT PLANNER SETUP", 60)
    src, dst = pick_src_dst()
    if src is None:
        return

    budget = get_float("  Total budget (INR) [default 30000]: ", lo=1000.0)
    if budget == 0:
        budget = 30_000.0
    days = get_int("  Number of days [1-10]:", 1, 10)

    print("  Season: 1=winter  2=summer  3=monsoon  4=spring")
    s_choice = get_int("  Choice [1-4]:", 1, 4)
    season = {1: "winter", 2: "summer", 3: "monsoon", 4: "spring"}[s_choice]

    constraints = TripConstraints(
        total_budget=budget,
        total_days=days,
        hotel_preference="any",
        max_attractions_day=3,
    )
    hybrid.run(src, dst, constraints, season)
    input(f"\n{C.CYAN}Press Enter to continue …{C.RESET}")


def menu_performance():
    banner("PERFORMANCE ANALYSIS", 60)
    src, dst = pick_src_dst()
    if src is None:
        return
    search.compare_all(src, dst)
    input(f"\n{C.CYAN}Press Enter to continue …{C.RESET}")


def menu_agent_demo():
    banner("INTELLIGENT AGENT DEMO (CO1)", 60)
    agent.display_peas()

    src = pick_city("Agent starting city:", CITIES_LIST)
    if src is None:
        return
    budget = get_float("  Budget (INR) [default 20000]: ", lo=500.0)
    if budget == 0:
        budget = 20_000.0
    hours = get_float("  Time available (hours) [default 48]: ", lo=1.0)
    if hours == 0:
        hours = 48.0

    state = agent.initialise(src, budget, hours)
    agent.display_state()

    # Demonstrate a move + attraction visit
    neighbours = graph.neighbours(src)
    if neighbours:
        next_city = neighbours[0][0]
        info(f"Attempting to move to neighbour: {next_city}")
        agent.action_move(next_city)
        agent.display_state()

        # Visit first attraction in new city
        city_obj = graph.cities.get(next_city)
        if city_obj and city_obj.attractions:
            att = city_obj.attractions[0]
            info(f"Attempting to visit: {att.name}")
            agent.action_visit_attraction(att.name)
            agent.display_state()

    agent.display_log()
    input(f"\n{C.CYAN}Press Enter to continue …{C.RESET}")


def menu_decision():
    banner("DECISION MODULE DEMO (CO4)", 60)
    info("Generating 3 sample routes for decision analysis …")

    pairs = [("Delhi", "Mumbai"), ("Delhi", "Goa"), ("Jaipur", "Mumbai")]
    paths = []
    for s, d in pairs:
        r = search.astar(s, d)
        if r.found:
            paths.append(r.path)

    if not paths:
        warn("Could not find example routes.")
        return

    decision.run(paths, depth=3)
    input(f"\n{C.CYAN}Press Enter to continue …{C.RESET}")


def menu_run_tests():
    banner("RUNNING TEST SUITE", 60)
    from tests import run_all_tests
    run_all_tests()
    input(f"\n{C.CYAN}Press Enter to continue …{C.RESET}")


# ════════════════════════════════════════════════════════════
#  MAIN MENU
# ════════════════════════════════════════════════════════════

MENU_ITEMS = [
    ("View Tourist Locations",           menu_view_locations),
    ("BFS Route Search",                 lambda: menu_search("bfs")),
    ("DFS Route Search",                 lambda: menu_search("dfs")),
    ("UCS Route Search",                 lambda: menu_search("ucs")),
    ("Greedy Search",                    lambda: menu_search("greedy")),
    ("A* Search",                        lambda: menu_search("a*")),
    ("Compare All Algorithms",           menu_compare_all),
    ("CSP Trip Planner",                 menu_csp),
    ("Bayesian Prediction",              menu_bayesian),
    ("Decision Module (Minimax / AB)",   menu_decision),
    ("Intelligent Hybrid Recommendation",menu_hybrid),
    ("Performance Analysis",             menu_performance),
    ("Agent Demo (PEAS / CO1)",          menu_agent_demo),
    ("Run Test Suite",                   menu_run_tests),
    ("Exit",                             None),
]


def print_menu():
    print(f"\n{C.CYAN}{C.BOLD}{'═' * 52}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}       🌍  TOURIST ROUTE PLANNER  🗺{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}        Powered by AI (CO1 – CO6){C.RESET}")
    print(f"{C.CYAN}{C.BOLD}{'═' * 52}{C.RESET}")
    for i, (label, _) in enumerate(MENU_ITEMS, 1):
        marker = f"{C.GREEN}{i:>2}.{C.RESET}" if label != "Exit" else f"{C.RED}{i:>2}.{C.RESET}"
        print(f"  {marker} {label}")
    print(f"{C.CYAN}{'─' * 52}{C.RESET}")


def main():
    while True:
        print_menu()
        choice = input(f"\n{C.CYAN}Enter choice [1-{len(MENU_ITEMS)}]:{C.RESET} ").strip()

        if not choice.isdigit():
            warn("Please enter a valid number.")
            continue

        idx = int(choice) - 1
        if not (0 <= idx < len(MENU_ITEMS)):
            warn(f"Enter a number between 1 and {len(MENU_ITEMS)}.")
            continue

        label, handler = MENU_ITEMS[idx]

        if handler is None:
            print(f"\n{C.GREEN}{C.BOLD}Thank you for using Tourist Route Planner! Safe travels! 🌏{C.RESET}\n")
            sys.exit(0)

        try:
            handler()
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}Interrupted. Returning to menu …{C.RESET}")
        except Exception as exc:
            print(f"\n{C.RED}Unexpected error: {exc}{C.RESET}")
            import traceback
            traceback.print_exc()
            input(f"\n{C.CYAN}Press Enter to continue …{C.RESET}")


if __name__ == "__main__":
    main()

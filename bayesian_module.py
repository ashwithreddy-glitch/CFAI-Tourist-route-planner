# ============================================================
# bayesian_module.py – Probabilistic Reasoning (CO5)
# Bayes Rule, Bayesian Network, Variable Elimination,
# Rejection Sampling, Likelihood Weighting, HMM, Markov Chain
# ============================================================

from __future__ import annotations
import random
import math
from typing import Dict, List, Tuple, Optional

from dataset import CITIES, CITY_PROBS
from utils import (C, banner, sub_banner, info, ai_reason, separator,
                   warn)


# ════════════════════════════════════════════════════════════
#  BAYESIAN NETWORK  (simplified DAG for one city)
#
#   Season → Weather → Trip_Success
#             ↑
#           Crowd  → Trip_Success
#            ↑
#          Traffic
# ════════════════════════════════════════════════════════════

# Conditional Probability Tables (CPTs)

# P(Season)
CPT_SEASON = {"summer": 0.30, "monsoon": 0.25, "winter": 0.30, "spring": 0.15}

# P(Good_Weather | Season)
CPT_WEATHER_GIVEN_SEASON = {
    "summer":  {"good": 0.50, "bad": 0.50},
    "monsoon": {"good": 0.20, "bad": 0.80},
    "winter":  {"good": 0.80, "bad": 0.20},
    "spring":  {"good": 0.75, "bad": 0.25},
}

# P(Heavy_Crowd | Season)
CPT_CROWD_GIVEN_SEASON = {
    "summer":  {"heavy": 0.70, "light": 0.30},
    "monsoon": {"heavy": 0.30, "light": 0.70},
    "winter":  {"heavy": 0.80, "light": 0.20},
    "spring":  {"heavy": 0.60, "light": 0.40},
}

# P(Traffic | Crowd)
CPT_TRAFFIC_GIVEN_CROWD = {
    "heavy": {"high": 0.80, "low": 0.20},
    "light": {"high": 0.25, "low": 0.75},
}

# P(Trip_Success | Weather, Crowd)
CPT_SUCCESS = {
    ("good", "light"): {"success": 0.95, "failure": 0.05},
    ("good", "heavy"): {"success": 0.70, "failure": 0.30},
    ("bad",  "light"): {"success": 0.60, "failure": 0.40},
    ("bad",  "heavy"): {"success": 0.25, "failure": 0.75},
}


class BayesianModule:
    """
    Implements probabilistic reasoning for tourism uncertainty.
    """

    def __init__(self):
        self.city_probs = CITY_PROBS

    # ── Bayes Rule ───────────────────────────────────────────

    def bayes_update(
        self,
        prior: float,
        likelihood_given_true: float,
        likelihood_given_false: float,
    ) -> float:
        """
        P(H|E) = P(E|H) * P(H) / P(E)
        P(E)   = P(E|H)*P(H) + P(E|¬H)*P(¬H)
        """
        p_e = likelihood_given_true * prior + likelihood_given_false * (1 - prior)
        if p_e == 0:
            return 0.0
        posterior = (likelihood_given_true * prior) / p_e
        return posterior

    def compute_weather_given_evidence(self, city: str, season: str) -> Dict[str, float]:
        """Use Bayes to update weather belief given season evidence."""
        prior_good = self.city_probs.get(city, {}).get("good_weather", 0.6)
        # Evidence: P(season | good_weather) approximated from CPT
        p_season_good  = CPT_WEATHER_GIVEN_SEASON[season]["good"]
        p_season_bad   = CPT_WEATHER_GIVEN_SEASON[season]["bad"]

        posterior_good = self.bayes_update(prior_good, p_season_good, p_season_bad)
        return {"good_weather": round(posterior_good, 4),
                "bad_weather":  round(1 - posterior_good, 4)}

    # ── Variable Elimination ─────────────────────────────────

    def variable_elimination_success(self, season: str) -> Dict[str, float]:
        """
        Compute P(Trip_Success) by summing out Season, Weather, Crowd.
        This is exact inference via variable elimination.
        """
        ai_reason("VE", f"Eliminating hidden variables for season={season}")
        p_success = 0.0
        p_failure = 0.0

        p_weather = CPT_WEATHER_GIVEN_SEASON[season]
        p_crowd   = CPT_CROWD_GIVEN_SEASON[season]

        for w, pw in p_weather.items():
            crowd_key = "heavy" if w == "bad" else "light"
            # Use real crowd CPT
            for cr, pc in p_crowd.items():
                p_trip = CPT_SUCCESS.get((w, cr), {"success": 0.5, "failure": 0.5})
                p_success += pw * pc * p_trip["success"]
                p_failure += pw * pc * p_trip["failure"]

        # Normalise
        total = p_success + p_failure
        return {
            "success": round(p_success / total, 4),
            "failure": round(p_failure / total, 4),
        }

    # ── Rejection Sampling ───────────────────────────────────

    def rejection_sampling(self, evidence_weather: str, n_samples: int = 10_000) -> Dict[str, float]:
        """
        Estimate P(Trip_Success | Weather=evidence_weather) via sampling.
        Rejects samples inconsistent with evidence.
        """
        ai_reason("RS", f"Drawing {n_samples} samples | evidence weather={evidence_weather}")
        successes = 0
        accepted  = 0

        for _ in range(n_samples):
            # Sample Season
            season = random.choices(
                list(CPT_SEASON.keys()),
                weights=list(CPT_SEASON.values())
            )[0]
            # Sample Weather
            w_probs = CPT_WEATHER_GIVEN_SEASON[season]
            weather = random.choices(["good", "bad"], weights=[w_probs["good"], w_probs["bad"]])[0]

            # Reject if inconsistent with evidence
            if weather != evidence_weather:
                continue
            accepted += 1

            # Sample Crowd
            c_probs = CPT_CROWD_GIVEN_SEASON[season]
            crowd = random.choices(["heavy", "light"], weights=[c_probs["heavy"], c_probs["light"]])[0]

            # Sample Trip Success
            s_probs = CPT_SUCCESS.get((weather, crowd), {"success": 0.5, "failure": 0.5})
            outcome = random.choices(["success", "failure"],
                                      weights=[s_probs["success"], s_probs["failure"]])[0]
            if outcome == "success":
                successes += 1

        if accepted == 0:
            return {"success": 0.0, "failure": 0.0, "accepted_samples": 0}

        return {
            "success": round(successes / accepted, 4),
            "failure": round(1 - successes / accepted, 4),
            "accepted_samples": accepted,
        }

    # ── Likelihood Weighting ─────────────────────────────────

    def likelihood_weighting(self, evidence_weather: str, n_samples: int = 5_000) -> Dict[str, float]:
        """
        Estimate P(Success | Weather=evidence) via weighted samples.
        Each sample is weighted by P(evidence | parents).
        More efficient than rejection sampling.
        """
        ai_reason("LW", f"Likelihood weighting | evidence={evidence_weather}")
        w_success = 0.0
        w_total   = 0.0

        w_idx = 0 if evidence_weather == "good" else 1

        for _ in range(n_samples):
            season = random.choices(list(CPT_SEASON.keys()),
                                    weights=list(CPT_SEASON.values()))[0]
            # Weight by P(Weather=evidence | Season)
            p_w = CPT_WEATHER_GIVEN_SEASON[season]
            weight = p_w[evidence_weather]

            c_probs = CPT_CROWD_GIVEN_SEASON[season]
            crowd = random.choices(["heavy", "light"],
                                   weights=[c_probs["heavy"], c_probs["light"]])[0]

            s_probs = CPT_SUCCESS.get((evidence_weather, crowd), {"success": 0.5, "failure": 0.5})
            outcome = random.choices(["success", "failure"],
                                      weights=[s_probs["success"], s_probs["failure"]])[0]

            w_total += weight
            if outcome == "success":
                w_success += weight

        if w_total == 0:
            return {"success": 0.5, "failure": 0.5}
        return {
            "success": round(w_success / w_total, 4),
            "failure": round(1 - w_success / w_total, 4),
        }

    # ── Markov Chain (tourist movement) ─────────────────────

    def markov_chain_simulation(self, start_city: str, steps: int = 7) -> List[str]:
        """
        Simulate tourist movement as a first-order Markov Chain.
        Transition probability proportional to 1/distance.
        """
        from dataset import get_adjacency
        adj = get_adjacency()
        trajectory = [start_city]
        current = start_city

        ai_reason("MARKOV", f"Simulating {steps}-step tourist movement from {start_city}")

        for _ in range(steps):
            neighbours = adj.get(current, [])
            if not neighbours:
                break
            # Weight: closer cities more likely
            weights = [1 / (d + 1) for _, d, _, _ in neighbours]
            next_city = random.choices([n for n, *_ in neighbours], weights=weights)[0]
            trajectory.append(next_city)
            current = next_city

        return trajectory

    # ── Hidden Markov Model (tourist location tracking) ──────

    def hmm_viterbi(
        self,
        observations: List[str],
        states: List[str],
    ) -> List[str]:
        """
        Viterbi algorithm to find most likely sequence of hidden states
        (actual cities) given noisy observation sequence.

        States      : city names
        Observations: noisy location reports (may be incorrect)
        Emission    : P(obs | state) = 0.8 if same, uniform else
        Transition  : uniform among connected neighbours
        """
        from dataset import get_adjacency
        adj = get_adjacency()

        n_states = len(states)
        n_obs    = len(observations)
        state_idx = {s: i for i, s in enumerate(states)}

        # Emission probability P(obs | state)
        def emit(state: str, obs: str) -> float:
            return 0.80 if state == obs else 0.20 / max(1, n_states - 1)

        # Transition probability P(next | current) – uniform among neighbours
        def trans(cur: str, nxt: str) -> float:
            nbrs = [n for n, *_ in adj.get(cur, [])]
            if not nbrs:
                return 1.0 / n_states
            return (1.0 / len(nbrs)) if nxt in nbrs else 0.0

        # Initialise
        pi  = [[0.0] * n_states for _ in range(n_obs)]
        ptr = [[0]   * n_states for _ in range(n_obs)]
        init_obs = observations[0]
        for j, s in enumerate(states):
            pi[0][j] = (1.0 / n_states) * emit(s, init_obs)

        # Forward pass
        for t in range(1, n_obs):
            for j, s in enumerate(states):
                best_prob = -1.0
                best_prev = 0
                for i, ps in enumerate(states):
                    p = pi[t-1][i] * trans(ps, s) * emit(s, observations[t])
                    if p > best_prob:
                        best_prob = p
                        best_prev = i
                pi[t][j] = best_prob
                ptr[t][j] = best_prev

        # Backtrack
        path = [0] * n_obs
        path[-1] = max(range(n_states), key=lambda j: pi[-1][j])
        for t in range(n_obs - 2, -1, -1):
            path[t] = ptr[t + 1][path[t + 1]]

        return [states[i] for i in path]

    # ── City-level predictions ───────────────────────────────

    def predict_city(self, city: str, season: str = "winter") -> Dict[str, float]:
        """
        Return probability predictions for a given city and season.
        Combines prior data with Bayesian update.
        """
        priors = self.city_probs.get(city, {
            "good_weather": 0.6, "heavy_crowd": 0.65, "traffic": 0.55})

        weather = self.compute_weather_given_evidence(city, season)
        ve      = self.variable_elimination_success(season)

        # Expected utility
        sat_score  = sum(a.satisfaction for a in CITIES[city].attractions) / max(1, len(CITIES[city].attractions))
        exp_util   = ve["success"] * sat_score

        return {
            "city":           city,
            "season":         season,
            "P(good_weather)": weather["good_weather"],
            "P(heavy_crowd)":  priors["heavy_crowd"],
            "P(traffic)":      priors["traffic"],
            "P(trip_success)": ve["success"],
            "expected_utility": round(exp_util, 3),
        }

    # ── Display ──────────────────────────────────────────────

    def display_predictions(self, city: str, season: str = "winter") -> None:
        banner(f"BAYESIAN PREDICTIONS – {city} ({season})", 65)
        pred = self.predict_city(city, season)

        ai_reason("BAYES", f"Prior weather = {self.city_probs.get(city, {}).get('good_weather', 0.6):.2f}")
        ai_reason("VE",    f"Variable elimination over season={season}")

        sub_banner("Probabilities")
        for k, v in pred.items():
            if isinstance(v, float):
                bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
                print(f"  {k:<25}: {v:.4f}  [{bar}]")

        sub_banner("Rejection Sampling  (10 000 samples | weather='good')")
        rs = self.rejection_sampling("good")
        info(f"P(Success | Good Weather) ≈ {rs['success']:.4f}  "
             f"(from {rs['accepted_samples']} accepted samples)")

        sub_banner("Likelihood Weighting (5 000 samples | weather='good')")
        lw = self.likelihood_weighting("good")
        info(f"P(Success | Good Weather) ≈ {lw['success']:.4f}")

        sub_banner("Markov Chain – Simulated 7-Day Movement")
        traj = self.markov_chain_simulation(city, 7)
        info("Trajectory : " + " → ".join(traj))

        sub_banner("HMM – Viterbi Decoding")
        from dataset import get_city_list
        all_cities = get_city_list()[:6]  # use first 6 for demo
        obs_sequence = traj[:min(4, len(traj))]
        actual = [c for c in obs_sequence if c in all_cities]
        if len(actual) >= 2:
            decoded = self.hmm_viterbi(actual, all_cities)
            info(f"Observed  : {' → '.join(actual)}")
            info(f"Decoded   : {' → '.join(decoded)}")
        separator()

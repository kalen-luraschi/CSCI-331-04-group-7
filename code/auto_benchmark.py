# auto_benchmark.py
import time
import json
import csv
from game_2048 import Game2048, minmax_agent

# ===========================================
#  AI METHODS TO BENCHMARK (callable wrappers)
# ===========================================

def get_methods(ai, depth=4):
    return {
        "random": lambda g: ai.act_random(g),
        # "simple": lambda g: ai.act_simple(g),
        # "minmax": lambda g: ai.act_minimax(g, depth),
        # "alpha_beta": lambda g: ai.act_alpha_beta(g, depth),
        # "expectimax": lambda g: ai.act_expectimax(g, depth)
    }

# Recommended number of games based on speed
GAMES_PER_METHOD = {
    "random":     100000,
    "simple":     40000,
    "minmax":     50,
    "alpha_beta": 140,
    "expectimax": 100,
}

# =============================
#  RUN BENCHMARK FOR 1 METHOD
# =============================

def run_benchmark(ai_method, ai_func, num_games):
    print(f"\nRunning {ai_method.upper()}  ({num_games} games)...")

    ai_results = {
        "method": ai_method,
        "scores": [],
        "highest_tiles": [],
        "times_per_move": []
    }

    for game_i in range(num_games):
        game = Game2048()
        ai = minmax_agent()
        moves = 0
        move_times = []

        # Run full game
        while not game.is_game_over():
            t0 = time.perf_counter()
            move = ai_func(game)
            t1 = time.perf_counter()

            if move is None:
                break

            move_times.append(t1 - t0)
            game.move(move)
            moves += 1

            if moves > 5000:  # runaway safety valve
                break

        # Record data
        ai_results["scores"].append(game.score)
        ai_results["highest_tiles"].append(max(max(r) for r in game.grid))
        ai_results["times_per_move"].extend(move_times)

        # Status update every ~20%
        if num_games >= 5 and (game_i + 1) % (num_games // 5) == 0:
            print(".", end="", flush=True)

    print(" done.")
    return ai_results

# ============================
#  SUMMARY STATS PER ALGORITHM
# ============================

def summarize(results):
    times = results["times_per_move"]
    avg_time_move = sum(times) / len(times) if times else 0
    moves_per_sec = 1 / avg_time_move if avg_time_move else 0

    return {
        "method": results["method"],
        "avg_score": sum(results["scores"]) / len(results["scores"]),
        "max_score": max(results["scores"]),
        "min_score": min(results["scores"]),
        "avg_highest_tile": sum(results["highest_tiles"]) / len(results["highest_tiles"]),
        "max_highest_tile": max(results["highest_tiles"]),
        "avg_time_per_move": avg_time_move,
        "moves_per_second": moves_per_sec,
        "time_for_10_moves": avg_time_move * 10
    }

# ============================
#  MAIN AUTOBENCHMARK LOGIC
# ============================

def run_all_benchmarks(depth=4, save_csv=True, save_json=True):
    ai = minmax_agent()
    methods = get_methods(ai, depth)
    full_results = {}

    print("\n======================================")
    print("AUTO BENCHMARK — All AI Algorithms")
    print("======================================\n")

    for method_name, ai_func in methods.items():
        num_games = GAMES_PER_METHOD[method_name]
        raw_results = run_benchmark(method_name, ai_func, num_games)
        summary = summarize(raw_results)
        full_results[method_name] = summary

    # =========
    # OUTPUT
    # =========
    print("\n======================================")
    print("RESULT SUMMARY")
    print("======================================\n")

    for m, stats in full_results.items():
        print(f"{m.upper():<12} | "
              f"AvgScore {stats['avg_score']:.0f} | "
              f"MaxTile {stats['max_highest_tile']} | "
              f"t/move {stats['avg_time_per_move']:.5f}s | "
              f"Moves/s {stats['moves_per_second']:.1f}")

    # =========
    # SAVE FILES
    # =========
    if save_json:
        with open("benchmark_results.json", "w") as f:
            json.dump(full_results, f, indent=4)
        print("\nSaved: benchmark_results.json")

    if save_csv:
        with open("benchmark_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "method", "avg_score", "max_score", "min_score",
                "avg_highest_tile", "max_highest_tile",
                "avg_time_per_move", "moves_per_second", "time_for_10_moves"
            ])
            for stats in full_results.values():
                writer.writerow([
                    stats["method"],
                    stats["avg_score"],
                    stats["max_score"],
                    stats["min_score"],
                    stats["avg_highest_tile"],
                    stats["max_highest_tile"],
                    stats["avg_time_per_move"],
                    stats["moves_per_second"],
                    stats["time_for_10_moves"]
                ])
        print("Saved: benchmark_results.csv")

    print("\nBenchmark Complete.\n")
    return full_results


if __name__ == "__main__":
    run_all_benchmarks(depth=4)

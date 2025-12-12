# CSCI-331-04-group-7

Description:
Project-2
Create an AI agent to play the 2048 Classic Game (https://classic.play2048.co/) using any two
appropriate search algorithms eg. Minimax/Alpha-Beta, and compare their performance. A
presentable UI/interface is expected.
Reference: https://medium.com/data-science/a-puzzle-for-ai-eb7a3cb8e599

Developers:
- Kalen Luraschi
- Ethan White
- Daniel Birley

How to run the project guid:
- In the `code` folder run `python game_2048.py`
Run automated benchmarks:
- in the `code` folder run `python auto_benchmark.py`

The following AI strategies were implemented:
- Random – selects a random valid move (baseline).
- Simple Greedy – evaluates one move ahead using a heuristic.
- Minimax – multi-step search assuming worst-case tile placement.
- Alpha-Beta Pruning – optimized minimax that prunes unnecessary branches.
- Expectimax – probabilistic search that models random tile spawns.

The AI agents use a weighted heuristic function that considers:
- Number of empty tiles
- Smoothness between neighboring tiles
- Monotonic (snake-like) tile ordering
- Corner placement of the highest tile
- Merge potential for large tiles
- These heuristics guide the AI toward stable board states and high-value merges.
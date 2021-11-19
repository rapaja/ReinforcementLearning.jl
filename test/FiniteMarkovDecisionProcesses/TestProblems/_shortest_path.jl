# TEST PROBLEM: Minimum Distance in Small Graph
#
#   |-------|            |-------|
#   |   1   |<---------->|   2   |
#   |-------|            |-------|
#       ^                    ^
#       |                    |
#       |                    |
#       v                    v
#   |-------|            |-------|
#   |   3   |<---------->|   4   |
#   |-------|            |-------|

# The actions available in each node are UPPER (1) and 
# LOWER (2) transition.

# The problem is to find minimal distance to node 4.
# The cost of each transition is equal to one.

tp_shortest_path_V = [2 1 1 0]
tp_shortest_path_Q = [2 2; 3 1; 3 1; 2 2]
import heapq

# Waste collection locations with their coordinates
locations = {
    "Depot": (0, 0),
    "Bin_A": (2, 1),
    "Bin_B": (4, 2),
    "Bin_C": (5, 4),
    "Bin_D": (7, 3),
    "Plant": (9, 5)
}

# Road connections and distances
graph = {
    "Depot": [("Bin_A", 3), ("Bin_B", 5)],
    "Bin_A": [("Depot", 3), ("Bin_B", 2), ("Bin_C", 4)],
    "Bin_B": [("Depot", 5), ("Bin_A", 2), ("Bin_C", 3),
              ("Bin_D", 4)],
    "Bin_C": [("Bin_A", 4), ("Bin_B", 3), ("Bin_D", 2),
              ("Plant", 5)],
    "Bin_D": [("Bin_B", 4), ("Bin_C", 2), ("Plant", 3)],
    "Plant": [("Bin_C", 5), ("Bin_D", 3)]
}


# Heuristic function
def heuristic(current, goal):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]

    return abs(x1 - x2) + abs(y1 - y2)


# A* algorithm
def a_star(start, goal):

    open_list = []

    # (f_cost, g_cost, current_node, path)
    heapq.heappush(open_list, (0, 0, start, [start]))

    visited = set()

    while open_list:

        f, g, current, path = heapq.heappop(open_list)

        if current in visited:
            continue

        visited.add(current)

        # Goal reached
        if current == goal:
            return path, g

        # Explore neighboring locations
        for neighbor, cost in graph[current]:

            if neighbor not in visited:

                new_g = g + cost
                h = heuristic(neighbor, goal)
                new_f = new_g + h

                heapq.heappush(
                    open_list,
                    (new_f, new_g, neighbor, path + [neighbor])
                )

    return None, float("inf")


# Main program
start = "Depot"
goal = "Plant"

path, distance = a_star(start, goal)

if path:
    print("Optimized Waste Pickup Route:")
    print(" -> ".join(path))

    print("Total Distance:", distance, "km")
else:
    print("No route found.")

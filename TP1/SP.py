def SP_naive (graph, s):
    frontier = [s]
    dist = {s:0}

    while len(frontier) > 0:

        x = min(frontier, key = lambda k: dist[k])
        frontier.remove(x)

        for y, dxy in graph[x].items():
            dy = dist[x] + dxy

            if y not in dist:
                frontier.append(y)
                dist[y] = dy

            elif dist[y] > dy:
                dist[y] = dy

    return dist

from heapq import *

def SP_heap (graph, s):
    frontier = []
    heappush(frontier, (0, s))
    done = set()
    dist = {s: 0}

    while len(frontier) > 0:

        dx, x = heappop(frontier)
        if x in done:
            continue

        done.add(x)

        for y, dxy in graph[x].items():
            dy = dx + dxy

            if y not in dist or dist[y] > dy:
                heappush(frontier,(dy, y))
                dist[y] = dy

    return dist
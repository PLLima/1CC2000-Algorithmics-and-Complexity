def add_source(graph, src):

  if src in graph:
    return "error: the source vertex is already in the graph"

  graph_src = graph.copy()
  
  graph_src[src] = {}
  for v in graph.keys():
    graph_src[src][v] = 0

  return graph_src
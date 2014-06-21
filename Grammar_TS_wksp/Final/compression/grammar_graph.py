'''
Created on 18 avr. 2014

@author: francois
'''

import pydot

def add_rule(graph, 
             lhs, 
             grammar):
        rhs = grammar[lhs]
        lhs_node = pydot.Node(lhs, style = 'filled', fillcolor = "lightseagreen")
        previous_node = lhs_node
        cluster = pydot.Cluster(lhs, rankdir = 'TB', style = 'filled')
        cluster.add_node(previous_node)
        for i, x in enumerate(rhs):
            if(x in grammar):
                other_lhs_node = add_rule(graph, x, grammar)
                next_node = pydot.Node(lhs + ' rhs ' + str(i) + ', ' + x, 
                                       style = 'filled', 
                                       fillcolor = "lightcoral")
                graph.add_edge(pydot.Edge(next_node, other_lhs_node))
            else:
                next_node = pydot.Node(lhs + ' rhs ' + str(i) + ', ' + x, 
                                       style = 'filled', 
                                       fillcolor = "lightsteelblue")
            cluster.add_node(next_node)
            cluster.add_edge(pydot.Edge(previous_node, next_node))
            previous_node = next_node
        graph.add_subgraph(cluster)
        return lhs_node

def grammar_to_graph(file_path, 
                     grammar,
                     root_sentence,
                     cut = 10):
    graph = pydot.Dot(graph_type='digraph', rankdir = 'TB')
    grammar['root'] = root_sentence[-cut:]
    add_rule(graph, 'root', grammar)
    print file_path
    del grammar['root']
    graph.write_png(file_path)
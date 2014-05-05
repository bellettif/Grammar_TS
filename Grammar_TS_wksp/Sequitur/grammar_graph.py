'''
Created on 18 avr. 2014

@author: francois
'''

import pydot

def add_rule(graph, lhs, grammar):
        rhs = grammar[lhs][0]
        stringed_rhs = [str(x) for x in rhs]
        print 'Rule %s: %s' % (lhs, ' '.join(stringed_rhs))
        if lhs == 0:
            lhs_node = pydot.Node('Rule %d' % (-lhs), style = 'filled', fillcolor = "purple")
        else:
            lhs_node = pydot.Node('Rule %d' % (-lhs), style = 'filled', fillcolor = "green")
        previous_node = lhs_node
        cluster = pydot.Cluster(str(-lhs), rankdir = 'LR', style = 'filled')
        cluster.add_node(previous_node)
        for i, x in enumerate(rhs):
            if(x in grammar):
                other_lhs_node = add_rule(graph, x, grammar)
                next_node = pydot.Node(str(lhs) + '_' + str(i) + '_' + str(x), style = 'filled', fillcolor = "red")
                graph.add_edge(pydot.Edge(next_node, other_lhs_node))
            else:
                next_node = pydot.Node(str(lhs) + '_' + str(i) + '_' + str(x), style = 'filled', fillcolor = "blue")
            cluster.add_node(next_node)
            cluster.add_edge(pydot.Edge(previous_node, next_node))
            previous_node = next_node
        graph.add_subgraph(cluster)
        return lhs_node

def grammar_to_graph(file_path, grammar):
    graph = pydot.Dot(graph_type='digraph', rankdir = 'LR')
    lhs = 0
    add_rule(graph, lhs, grammar)
    print file_path
    graph.write_png(file_path)
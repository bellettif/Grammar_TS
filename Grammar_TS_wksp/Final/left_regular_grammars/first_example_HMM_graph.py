'''
Created on 9 avr. 2014

@author: francois
'''


import pydot

start_color = 'gray'
end_color = 'gray'
rule_1_color = '#0066FF'
rule_2_color = '#FF9933'
rule_3_color = '#99CCFF'

# this time, in graph_type we specify we want a DIrected GRAPH
graph = pydot.Dot(graph_type='digraph', rankdir = 'LR')

node_Start = pydot.Node("Start\nemits\na, p=0.79\nb, p=0.07\nc, p=0.07\nd, p=0.07",
                     style = 'filled', fillcolor = start_color, fillalpha = 0.1, shape = 'box')
node_rule_1 = pydot.Node("rule 1\nemits\na, p=0.07\nb, p=0.79\nc, p=0.07\nd, p=0.07",
                     style = 'filled', fillcolor = rule_1_color, fillalpha = 0.1, shape = 'box')
node_rule_2 = pydot.Node("rule 2\nemits\na, p=0.07\nb, p=0.07\nc, p=0.79\nd, p=0.07",
                     style = 'filled', fillcolor = rule_2_color, fillalpha = 0.1, shape = 'box')
node_rule_3 = pydot.Node("rule 3\nemits\na, p=0.71\nb, p=0.07\nc, p=0.07\nd, p=0.15",
                     style = 'filled', fillcolor = rule_3_color, fillalpha = 0.1, shape = 'box')
node_rule_4 = pydot.Node("rule 4\nemits #end",
                        style = 'filled', fillcolor = end_color, fillalpha = 0.1, shape = 'box')

graph.add_node(node_Start)
graph.add_node(node_rule_1)
graph.add_node(node_rule_2)
graph.add_node(node_rule_3)
graph.add_node(node_rule_4)

graph.add_edge(pydot.Edge(node_Start, node_rule_1, label = '0.9', penwidth = 0.9 * 4,
                          color = start_color, fontcolor = start_color))
graph.add_edge(pydot.Edge(node_Start, node_rule_2, label = '0.05', penwidth = 0.05 * 4,
                          color = start_color, fontcolor = start_color))
graph.add_edge(pydot.Edge(node_Start, node_rule_3, label = '0.05', penwidth = 0.05 * 4,
                          color = start_color, fontcolor = start_color))

graph.add_edge(pydot.Edge(node_rule_1, node_rule_1, label = '0.05', penwidth = 0.05 * 4,
                          color = rule_1_color, fontcolor = rule_1_color))
graph.add_edge(pydot.Edge(node_rule_1, node_rule_2, label = '0.9', penwidth = 0.9 * 4,
                          color = rule_1_color, fontcolor = rule_1_color))
graph.add_edge(pydot.Edge(node_rule_1, node_rule_3, label = '0.05', penwidth = 0.05 * 4,
                          color = rule_1_color, fontcolor = rule_1_color))

graph.add_edge(pydot.Edge(node_rule_2, node_rule_1, label = '0.05', penwidth = 0.05 * 4,
                          color = rule_2_color, fontcolor = rule_2_color))
graph.add_edge(pydot.Edge(node_rule_2, node_rule_2, label = '0.05', penwidth = 0.05 * 4,
                          color = rule_2_color, fontcolor = rule_2_color))
graph.add_edge(pydot.Edge(node_rule_2, node_rule_3, label = '0.9', penwidth = 0.9 * 4,
                          color = rule_2_color, fontcolor = rule_2_color))

graph.add_edge(pydot.Edge(node_rule_3, node_rule_1, label = '0.8', penwidth = 0.8 * 4,
                          color = rule_3_color, fontcolor = rule_3_color))
graph.add_edge(pydot.Edge(node_rule_3, node_rule_2, label = '0.05', penwidth = 0.05 * 4,
                          color = rule_3_color, fontcolor = rule_3_color))
graph.add_edge(pydot.Edge(node_rule_3, node_rule_3, label = '0.05', penwidth = 0.05 * 4,
                          color = rule_3_color, fontcolor = rule_3_color))
graph.add_edge(pydot.Edge(node_rule_3, node_rule_4, label = '0.1', penwidth = 0.1 * 4,
                          color = rule_3_color, fontcolor = rule_3_color))

graph.write_png('Left_reg_ex_HMM_graph.png')

# this is too good to be true!
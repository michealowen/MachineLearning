from graphviz import Digraph

g = Digraph('tree',strict=False)
g.node("sss",'ffff')
g.node("ddd",'gggg')
g.edge('sss','ddd')

print(g)
g.view()

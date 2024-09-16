import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

#CREATE FIGURE AND WINDOW
fig, ax = plt.subplots(nrows=2,ncols=1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(8,7))
fig.canvas.manager.set_window_title("Pagerank visualiser")


#CREATE GRAPH -----------------
def create_graph(nnodes, avgnedges):
    #Can't have more average edges than number of nodes.
    if nnodes < avgnedges:
        return None
    #Using the Watts-Strogatz model, which may be not the best idea as although it does simulate small-worldedness, it more simulates social networks rather than the internet.
    ws = nx.watts_strogatz_graph(nnodes,avg_edges,0.5)
    #Convert to directed graph (with each undirected edge converted to 2 directed edges), then removing one of each of these directed edge pairs.
    ed = ws.edges()
    g = ws.to_directed()
    g.remove_edges_from(ed)

            
    #Here is the actual implementation of the PageRank algorithm:
    damping_factor = 0.85
    #Column vector containing nnodes entries all of 1/nnodes. This is to split 1 PageRank between all nodes.
    pagerank = np.ones([nnodes,1],dtype=float) * np.array(1/nnodes)

    #Creating link_matrix M (starting as nnodes square 0 matrix) 
    M = np.zeros([nnodes,nnodes],dtype=float)
    for e in g.edges:
        #M_ij = 1 / (number of out edges of node j if j links to i)
        M[e[1]][e[0]] = 1 / len(g.out_edges(e[0]))

    #Iterate the matrix multiplication: pagerank(t+1) = d * M * pagerank(t) + ((1-d)/nnodes)*1
        #1 is the column vector that is nnodes long containing all 1s
    #This continues until the difference between pagerank(t+1) and pagerank(t) is negligible
    while True:
        prev_pagerank = pagerank
        #pagerank(t+1) = d * M * pagerank(t) + ((1-d)/nnodes)*1
        pagerank = np.array(damping_factor) * np.dot(M, pagerank) + np.array( (1-damping_factor) / nnodes ) * np.ones([nnodes,1])
        if np.linalg.norm(pagerank - prev_pagerank) < 1e-6:
            break

    #print(pagerank)

#DRAWING
    #Random layout of nodes in the window, saved for positioning of edges and nodes.
    g_pos=nx.random_layout(g)

    #Draw all the nodes with both size and blue colour proportional to pagerank. Taking the min between 1 and the blue value just for edge cases where we exceed 1.
    nx.draw_networkx_nodes(g, node_size=800*pagerank, pos=g_pos, node_color=[[0,0,min(1, float(pagerank[i][0])*10)] for i in range(nnodes)])
    #Drawing the edges w/ transparency proportional to pagerank of destination node.
    alphas = [min(1, pagerank[e[1]]) for e in g.edges]
    drawn_edges = nx.draw_networkx_edges(g, width = 0.5, pos=g_pos, alpha=alphas, arrowsize=4)

#Function to refresh the graph, clearing the graph axes and creating a new one according to the slider values.
def load_new_example(event):
    plt.axes(ax[0])
    plt.cla()
    create_graph(node_slider.val, edge_slider.val)
    plt.show()

#Creating buttons and sliders as well as axes to contain them.
fig.canvas.toolbar.pack_forget()
fig.tight_layout()
plt.axes(ax[0])
create_graph(10, 10)
ax[1].set_visible(False)
b1axis=plt.axes([0.06,0.1,0.1,0.1])
brefresh = Button(b1axis, "Refresh")


axnode = fig.add_axes([0.35, 0.2, 0.5, 0.01])
node_slider = Slider(
    ax=axnode,
    label='Number of nodes',
    valmin=10,
    valmax=200,
    valinit=num_nodes,
    valstep=1
)
axedge = fig.add_axes([0.35, 0.1, 0.5, 0.01])
edge_slider = Slider(
    ax=axedge,
    label='Avg. edges',
    valmin=1,
    valmax=50,
    valinit=avg_edges,
    valstep=1
)
brefresh.on_clicked(load_new_example)


plt.show()


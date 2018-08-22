import plotly.offline as offline
import plotly.graph_objs as go
import pandas as pd
from collections import Counter
import numpy as np
from matplotlib import cm

import math
import numpy as np

numeric_stability = 0.0000001

def acosh(x):
    return math.log(x+(x**2-1)**0.5)
    
def hyperbolic_distance(u, v):
    # this could probably be one line (if matrix is always the same)
    uu = (u**2).sum()
    uv = (u*v).sum()   
    vv = (v**2).sum()
    return acosh(max(1.,1+2*(uu-2*uv+vv) / max(numeric_stability, 1-uu) / max(numeric_stability, 1-vv)))

def transitive_isometry(t1, t0):
    u'''
    computing isometry which move t1 to t0
    '''

    (x1, y1), (x0,y0) = t1, t0

    def to_h(z):
        return (1 + z)/(1 - z) * complex(0,1)

    def from_h(h):
        return (h - complex(0,1)) / (h + complex(0,1))

    z1 = complex(x1, y1)
    z0 = complex(x0, y0)

    h1 = to_h(z1)
    h0 = to_h(z0)

    def f(h):
        assert( h0.imag > 0 )
        assert( h1.imag > 0 )
        return h0.imag/h1.imag * (h - h1.real) + h0.real

    def ret(z):
        z = complex(z[0], z[1])
        h = to_h(z)
        h = f(h)
        z = from_h(h)
        return z.real, z.imag

    return ret

def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = map(np.uint8, np.array(cmap(k*h)[:3])*255)
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale

def poincare_icd_2d_visualization(embeddings, tree, figure_title, nodes_colors = [], nodes_size = [],
                                  num_nodes=50, center=None, show_node_labels=(), chapter_names=False):
    """Create a 2-d plot of the nodes and edges of a 2-d poincare embedding.

    Parameters
    ----------
    model : pandas 
    tree : set
        Set of tuples containing the direct edges present in the original dataset.
    figure_title : str
        Title of the plotted figure.
    num_nodes : int or None
        Number of nodes for which edges are to be plotted.
        If `None`, all edges are plotted.
        Helpful to limit this in case the data is too large to avoid a messy plot.
    show_node_labels : iterable
        Iterable of nodes for which to show labels by default.

    Returns
    -------
    :class:`plotly.graph_objs.Figure`
        Plotly figure that contains plot.

    """
    if center:
        c = embeddings.loc[embeddings['icd'] == center].head(1)
        isom = transitive_isometry((c['x'], c['y']), (0, 0))
        
        node_labels = embeddings['icd']
        nodes_x, nodes_y = [], []
        
        for i, p in embeddings.iterrows():
            xy = isom((p['x'], p['y']))
            nodes_x.append(xy[0])
            nodes_y.append(xy[1])
#             if p['icd'][2] == 'v':
#             nodes_colors.append(int(p['icd'][2]))
        
    else:
        node_labels = embeddings['icd']
        nodes_x = list(embeddings['x'])
        nodes_y = list(embeddings['y'])
#         nodes_colors = list(embeddings['icd'].str[2].astype(int))

    if chapter_names:  
        nodes = go.Scatter(
            x=nodes_x, y=nodes_y,
            mode='markers',
            marker=dict(color=nodes_colors, colorscale='Rainbow', size=nodes_size,
                         colorbar=dict(title='Colorbar', tickvals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16],
                         ticktext = ["001-139", "140-239", "240-279", "280-289", "290-319", "320-389", "390-459", "460-519", "520-579", "580-629", "630-679", "680-709", "710-739", "740-759", "760-779", "780-799", "800-999"],),
                       ),
            #marker=dict(color='rgb(30, 100, 200)'),
            text=node_labels,
            textposition='bottom center'
        )
    else:
        nodes = go.Scatter(
            x=nodes_x, y=nodes_y,
            mode='markers',
            marker=dict(color=nodes_colors, colorscale='Rainbow', size=nodes_size),
            #marker=dict(color='rgb(30, 100, 200)'),
            text=node_labels,
            textposition='bottom center'
        )

    nodes_x, nodes_y, node_labels = [], [], []
    for node in show_node_labels:
        vector = embeddings.loc[embeddings['icd'] == node]
        xy = isom((vector['x'], vector['y']))
        nodes_x.append(xy[0])
        nodes_y.append(xy[1])
        node_labels.append(node)
    nodes_with_labels = go.Scatter(
        x=nodes_x, y=nodes_y,
        mode='text',
        marker=dict(color='rgb(200, 100, 200)'),
        text=node_labels,
        textposition='bottom center'
    )

    node_out_degrees = Counter(hypernym_pair[1] for hypernym_pair in tree)
    if num_nodes is None:
        chosen_nodes = list(node_out_degrees.keys())
    else:
        chosen_nodes = list(sorted(node_out_degrees.keys(), key=lambda k: -node_out_degrees[k]))[:num_nodes]

    edges_x = []
    edges_y = []
    for u, v in tree:
        vector_u = embeddings.loc[embeddings['icd'] == u]
        vector_v = embeddings.loc[embeddings['icd'] == v]
        
        if vector_u.shape[0] == 1 and vector_v.shape[0] == 1:
            vector_u_isom = isom((vector_u['x'], vector_u['y']))
            vector_v_isom = isom((vector_v['x'], vector_v['y']))
            
            if len(vector_u_isom) == 2 and len(vector_v_isom) == 2:
                edges_x += [vector_u_isom[0], vector_v_isom[0], None]
                edges_y += [vector_u_isom[1], vector_v_isom[1], None]
            else:
                raise Exception(vector_u, vector_v)
    
    edges = go.Scatter(
        x=edges_x, y=edges_y, mode='lines', hoverinfo=None,
        line=dict(color='rgb(50,50,50)', width=1))

    layout = go.Layout(
        title=figure_title, showlegend=False, hovermode='closest', width=800, height=800)
    return go.Figure(data=[edges, nodes, nodes_with_labels], layout=layout)

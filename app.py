# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from eventdata import EventData
from inference_functions import *

import os
os.environ['PYTHONDONTWRITEBYTECODE'] = "1"

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
eventdata = EventData('data')
app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                # Column for user controls
                html.Div(
                    className="four columns div-user-controls",
                    children=[
                      html.H1("LArTPC EVENT DISPLAY"),
                      html.P(
                          """
            2D and 3D event display app for fast visualization
            and graphical user interface. You can also run inference using
            different clustering algorithms (ex. DBSCAN, HDBSCAN, OPTICS, etc.)
            """
                      ),
                        html.Div(
                          className="row",
                          children=[
                              html.H4("FILTERS"),
                              html.Div(
                                  className="div-for-dropdown",
                                  children=[
                                      # Dropdown for locations on map
                                      html.H5("BATCH INDEX"),
                                      dcc.Dropdown(
                                          id="batch-id-dropdown",
                                          options=[
                                              {"label": i, "value": i}
                                              for i in eventdata._events
                                          ],
                                          placeholder="Select an Event Number",
                                          value=0
                                      )
                                  ],
                              ),
                              html.Div(
                                  className="div-for-dropdown",
                                  children=[
                                      # Dropdown to select times
                                      html.H5("SEMANTIC CLASS"),
                                      dcc.Dropdown(
                                          id="semantic-class-selector",
                                          options=[
                                              {"label": 'All', "value": 'All'},
                                              {"label": 'HIP', "value": 0},
                                              {"label": 'MIP', "value": 1},
                                              {"label": 'EM Shower', "value": 2},
                                              {"label": 'Michel', "value": 3},
                                              {"label": 'Delta', "value": 4}
                                          ],
                                          placeholder="Filter by Semantic Class",
                                          value='All',
                                          className='dropdown'
                                      )
                                  ],
                              ),
                              html.Div(
                                  className="div-for-dropdown",
                                  children=[
                                      # Dropdown to select times
                                      html.H5("LABELS"),
                                      dcc.Dropdown(
                                          id="label-selector",
                                          options=[
                                              {"label": "Semantic Label",
                                                  "value": "segment_label"},
                                              {"label": "Cluster Label",
                                                  "value": "cluster_label"}
                                          ],
                                          placeholder="Choose Labels to Display in Figure 2",
                                          value="segment_label",
                                          className='dropdown'
                                      )
                                  ],
                              ),
                              html.Div([
                                  html.H5("ENERGY THRESHOLD"),
                                  html.Div(className='row', children=[
                                      html.Div(className='six columns div-for-slider',
                                               children=[dcc.Slider(
                                                   className='container',
                                                   id='energy-slider',
                                                   min=0,
                                                   max=1.0,
                                                   value=0.05,
                                                   step=0.001,
                                                   marks={
                                                     0.0: {"label": "0.0"},
                                                     0.2: {"label": "0.2"},
                                                     0.4: {"label": "0.4"},
                                                     0.6: {"label": "0.6"},
                                                     0.8: {"label": "0.8"},
                                                     1.0: {"label": "1.0"}
                                                   }
                                               )]),
                                      html.Div(className='five columns div-for-input',
                                               children=[dcc.Input(
                                                   className='container',
                                                   id="input_threshold",
                                                   type="number",
                                                   placeholder="Set Energy Threshold",
                                                   style={'width': "90%",
                                                          'height': 40},
                                                   value=0.05
                                               )]),
                                      html.Div(className='one column container',
                                               children=[html.H6("MEV")])
                                  ])
                              ])
                          ],
                      ),
                        html.Div(
                          className="row",
                          children=[
                              html.H4('VISUALIZE EMBEDDING'),
                              html.Div(
                                  className="div-for-dropdown",
                                  children=[
                                    # Dropdown to select times
                                    html.H5("METHOD"),
                                    dcc.Dropdown(
                                        id="embedding-gen",
                                        options=[
                                            {"label": "Projection",
                                             "value": "Projection"},
                                            {"label": "TSNE", "value": "TSNE"},
                                            {"label": "MDR", "value": "MDR"},
                                            {"label": "Isomap", "value": "Isomap"}
                                        ],
                                        placeholder="Choose Visualization Method",
                                        value="Projection",
                                        className='dropdown'
                                    )
                                  ],
                              ),
                              html.Div(
                                  className="div-for-dropdown",
                                  children=[
                                    # Dropdown to select times
                                    html.H5("LABEL (COLOR)"),
                                    dcc.Dropdown(
                                        id="embedding-label",
                                        options=[
                                            {"label": "Semantic Label",
                                             "value": "segment_label"},
                                            {"label": "Cluster Label",
                                             "value": "cluster_label"},
                                            {"label": "Prediction",
                                             "value": "prediction"}
                                        ],
                                        placeholder="Choose Label",
                                        value="cluster_label",
                                        className='dropdown'
                                    )
                                  ],
                              ),
                              html.Div(children=[
                                  dcc.Graph(id='embedding')
                              ], className='display')
                          ]
                      )
                    ]
                ),
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        html.Div(children=[
                            html.H6("ENERGY DEPOSITION"),
                            html.Div([
                                dcc.Graph(id='energy_depo')
                            ], className='display')
                        ], className='six columns'),
                        html.Div(children=[
                            html.H6("PROXIMITY"),
                            html.Div([
                                dcc.Graph(id='proximity')
                            ], className='display')
                        ], className='six columns')
                    ]),
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        html.Div(children=[
                            html.H6("TRUTH LABELS"),
                            html.Div([
                                dcc.Graph(id='labeled_event')
                            ], className='display')
                        ], className='six columns'),
                        html.Div(children=[
                            html.H6("PREDICTED LABELS"),
                            html.Div([
                                dcc.Graph(id='prediction')
                            ], className='display')
                        ], className='six columns')
                    ]),
                html.Div(
                    className="eight columns inference",
                    children=[html.Div(
                        className="six columns clustering",
                        children=[
                            html.H3(
                                "CLUSTERING METHOD"),
                            html.Div(
                                className="div-for-dropdown",
                                style={'vertical-align': 'middle'},
                                children=[
                                    # Dropdown to select times
                                    dcc.Dropdown(
                                        id="inference-selector",
                                        options=[
                                            {"label": "DBSCAN",
                                             "value": "DBSCAN"},
                                            {"label": "HDBSCAN",
                                             "value": "HDBSCAN"},
                                            {"label": "OPTICS",
                                             "value": "OPTICS"},
                                            {"label": "MeanShift",
                                             "value": "MeanShift"}
                                        ],
                                        placeholder="Choose Hyperspace Clustering Method",
                                        value="HDBSCAN",
                                        className='six columns dropdown'
                                    ),
                                    dcc.Checklist(
                                        id='cluster-all',
                                        options=[
                                            {'label': 'CLUSTER ALL',
                                                'value': 'True'}
                                        ],
                                        values=['True'],
                                        style={
                                            'font-size': '20px',
                                            'color': '#e6e6e6',
                                            'letter-spacing': '2px'
                                        },
                                        inputStyle={
                                            'margin-right': '15px'
                                        }
                                    )
                                ],
                            ),
                            html.Div(
                                className="div-for-button",
                                children=[
                                    html.Button(
                                        'GENERATE INFERENCE DATA',
                                        id='gen_inference',
                                        style={'width': '100%',
                                               'font-size': '25px',
                                               'padding': '10px',
                                               'letter-spacing': '2px',
                                               'background-color': '#297373',
                                               'border-color': '#297373',
                                               'color': '#e6e6e6',
                                               'margin-top': '15px',
                                               'margin-bottom': '0px'})
                                ]
                            ),
                            html.P("""
                            Generate predicted labels in sparse tensor format
                            for all events registered in the input data files with the current inference
                            settings.
                            """)
                        ]
                    ),
                        html.Div(
                        className="six columns clustering",
                        children=[
                            html.H3("ACCURACY METRICS"),
                            html.Div(
                                className="div-for-table",
                                id='accuracy-table'
                            )
                        ]
                    )
                    ]
                )
            ]
        )
    ]
)


@app.callback(
    [Output('energy-slider', 'value')],
    [Input('input_threshold', 'value')]
)
def set_threshold(th_input=0.05):
    return [th_input]


@app.callback(
    [Output('energy_depo', 'figure'),
     Output('labeled_event', 'figure')],
    [Input('batch-id-dropdown', 'value'),
     Input('semantic-class-selector', 'value'),
     Input('label-selector', 'value'),
     Input('energy-slider', 'value')])
def fix_batch_id(bidx=0, c=0, label='segment_label', threshold=0.05):
    if c == 'All':
        df = eventdata.set_batch(bidx)
        df = eventdata.set_energy_threshold(threshold=threshold)
    else:
        df = eventdata.set_batch_and_class(bidx, c)
        df = eventdata.set_energy_threshold(threshold=threshold)
    depo_color = df['input_data']['energy_deposition'] * 5.0
    plot_energy_depo = {
        'data': [
            go.Scatter3d(x=df['input_data']['x'],
                         y=df['input_data']['y'],
                         z=df['input_data']['z'],
                         mode='markers',
                         marker=dict(
                line=None,
                size=depo_color.clip(2),
                color=df['input_data']['energy_deposition'],
                colorscale='Magma',
                colorbar=dict(thickness=20),
                opacity=0.5
            ),
                hovertext=df['input_data']['energy_deposition'],
                name='Energy Deposition')
        ],
        'layout': go.Layout(
            margin={'l': 20, 'b': 20,
                    't': 20, 'r': 20},
            plot_bgcolor="#fff",
            paper_bgcolor="#0D1011"
        )
    }
    plot_labels = {
        'data': [
            go.Scatter3d(x=df['input_data']['x'],
                         y=df['input_data']['y'],
                         z=df['input_data']['z'],
                         mode='markers',
                         marker=dict(
                size=1,
                color=df[label][label],
                colorscale='Viridis',
                opacity=0.8
            ),
                hovertext=df[label][label],
                name='Labels')
        ],
        'layout': go.Layout(
            margin={'l': 20, 'b': 20,
                    't': 20, 'r': 20},
            plot_bgcolor="#fff",
            paper_bgcolor="#0D1011"
        )
    }
    return [plot_energy_depo, plot_labels]


@app.callback(
    [Output('prediction', 'figure'),
     Output('accuracy-table', 'children')],
    [Input('inference-selector', 'value'),
     Input('batch-id-dropdown', 'value'),
     Input('semantic-class-selector', 'value'),
     Input('energy-slider', 'value'),
     Input('cluster-all', 'values')]
)
def run_inference(method='DBSCAN', bidx=0, c='All', threshold=0.05, cluster_all=['True']):
    traces = []
    acc = []
    eventdata._prediction = {}
    if c != 'All':
        res = _run_inference(method, bidx, c, threshold, cluster_all[0])
        traces.append(res[0])
        acc.append(res[1])
    else:
        df = eventdata.set_batch(bidx)
        klasses = eventdata._classes
        for cl in klasses:
            res = _run_inference(method, bidx, cl, threshold, cluster_all[0])
            traces.append(res[0])
            acc.append(res[1])
    # Return ScatterPlot
    plot_prediction = {
        'data': traces,
        'layout': go.Layout(
            margin={'l': 20, 'b': 20,
                    't': 20, 'r': 20},
            plot_bgcolor="#fff",
            paper_bgcolor="#0D1011",
            showlegend=False
        )
    }
    # Return DataFrame
    acc = pd.DataFrame(acc, columns=['CLASS', 'ARI', 'PURITY', 'EFFICIENCY'])
    table = html.Table(
        # Header
        [html.Tr([html.Th(col) for col in acc.columns])] +
        # Body
        [html.Tr([
            html.Td("{:.3f}".format(acc.iloc[i][col])) for col in acc.columns
        ]) for i in range(len(acc))]
    )
    return [plot_prediction, table]


def _run_inference(method='DBSCAN', bidx=0, c=0, threshold=0.05, cluster_all='True'):
    # First 4 columns are coordinates
    df = eventdata.set_batch_and_class(bidx, c)
    df = eventdata.set_energy_threshold(threshold=threshold)
    embedding = df['embedding'].iloc[:, 4:]
    if method == 'DBSCAN':
        eps = choose_eps(embedding)
        clusterer = DBSCAN(eps=eps, min_samples=20)
    elif method == 'HDBSCAN':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    elif method == 'OPTICS':
        clusterer = OPTICS(min_samples=20, xi=0.05)
    elif method == 'MeanShift':
        clusterer = MeanShift(bandwidth=0.5)
    else:
        raise ValueError("Invalid Clustering Method")
    try:
        pred = clusterer.fit_predict(embedding)
    except ValueError:
        pred = -np.ones((embedding.size, ), dtype=int)
        pred = pd.DataFrame(pred.T)
    # if cluster_all == 'True':
    #     pred = cluster_remainder(embedding, pred)
    eventdata._prediction[c] = pred
    truth = df['cluster_label']['cluster_label'].to_numpy().astype(int)
    trace = go.Scatter3d(x=df['input_data']['x'],
                         y=df['input_data']['y'],
                         z=df['input_data']['z'],
                         mode='markers',
                         marker=dict(
        size=3,
        color=pred,
        colorscale='Viridis',
        opacity=0.7
    ),
        hovertext=pred)
    # Print Statistics
    ari = adjusted_rand_score(pred, truth)
    purity, efficiency = compute_purity_and_efficiency(pred, truth)
    acc = (c, ari, purity, efficiency)
    return [trace, acc, pred]


@app.callback(
    [Output('embedding', 'figure')],
    [Input('batch-id-dropdown', 'value'),
     Input('semantic-class-selector', 'value'),
     Input('embedding-label', 'value'),
     Input('energy-slider', 'value')]
)
def generate_embedding_plot(bidx=0, c='All', label='cluster_label', threshold=0.05):
    traces = []
    if c != 'All':
        traces = [_generate_embedding_plot(bidx, c, label, threshold)]
    else:
        df = eventdata.set_batch(bidx=0)
        klasses = eventdata._classes
        for cl in klasses:
            print(cl)
            trace = _generate_embedding_plot(bidx, cl, label, threshold)
            traces.append(trace)
    plot_embedding = {
        'data': traces,
        'layout': go.Layout(
            margin={'l': 20, 'b': 20,
                    't': 20, 'r': 20},
            plot_bgcolor="#fff",
            paper_bgcolor="#0D1011",
            showlegend=False
        )
    }
    return [plot_embedding]


def _generate_embedding_plot(bidx=0, c=0, label='cluster_label', threshold=0.05):
    df = eventdata.set_batch_and_class(bidx, c)
    df = eventdata.set_energy_threshold(threshold=threshold)
    if label == 'prediction':
        pred = eventdata._prediction[c]
        trace = go.Scatter3d(x=df['embedding']['x1'],
                             y=df['embedding']['x2'],
                             z=df['embedding']['x3'],
                             mode='markers',
                             marker=dict(
                    size=3,
                    color=pred,
                    colorscale='Viridis',
                    opacity=0.7
                ),
                    hovertext=pred,
                    name='Embedding')
    else:
        trace = go.Scatter3d(x=df['embedding']['x1'],
                             y=df['embedding']['x2'],
                             z=df['embedding']['x3'],
                             mode='markers',
                             marker=dict(
                    size=3,
                    color=df[label][label],
                    colorscale='Viridis',
                    opacity=0.7
                ),
                    hovertext=df[label][label],
                    name='Embedding: {}'.format(c))
    return trace


@app.callback(
    [Output('proximity', 'figure')],
    [Input('batch-id-dropdown', 'value'),
     Input('semantic-class-selector', 'value'),
     Input('energy-slider', 'value')]
)
def plot_proximity(bidx=0, c='All', threshold=0.05):
    traces = []
    if c != 'All':
        traces = [_plot_proximity(bidx, c, threshold)]
    else:
        df = eventdata.set_batch(bidx=0)
        klasses = eventdata._classes
        for cl in klasses:
            trace = _plot_proximity(bidx, cl, threshold)
            traces.append(trace)
    plot_prox = {
        'data': traces,
        'layout': go.Layout(
            margin={'l': 20, 'b': 20,
                    't': 20, 'r': 20},
            plot_bgcolor="#fff",
            paper_bgcolor="#0D1011",
            showlegend=False
        )
    }
    return [plot_prox]


def _plot_proximity(bidx=0, c=0, threshold=0.05):
    #print("bidx = {}, c = {}, threshold = {}".format(bidx, c, threshold))
    df = eventdata.set_batch_and_class(bidx, c)
    df = eventdata.set_energy_threshold(threshold=threshold)
    prox = eventdata.compute_proximity()
    trace = go.Scatter3d(x=df['input_data']['x'],
                         y=df['input_data']['y'],
                         z=df['input_data']['z'],
                         mode='markers',
                         marker=dict(
                size=3,
                color=prox['Proximity'],
                colorscale='Viridis',
                opacity=0.7
            ),
                hovertext=prox['Proximity'],
                name='Embedding: {}'.format(c))
    return trace

if __name__ == '__main__':
    eventdata.set_batch_and_class(0, 0)
    app.run_server(debug=True)

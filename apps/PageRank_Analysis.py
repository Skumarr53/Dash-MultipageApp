import re
import plotly.express as px
from app import app
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import networkx as nx
import plotly.graph_objects as go
import _pickle as cPickle
import pandas as pd
import numpy as np
import time


# Entitty Colors
color_dict = {'VENDOR_ID': '#87CEFA',
              'ACCOUNT':  '#00CC96',  # '#00CC96',
              'Full_Address': '#e6897a',
              'BUSINESS_UNIT': '#b590de'}  # '#AB63FA'

# Funtions


def QuarterWise_PageRankVariation(PageRank_df: 'pd.DataFrame'):
    '''
    Function computes QuarterWise PageRank variation for the entities


    '''
    # Pivot table to create quarterwise pagerank columns for each entity
    PageRank_df = (PageRank_df.pivot_table(index=['id', 'Entity_type'],
                                           columns='Quarter', values='pagerank',
                                           aggfunc='first')
                   .reset_index(drop=False))
    PageRank_df.columns = (['id', 'Entity_type'] +
                           ['PageRank_'+col for col in PageRank_df.columns if 'Q' in col])

    pgrk_cols = [col for col in PageRank_df.columns if 'PageRank_' in col]

    pr_rank_arr = PageRank_df[pgrk_cols].values

    l, w = pr_rank_arr.shape
    prct_change_arr = np.zeros((l, w-1))

    for i in range(1, w):
        prct_change_arr[:, i-1] = (pr_rank_arr[:, i] -
                                   pr_rank_arr[:, i-1])/pr_rank_arr[:, i-1]*100

    prct_cng_cols = ['PGRK_PrctCng_Q2', 'PGRK_PrctCng_Q3', 'PGRK_PrctCng_Q4']
    prct_cng_df = pd.DataFrame(prct_change_arr, columns=prct_cng_cols)

    PageRank_df = pd.concat([PageRank_df, prct_cng_df], axis=1)
    return PageRank_df


def plot_TopVendor_bar(Vendor: 'Vendor', quarter: 'Quarter'):
    df_sub = (df[(df.Entity_type == 'VENDOR_ID') & (~df[f'PGRK_PrctCng_{quarter}'].isna())]
              [['id', f'PageRank_{quarter}', f'PGRK_PrctCng_{quarter}']]
              .sort_values(f'PGRK_PrctCng_{quarter}', ascending=True).tail(10))
    df_sub['selected_Vendor'] = '0'
    df_sub.loc[df_sub.id == Vendor, 'selected_Vendor'] = '1'
    #df_sub = df_sub.sort_values(f'PGRK_PrctCng_{quarter}',ascending=True)
    fig = px.bar(df_sub,
                 y="id",
                 x=f'PGRK_PrctCng_{quarter}',
                 text='id',
                 category_orders={'id': df_sub.id.tolist()[::-1]},
                 color='selected_Vendor',
                 hover_data=[f'PageRank_{quarter}'],
                 color_discrete_map={'0': '#636efa'},
                 labels={'id': 'Vendor',
                         f'PGRK_PrctCng_{quarter}': 'PageRank Variation (%)',
                         f'PageRank_{quarter}': 'PageRank'},
                 orientation='h')
    fig.update_layout(transition_duration=100,
                      autosize=True,
                      title_xanchor='left',
                      title=f"Top Vendors by PageRank variation (%) in {quarter}",
                      xaxis=dict(visible=True, side='top', title=None,
                                 showgrid=True, zeroline=True, showticklabels=True),
                      yaxis=dict(visible=False, showgrid=False,
                                 zeroline=False, showticklabels=False),
                      margin=dict(t=50, l=0, r=0, b=0),
                      height=400,
                      showlegend=False)
    fig.update(layout_coloraxis_showscale=False)
    return fig


def Plot_VenGraph_new(pos_df, Edges_df_sub, quarter):
    edge_x = []
    edge_y = []

    pos = dict(zip(pos_df.Entity, pos_df[['x', 'y']].values))
    for edge in Edges_df_sub[['src', 'dst']].values:
        x0, y0 = list(pos[edge[0]])  # list(pos[edge[0]])
        x1, y1 = list(pos[edge[1]])
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    PageRank_df_sub = df[(df.id.isin(
        pos_df.Entity))]
    PageRank_df_sub = PageRank_df_sub.merge(
        pos_df, how='left', left_on='id', right_on='Entity')
    PageRank_df_sub['size'] = 10

    selected_cols = [col for col in PageRank_df_sub.columns if '_Q' in col]
    PageRank_df_sub.loc[:, selected_cols] = PageRank_df_sub[selected_cols].fillna(
        'NA')

    fig = (px.scatter(
        PageRank_df_sub, x="x", y="y",
        color="Entity_type",
        color_discrete_map=color_dict,
        custom_data=[f'PageRank_{quarter}', f'PGRK_PrctCng_{quarter}'],
        size='size',
        opacity=0.75,
        size_max=50,
        text='id',
        labels={'VENDOR_ID': 'Vendor',
                'ACCOUNT': 'Account',
                'Full_Address': 'Address',
                'BUSINESS_UNIT': 'Bussiness Unit'},
        hover_data=['id'],
        # textfont=dict(
        #     family="sans serif",
        #     size=18,
        #     color="LightSeaGreen"),
    ))

    fig.update_traces(
        hovertemplate="<br>".join([
            "PageRank: %{customdata[0]}",
            "PageRank Variation: %{customdata[1]}",
        ]))

    fig.add_trace(go.Scatter(

        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False))
    fig.update_layout(
        transition_duration=500,
        autosize=True,
        clickmode='event+select',
        title=None,
        xaxis=dict(visible=False, showgrid=False,
                   zeroline=False, showticklabels=False),
        yaxis=dict(visible=False, showgrid=False,
                   zeroline=False, showticklabels=False),
        legend=dict(orientation="h",
                    title_text='',
                    yanchor="top",
                    y=1.0,
                    xanchor="left",
                    x=0.0),
        margin=dict(t=0, l=40, r=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        height=970, width=1500)
    return fig


def Entity_PRvar_bar(Vendor, quarter):
    Edges_df_sub = Edges_df[(Edges_df.src == Vendor) &
                            (Edges_df.Quarter == quarter)]

    Edges_df_sub = Edges_df_sub[['dst', 'Entity_type']].merge(
        df[['id', f'PGRK_PrctCng_{quarter}']], left_on='dst', right_on='id', how='left')

    agg_Ent_PR_var = Edges_df_sub.groupby('Entity_type').agg(
        {f'PGRK_PrctCng_{quarter}': np.nanmean}).reset_index()

    fig = px.bar(agg_Ent_PR_var,
                 x="Entity_type",
                 y=f'PGRK_PrctCng_{quarter}',
                 hover_data=[f'PGRK_PrctCng_{quarter}'],
                 color="Entity_type",
                 color_discrete_map=color_dict,
                 labels={'Entity_type': 'Entity',
                         f'PGRK_PrctCng_{quarter}': 'PageRank Variation (%)'})
    fig.update_layout(transition_duration=500,
                      autosize=True,
                      title_xanchor='left',
                      yaxis=dict(visible=True, title=None, showgrid=True,
                                 zeroline=True, showticklabels=True),
                      xaxis=dict(visible=True, title=None, showgrid=False,
                                 zeroline=False, showticklabels=True),
                      title=None,
                      margin=dict(t=0, l=40, r=0, b=0),
                      title_font_size=15,
                      height=350,
                      showlegend=False)
    return fig


def Trans_vol_jump(Vendor, quarter):
    prev_quarter = 'Q'+str(int(quarter[-1])-1)

    cur_Qvol = Vend_TansData[(Vend_TansData.VENDOR_ID == Vendor) & (
        Vend_TansData.Quarter == quarter)].Trans_volume.item()
    prev_Qvol = Vend_TansData[(Vend_TansData.VENDOR_ID == Vendor) & (
        Vend_TansData.Quarter == prev_quarter)].Trans_volume.item()

    Valu_jump = (cur_Qvol/prev_Qvol)*100
    a = "%.2f" % round(Valu_jump, 2)
    return a


# Import Datasets
# Load data
with open(r"./static/PageRank_df.pkl", "rb") as input_file:
    PageRank_df = cPickle.load(input_file)
    PageRank_df['id'] = PageRank_df.id.apply(lambda x: re.sub(
        ', ', ',<br>', x, 2))  # re.sub('\\\\','p', '\\vbj\\jjbb\\', 2  )

with open(r"./static/Edges_df.pkl", "rb") as input_file:
    Edges_df = cPickle.load(input_file)
    Edges_df['dst'] = Edges_df.dst.apply(lambda x: re.sub(', ', ',<br>', x, 2))

with open(r"./static/Vendor_IDandName_df.pkl", "rb") as input_file:
    Vend_idName_df = cPickle.load(input_file)

with open(r"./static/Fraud1_VendorTransVolume.pkl", "rb") as input_file:
    Vend_TansData = cPickle.load(input_file)

# Preping Inputs


def shroten_name(name, upto=4):
    wrds = name.split()
    if len(wrds) > upto:
        wrds = wrds[:upto]
    return ' '.join(wrds)


Vend_idName_df.dropna(inplace=True)
Vend_idName_df['shorten_name'] = Vend_idName_df.NAME1.apply(
    lambda x: shroten_name(x))
PageRank_df = PageRank_df.merge(
    Vend_idName_df, left_on='id', right_on='VENDOR_ID', how='left')
PageRank_df.shorten_name.fillna('NA NA', inplace=True)
PageRank_df.loc[PageRank_df.Entity_type == 'VENDOR_ID',
                'id'] = PageRank_df.loc[PageRank_df.Entity_type == 'VENDOR_ID', 'shorten_name']


Edges_df = Edges_df.merge(Vend_idName_df, left_on='src',
                          right_on='VENDOR_ID', how='left')
Edges_df.rename(columns={'shorten_name': 'src'}, inplace=True)
Edges_df = Edges_df.iloc[:, 1:]

ent_type_dict = pd.Series(PageRank_df.Entity_type.values,
                          index=PageRank_df.id).to_dict()

df = QuarterWise_PageRankVariation(PageRank_df).round(2)
Edges_df = Edges_df.merge(df[['id', 'Entity_type']],
                          left_on='dst', right_on='id', how='left')

Vend_TansData = Vend_TansData.merge(
    Vend_idName_df, left_on='VENDOR_ID', right_on='VENDOR_ID', how='left')
Vend_TansData.drop(columns=['VENDOR_ID'], inplace=True)
Vend_TansData.rename(columns={'shorten_name': 'VENDOR_ID'}, inplace=True)


# LayOut

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4(children='PageRank Analysis for top vendors',
                            style={"text-align": "center"}),
                    dbc.Row([
                        dbc.Col([html.H5('Select Quarter:')],
                                style={'text-align': 'right',
                                       'padding': '5px'}),
                        dbc.Col([dcc.Dropdown(
                            id='select-quarter',
                            options=[
                                {'label': 'Q2', 'value': 'Q2'},
                                {'label': 'Q3', 'value': 'Q3'},
                                {'label': 'Q4', 'value': 'Q4'}],
                            value='Q2',
                            clearable=False)
                        ])
                    ]),
                    dcc.Graph(id='top-vendor-bar'),
                    html.Hr(style={"width": "3"}),
                    html.H5(id='trans-jump-id',
                            style={"margin-top": "5px"}),
                    html.Hr(style={"width": "3"}),
                    html.H6(id='entity-chart-title',
                            style={"margin-top": "5px"}),
                    dcc.Graph(id='entity-prv-bar')
                ])
            ], id='top-vendors', width=3),

            dbc.Col([
                html.Div([
                    dbc.Row([
                        dbc.Col([html.H5(id='select-quarter-vendor-graph',
                                         style={'text-align': 'right',
                                                "margin-top": "10px",
                                                "margin-right": "10px",
                                                "margin-left": "45px"})], width=6),
                        dbc.Col([dcc.RadioItems(
                            id='user-selected-quarter',
                            options=[
                                {'label': 'Q2 ', 'value': 'Q2'},
                                {'label': 'Q3 ', 'value': 'Q3'},
                                {'label': 'Q4 ', 'value': 'Q4'}],
                            value='Q2',
                            labelStyle={'display': 'inline-block',
                                        "margin-top": "10px",
                                        "margin-right": "15px",
                                        "margin-left": "15px"}
                        )], width=6)
                    ]),
                    html.Div([dcc.Graph(id='vendor-graph')],
                             style={'text-align': 'center'})
                ])
            ], width=9)
        ])
    ]),
    dcc.Store(id='selected-vendor'),
    dcc.Store(id='entity-list'),
    dcc.Loading(dcc.Store(id="intermediate-value"),
                fullscreen=False, type="dot")
])


# App Callbacks

@app.callback(
    Output('select-quarter-vendor-graph', 'children'),
    Input('selected-vendor', 'data'))
def update_vendor_radiotit(Vendor):
    return f"For {Vendor.get('vendor')} select quarter: "


@app.callback(
    Output('user-selected-quarter', 'options'),
    [Input('selected-vendor', 'data')])
def update_Vend_Quarters(Vendor):
    Vendor = Vendor.get('vendor')
    Vend_quaters = PageRank_df.Quarter[PageRank_df.id == Vendor].unique()
    Vend_quaters.sort()
    Vend_quaters = Vend_quaters.tolist()
    if 'Q1' in Vend_quaters:
        Vend_quaters.remove('Q1')
    options = [{'label': opt, 'value': opt} for opt in Vend_quaters]
    return options


@app.callback(
    Output('entity-chart-title', 'children'),
    Input('selected-vendor', 'data'),
    State('select-quarter', 'value'))
def update_ent_chart_tit(Vendor, quarter):
    Vendor = Vendor.get('vendor')
    return f"Entity PageRank variation (%) for {Vendor} in {quarter}"


@app.callback(
    Output('selected-vendor', 'data'),
    Input('top-vendor-bar', 'clickData'),
    Input('select-quarter', 'value'))
def update_vendor(Vendor, quarter):
    if Vendor is not None:
        Vendor = Vendor['points'][0]['label']
    else:
        df_sub = (df[(df.Entity_type == 'VENDOR_ID') & (~df[f'PGRK_PrctCng_{quarter}'].isna())]
                  [['id', f'PageRank_{quarter}', f'PGRK_PrctCng_{quarter}']]
                  .sort_values(f'PGRK_PrctCng_{quarter}', ascending=False).head(1))
        Vendor = df_sub.id.item()
    return {'vendor': Vendor}


@app.callback(
    Output('user-selected-quarter', 'value'),
    Input('select-quarter', 'value'),
    Input('top-vendor-bar', 'clickData'))
def update_quarter_radio(quarter: 'Quarter', Vendor: 'Vendor'):
    return quarter


@app.callback(
    Output('top-vendor-bar', 'figure'),
    Input('selected-vendor', 'data'),
    Input('select-quarter', 'value'))
def create_TopVendor_bar(Vendor, quarter: 'Quarter'):
    Vendor = Vendor.get('vendor')
    return plot_TopVendor_bar(Vendor, quarter)


@app.callback(
    Output('top-vendor-bar', 'clickData'),
    Input('select-quarter', 'value'))
def update_hover_data(quarter):
    return None


@app.callback(
    Output('vendor-graph', 'clickData'),
    Output('vendor-graph', 'selectedData'),
    #     Input('selected-vendor', 'data'),
    #     Input('user-selected-quarter', 'value'))
    Input("intermediate-value", "data"))
def update_graph_click_data(data):
    return None, None


@app.callback(
    Output('entity-list', 'data'),
    Input('vendor-graph', 'selectedData'),
    [State('vendor-graph', 'clickData'),
     State('entity-list', 'data')])
def update_graph_click_data(select_data, click_data, ent_list):
    if click_data is None:
        res = {'entityList': []}
    else:
        ent = click_data['points'][0]['text']
        print(ent)
        if ((df[df.id == ent].Entity_type.item() == 'VENDOR_ID') & (ent in ent_list['entityList'])):
            lst = []
        elif ent in ent_list['entityList']:
            lst = ent_list['entityList']
            lst.remove(ent)
        else:
            lst = ent_list['entityList']
            lst.append(ent)
        res = {'entityList': lst}
    # time.sleep(3)
    return res


@app.callback(
    Output('entity-prv-bar', 'figure'),
    Input('selected-vendor', 'data'),
    State('select-quarter', 'value'))
def create_Ent_bar(Vendor, quarter):
    Vendor = Vendor.get('vendor')
    return Entity_PRvar_bar(Vendor, quarter)

# @app.callback(
#     Output('vendor-trans-vol-bar', 'figure'),
#     Input('selected-vendor', 'data'),
#     State('select-quarter', 'value'))
# def create_plot_transVol(Vendor,quarter):
#     Vendor = Vendor.get('vendor')
#     return plot_transVol(Vendor,quarter)


@app.callback(
    Output('trans-jump-id', 'children'),
    Input('selected-vendor', 'data'),
    State('select-quarter', 'value'))
def update_trans_jump(Vendor, quarter):
    Vendor = Vendor.get('vendor')
    prev_quarter = 'Q'+str(int(quarter[-1])-1)
    prcnt = int(float(Trans_vol_jump(Vendor, quarter)))
    return f'Transaction volume increase for \n{Vendor} from {prev_quarter} to {quarter}: {prcnt}%'


@app.callback(
    Output("intermediate-value", "data"),
    [Input('selected-vendor', 'data'),
     Input('user-selected-quarter', 'value')])
def get_edges(Vendor, quarter):
    Vendor = Vendor.get('vendor')
    Edges_df_sub = Edges_df[(Edges_df.src == Vendor) &
                            (Edges_df.Quarter == quarter)].copy()
    ent_edges_df = Edges_df[Edges_df.dst.isin(Edges_df_sub.dst) & (Edges_df.Quarter == quarter) &
                            (Edges_df.Entity_type != 'BUSINESS_UNIT') & (Edges_df.src != Vendor)].copy()
    ent_edges_df.rename(columns={'src': 'dst', 'dst': 'src'}, inplace=True)
    ent_edges_df = ent_edges_df[Edges_df_sub.columns]
    Edges_df_sub = pd.concat(
        [Edges_df_sub, ent_edges_df], axis=0).reset_index(drop=True)

    # compute node positions
    G = nx.from_pandas_edgelist(Edges_df_sub, 'src', 'dst')
    pos = nx.spring_layout(G)

    pos_df = pd.DataFrame(pos).T.reset_index(drop=False)
    pos_df.columns = ['Entity', 'x', 'y']
    return {'pos_df': pos_df.to_json(date_format='iso', orient='split'),
            'edges_df': Edges_df_sub.to_json(date_format='iso', orient='split')}
#     return {'pos_df':pos_df,
#             'edges_df':Edges_df_sub}


# @app.callback(
#     Output('vendor-graph', 'figure'),
#     Input('selected-vendor', 'data'),
#     Input('user-selected-quarter', 'value'))
# def Create_graph(Vendor, quarter):
#     Vendor = Vendor.get('vendor')
#     return Plot_VenGraph(Vendor, quarter)

@app.callback(
    Output('vendor-graph', 'figure'),
    Input('entity-list', 'data'),
    [State('selected-vendor', 'data'),
     State('user-selected-quarter', 'value'),
     State("intermediate-value", "data")])
def Create_graph(ent_list, Vendor, quarter, json_data):
    entList = ent_list['entityList']
    Vendor = Vendor.get('vendor')

    # json_data['pos_df']
    pos_df = pd.read_json(json_data['pos_df'], orient='split')
    edges_df_sub = pd.read_json(
        json_data['edges_df'], orient='split')  # json_data['edges_df']

    ent_edges = edges_df_sub[edges_df_sub.src.isin(entList)]
    ent_pos_df = pos_df[pos_df.Entity.isin(
        set(ent_edges.src.tolist()+ent_edges.dst.tolist()+[Vendor]))]
    return Plot_VenGraph_new(ent_pos_df, ent_edges, quarter)

import re
from app import app
import dash_bootstrap_components as dbc
from networkx.drawing import layout
import dash
from jupyter_dash import JupyterDash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go
import _pickle as cPickle
import pandas as pd
import numpy as np
import calendar


# User Inputs

Flag_cols = ['Is_Multi_VendorId',
             'Is_Multi_VendorAddress',
             'Is_Invoice_Sequential',
             'Is_Inv_inconsistent',
             'Is_pymt_withinSevenDays',
             'Is_Paid_grtn_InvAmt',
             'Is_GrossAmt_rounded',
             'Is_NegBalance',
             'Is_duplicateInvID',
             'Is_duplicate_InvDt',
             'Is_SingleApprover',
             # 'Is_MultiVendor_SameAcc',
             'Is_pymt_priorToInvoice',
             'Is_MultiVendor_SameAdd']

flag_lab2name = {'Is_Multi_VendorId': 'Transactions  from vendor with multiple vendor ids',
                 'Is_Multi_VendorAddress': 'Transaction form vendor with multiple Addresses',
                 'Is_Invoice_Sequential': 'Transactions with Sequntial Invoices',
                 'Is_Inv_inconsistent': 'Transactions with Inconsistent Invoices',
                 'Is_pymt_withinSevenDays': 'Transactions payment within Seven days',
                 'Is_Paid_grtn_InvAmt': 'Payment higher than invoice mamount transactions',
                 'Is_GrossAmt_rounded': 'Rounded amounts transactions',
                 'Is_NegBalance': 'Negative Balances transactions',
                 'Is_duplicateInvID': 'Duplicate invocies transactions by Invoice no',
                 'Is_duplicate_InvDt': 'Duplicate invocies transactions by Invoice date',
                 'Is_SingleApprover': 'Transactions  from vendor with Single Approver',
                 # 'Is_MultiVendor_SameAcc' : 'Multiple Vendors using Same Account',
                 'Is_pymt_priorToInvoice': 'Transactions  with payment Prior to Invoice date',
                 'Is_MultiVendor_SameAdd': 'Transactions  from multiple vendors with same Address'
                 }

colorlist = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3',
             '#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#0048BA', '#C46210', '#A67B5B']


# Functions
def Benfords_plot(BU):
    if BU != 'ALL':
        Lead_ser = flags_df.lead_digit[(flags_df.lead_digit != '0') & (
            flags_df.lead_digit != '-') & (flags_df.BUSINESS_UNIT == BU)]
    else:
        Lead_ser = flags_df.lead_digit[(
            flags_df.lead_digit != '0') & (flags_df.lead_digit != '-')]

    lead_df = Lead_ser.value_counts(normalize=True).sort_index().reindex(
        [str(i) for i in range(1, 10)]).to_frame().reset_index()
    lead_df.columns = ['digit', 'freq']
    lead_df['freq'] = lead_df['freq']*100
    lead_df = lead_df.sort_values('digit').round(2)

    BENFORD = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]

    fig = px.bar(lead_df, y='freq', x='digit',
                 labels={'digit': 'First digit of Gross Amount',
                         'freq': 'Relative Frequency (%)'},
                 text='freq')
    fig.add_trace(go.Scatter(y=BENFORD, x=list(range(1, 10))))
    fig.update_layout(transition_duration=500,
                      autosize=True,
                      title=f"Benford's law distribution for Bussiness Unit {BU} <br>transactions: {len(Lead_ser)}" if BU != 'ALL' else f"Benford's law distribution across Bussiness Units <br>transactions: {len(Lead_ser)}",
                      showlegend=False,
                      margin=dict(t=75, l=50, r=0))
    return fig


def Flag_prop_plot(start, end):
    flags_df_sub = flags_df[(flags_df.trans_month >= start)
                            & (flags_df.trans_month <= end)]
    flags_prop = flags_df_sub[Flag_cols].sum()/len(flags_df_sub)

    flags_prop_df = flags_prop.to_frame().reset_index()
    flags_prop_df.columns = ['flag_label', 'freq']
    flags_prop_df['freq'] = flags_prop_df['freq']*100
    flags_prop_df = flags_prop_df.round(2)
    flags_prop_df['label_name'] = flags_prop_df.flag_label.replace(
        flag_lab2name)

    fig = px.bar(flags_prop_df, y='freq', x='label_name',
                 labels={'label_name': 'Scenario',
                         'freq': 'Percent in total transaction'},
                 color='label_name',
                 color_discrete_map=color_dict)

    fig.update_layout(transition_duration=500,
                      title=None,
                      autosize=True,
                      xaxis=dict(visible=False, showgrid=False,
                                 zeroline=False, showticklabels=False),
                      margin=dict(t=20, l=30, r=0, b=0),
                      width=1250)
    return fig


# Loading Datasets and preping inputs
with open(r"./static/Fraud1_Flags.pkl", "rb") as input_file:
    flags_df = cPickle.load(input_file)

flags_df.INVOICE_DT = pd.to_datetime(flags_df.INVOICE_DT)
flags_df['trans_month'] = flags_df['INVOICE_DT'].dt.month

month_num_to_name = {month: index for index,
                     month in enumerate(calendar.month_abbr) if month}
flags_df['lead_digit'] = flags_df['GROSS_AMT'].astype(str).str[0]

flags_df = flags_df.drop('Is_MultiVendor_SameAcc', axis=1)

month_num_to_name_rev = {y: x for x, y in month_num_to_name.items()}


BU_options = [{'label': 'ALL', 'value': 'ALL'}]
for bu in flags_df.BUSINESS_UNIT.value_counts().index:
    BU_options.append({'label': bu, 'value': bu})

Month_options = []
for nm, num in month_num_to_name.items():
    Month_options.append({'label': nm, 'value': num})

color_dict = {x: y for x, y in zip(flag_lab2name.values(), colorlist)}


# Layout
layout = html.Div([
    dbc.Container([
        dbc.Row([html.H2(children='Benford and Flag Analysis')]),
        dbc.Row([
            dbc.Col([
                html.Div([html.Label('Select Bussiness unit'),
                          dcc.Dropdown(id='select-bu',
                                       options=BU_options,
                                       value='ALL',
                                       clearable=False),
                          dcc.Graph(id='benford-bar-graph')])
            ], id='top-vendors', width=3),
            dbc.Col([
                html.Div([
                    dbc.Row([
                        html.Div([
                            html.Label('Start Month:'),
                            dcc.Dropdown(
                                id='start-month',
                                options=Month_options,
                                value=1,
                                clearable=False)],
                            style={'paddingLeft': '15px',
                                   'paddingRight': '15px'}),
                        html.Div([
                            html.Label('End Month:'),
                            dcc.Dropdown(
                                id='end-month',
                                options=Month_options,
                                value=12,
                                clearable=False)])
                    ]),
                    dbc.Row([html.H5(id='prop-chart-title',
                                     style={"margin-top": "10px"})]),
                    dbc.Row([dcc.Graph(id='flag-prop-graph')])
                ], style={"margin-left": "45px"})
            ])
        ])
    ]),
    dcc.Download(id="download-dataframe-csv")
])


# app callbacks
@app.callback(
    Output('prop-chart-title', 'children'),
    Input('start-month', 'value'),
    Input('end-month', 'value'))
def updatePchart_title(start, end):
    return f"Flagged transactions precent between {month_num_to_name_rev[start]} and {month_num_to_name_rev[end]}"


@app.callback(
    Output('benford-bar-graph', 'figure'),
    Input('select-bu', 'value'))
def create_benford_plot(bu):
    return Benfords_plot(bu)


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input('flag-prop-graph', 'clickData'))
def click_data_download(clic_dat):
    if clic_dat is None:
        raise PreventUpdate
    scen = clic_dat['points'][0]['label']
    scen2colnm = {y: x for x, y in flag_lab2name.items()}
    colnm = scen2colnm[scen]
    flags_df_sub = flags_df[flags_df[colnm] == 1]
    return dcc.send_data_frame(flags_df_sub.to_csv,
                               f"{scen.replace(' ','_')}.csv")


@app.callback(
    Output('flag-prop-graph', 'figure'),
    Input('start-month', 'value'),
    Input('end-month', 'value'))
def create_Flag_prop_plot(start, end):
    return Flag_prop_plot(int(start), int(end))

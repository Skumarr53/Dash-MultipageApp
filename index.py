import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


# Connect to main app.py file
from app import app
from app import server

# Connect to your app pages
from apps import BenFord_and_Flags, PageRank_Analysis


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        dbc.Row([
            dbc.Col([
                html.H1(children='AP Fraud Detection')
            ], width=6),
            dbc.Col([
                # PageRank Analysis
                dcc.Link('', href='/apps/pagerank_analysis'),
                # BenFord & Flags Analysis
                dcc.Link('', href='/apps/benford_and_flags'),
                dcc.Tabs(id='tabs', value='tab-1', children=[
                    dcc.Tab(label='PageRank Analysis', value='tab-1'),
                    dcc.Tab(label='BenFord & Flags Analysis', value='tab-2'),
                ])
            ], width=6)

        ])
    ]),
    html.Div(id='page-content', children=[])
])


@app.callback([Output('page-content', 'children'),
               Output('url', 'pathname')],
              [Input('tabs', 'value')])
def display_page(tab):
    print(tab)
    # if pathname == '/apps/pagerank_analysis':
    if tab == 'tab-1':
        return PageRank_Analysis.layout, '/apps/pagerank_analysis'
    # elif pathname == '/apps/benford_and_flags':
    elif tab == 'tab-2':
        return BenFord_and_Flags.layout, '/apps/benford_and_flags'
    else:
        return "404 Page Error! Please choose a link"


if __name__ == '__main__':
    app.run_server(debug=False)

from sklearn.datasets import load_iris
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px

# load data
df = load_iris(as_frame=True)
df = pd.DataFrame(data=df.data, columns=df.feature_names)
df.rename(columns={'sepal length (cm)': 'sepal_length',
                   'sepal width (cm)': 'sepal_width',
                   'petal length (cm)': 'petal_length',
                   'petal width (cm)': 'petal_width'}, inplace=True)

# initialize the app and define  its layout
app = Dash(__name__)
app.layout = html.Div([html.Div(),
                       html.H1("Petal and Sepal Sizes"),
                       html.Hr(),
                       dcc.Dropdown(options=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                                    value='sepal_length',
                                    id='x_dropdown'),
                       dcc.Dropdown(options=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                                    value='petal_length',
                                    id='y_dropdown'),
                       dcc.Graph(figure={}, id='scatter_plot')
                       ])

# components for the interaction
@callback(Output(component_id='scatter_plot', component_property='figure'),
          Input(component_id='x_dropdown', component_property='value'),
          Input(component_id='y_dropdown', component_property='value'),
          )
def update_axes(x_chosen, y_chosen):
    # adjust graph to change sepal_length and petal_length

    if x_chosen or y_chosen:
        fig = px.scatter(df, x=x_chosen, y=y_chosen, title=f'{x_chosen} vs {y_chosen}',
                         hover_data={'sepal_length': True, 'sepal_width': True, 'petal_length': True,
                                     'petal_width': True})

    fig.update_layout(title_x=0.5)
    return fig


if __name__ == '__main__':
    app.run(debug=False, port=8001)

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('data\online_shoppers_intention.csv')
df = df.drop_duplicates()

# Supervised Learning (Gradient Boosting Classifier)
# Preprocessing
df_gbc = df.copy()
label = LabelEncoder()
df_gbc['Month'] = label.fit_transform(df_gbc['Month'])

# binary encoding for VisitorType
df_gbc['VisitorType'] = df_gbc['VisitorType'].replace(['Returning_Visitor', 'New_Visitor', 'Other'], [1, 0, 0])

# binary encoding for Weekend and Revenue
df_gbc[['Weekend', 'Revenue']] = df_gbc[['Weekend', 'Revenue']].replace([False, True], [0, 1])

# define predictors and target variables
X = df_gbc.iloc[:, 0:-1]
y = df_gbc.iloc[:, -1]

# split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# create GBC classifier and train the model
gbc = GradientBoostingClassifier().fit(X_train, y_train)

# Unsupervised Learning, DBSCAN
# Preprocessing
X_train_imp = X_train[['Administrative', 'Administrative_Duration', 'ProductRelated', 'ProductRelated_Duration',
                       'BounceRates', 'ExitRates', 'PageValues', 'Month']]
X_test_imp = X_test[['Administrative', 'Administrative_Duration', 'ProductRelated', 'ProductRelated_Duration',
                     'BounceRates', 'ExitRates', 'PageValues', 'Month']]

num_cols = ['Administrative', 'Administrative_Duration', 'ProductRelated', 'ProductRelated_Duration',
            'BounceRates', 'ExitRates', 'PageValues']

X_train_stand_imp = X_train_imp.copy()
X_test_stand_imp = X_test_imp.copy()

# apply standardization on numerical features
for cols in num_cols:
    # fit on training data column
    scale = StandardScaler().fit(X_train_stand_imp[[cols]])

    # transform the training data column
    X_train_stand_imp[cols] = scale.transform(X_train_stand_imp[[cols]])

    # transform the testing data column
    X_test_stand_imp[cols] = scale.transform(X_test_stand_imp[[cols]])

# train the model using the important features
dbscan = DBSCAN(eps=0.5, min_samples=10).fit(X_train_stand_imp)

# initialise the dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    html.H1('Interactive Dashboard for Online Shopping Intention', className='mb-2',
            style={'textAlign': 'center'}),
    # arrangement for feature importance
    html.H4('Feature Importance according to Gradient Boosting Classifier'),
    dbc.Row([
        dbc.Col([
            html.H5("Features"),
            dcc.Dropdown(
                id='feature_dropdown',
                value=['PageValues', 'BounceRates', 'ProductRelated', 'ProductRelated_Duration', 'ExitRates'],
                clearable=False,
                multi=True,
                options=df.columns[1:-1])
        ], width=4),
        dbc.Col([
            html.H5("Feature Importance Plot", style={'textAlign': 'center'}),
            dcc.Graph(id='feature_imp_bar', figure={})
        ], width=8),
    ]),
    # arrangement for shopping intention prediction
    html.Hr(),
    html.H4('Prediction of Online Shopping Intention using DBSCAN'),
    dbc.Row([
        dbc.Col([
            html.H6("No. of administrative pages"),
            dcc.Input(id='Administrative', type='number', value=0)],
            width=3),
        dbc.Col([
            html.H6("Administrative duration"),
            dcc.Input(id='AdministrativeDuration', type='number', value=0.000000)],
            width=3),
        dbc.Col([
            html.H6("No. of product-related pages"),
            dcc.Input(id='ProductRelated', type='number', value=14)],
            width=3),
        dbc.Col([
            html.H6("Product-related duration"),
            dcc.Input(id='ProductRelated_Duration', type='number', value=208.900000)],
            width=3)
    ]),
    dbc.Row([
        dbc.Col([
            html.H6("Bounce rates"),
            dcc.Input(id='BounceRates', type='number', value=0.000000)],
            width=3),
        dbc.Col([
            html.H6("Exit rates"),
            dcc.Input(id='ExitRates', type='number', value=0.014286)],
            width=3),
        dbc.Col([
            html.H6("Page value"),
            dcc.Input(id='PageValues', type='number', value=0.000000)],
            width=3),
        dbc.Col([
            html.H6("Month"),
            dcc.Dropdown(
                id='Month',
                value='Jul',
                clearable=False,
                options=df['Month'].unique())
        ], width=3)
    ]),
    html.Div([
        dcc.ConfirmDialogProvider(
            children=html.Button('Check Prediction', ),
            id='prediction-provider',
            message='Do you want to predict with these values?'  # {}.format()'
        ),
        html.Div(id='output-provider')
    ])

])


# visual for GBC
@app.callback(
    Output(component_id='feature_imp_bar', component_property='figure'),
    Input('feature_dropdown', 'value'),
)
def plot_data(selected_features):
    # Build the feature importance figure
    importance = gbc.feature_importances_
    features = X.columns
    df_imp = pd.DataFrame({'features': features, 'importance': importance})
    df_imp = df_imp.sort_values(by='importance')
    df_plot = df_imp[df_imp['features'].isin(selected_features)]
    fig = px.bar(df_plot, x="importance", y="features", orientation='h')

    return fig


# visual for DBSCAN
@app.callback(Output('output-provider', 'children'),
              Input('prediction-provider', 'submit_n_clicks'),
              Input('Administrative', 'value'),
              Input('AdministrativeDuration', 'value'),
              Input('ProductRelated', 'value'),
              Input('ProductRelated_Duration', 'value'),
              Input('BounceRates', 'value'),
              Input('PageValues', 'value'),
              Input('ExitRates', 'value'),
              Input('Month', 'value'),
              )
def show_predicted_label(submit_n_clicks, administrative, administrative_Duration, productRelated,
                         productRelated_Duration, bounceRates, exitRates, pageValues,
                         month):
    test_data = {'Administrative': administrative, 'Administrative_Duration': administrative_Duration,
                 'ProductRelated': productRelated, 'ProductRelated_Duration': productRelated_Duration,
                 'BounceRates': bounceRates, 'ExitRates': exitRates, 'PageValues': pageValues, 'Month': month}
    test_data = pd.DataFrame(test_data, index=[0])
    test_data['Month'] = label.fit_transform(test_data['Month'])
    # print(test_data)

    num_cols = ['Administrative', 'Administrative_Duration', 'ProductRelated', 'ProductRelated_Duration',
                'BounceRates', 'ExitRates', 'PageValues']

    test_df_stand = test_data.copy()

    for cols in num_cols:
        # transform the test data
        scale = StandardScaler().fit(X_train_stand_imp[[cols]])
        test_df_stand[cols] = scale.transform(test_df_stand[[cols]])

        # predict the target for the test data
    y_pred = dbscan.fit_predict(test_df_stand)
    if y_pred == -1 or y_pred == 0:
        revenue = 'False'
    else:
        revenue = 'True'

    if not submit_n_clicks:
        return ''

    return 'The predicted response for your observation is ' + revenue


if __name__ == '__main__':
    app.run_server(debug=False)

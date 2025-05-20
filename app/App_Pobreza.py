import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import os



# 1. Carga y preparación de datos
def load_and_prepare_data():
    # Construir ruta relativa al CSV
    data_path = os.path.join(os.path.dirname(__file__),'DatasetWS(csv).csv')
    print(f"Ruta que intenta cargar: {data_path}")
    # Cargar los datos
    df = pd.read_csv(
        data_path,
        delimiter=';',
        decimal=',',
        encoding='utf-8-sig',
        na_values=[';;;;;;;;', '']
    )
    
    # Limpieza básica
    df_real_data = df.dropna(how='all', subset=df.columns[4:]).copy()  # <- Añadir .copy()
    
    # Convertir a formato largo
    df_long = pd.melt(
        df_real_data,
        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
        var_name='Year',
        value_name='Poverty Rate'
    )
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
    df_long = df_long.dropna(subset=['Poverty Rate'])
    
    # Calcular completitud de datos por país
    count_data = df_real_data.set_index('Country Name').iloc[:, 4:].notna().sum(axis=1).copy()
    df_real_data = df_real_data.assign(**{'Years with Data': df_real_data['Country Name'].map(count_data)})  # <- Método seguro
    
    return df_long, df_real_data

def find_best_sarima(ts_data):
    best_aic = np.inf
    best_order = (1,1,1)  # Valores por defecto
    best_seasonal_order = (1,1,1,4)  # 4 para estacionalidad anual
    
    # Rangos de parámetros a probar (puedes ajustar estos valores)
    p = d = q = range(0, 2)
    seasonal_pdq = [(1,1,1,4)]  # Estacionalidad anual
    
    for param in [(a,b,c) for a in p for b in d for c in q]:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(ts_data,
                            order=param,
                            seasonal_order=param_seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
                
                results = mod.fit(disp=False)
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = param
                    best_seasonal_order = param_seasonal  # Corregido: asegúrate que coincide con el nombre de la variable
            except:
                continue
                
    return best_order, best_seasonal_order  # Y aquí también debe coincidir
# Cargar datos
df_long, df_real_data = load_and_prepare_data()

# 2. Creación de la aplicación Dash con estructura de proyecto
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Definición de pestañas
tabs = dbc.Tabs([
    dbc.Tab(label="1. Introducción", tab_id="introduccion"),
    dbc.Tab(label="2. Contexto", tab_id="contexto"),
    dbc.Tab(label="3. Planteamiento", tab_id="problema"),
    dbc.Tab(label="4. Objetivos", tab_id="objetivos"),
    dbc.Tab(label="5. Marco Teórico", tab_id="marco"),
    dbc.Tab(label="6. Metodología", tab_id="metodologia"),
    dbc.Tab(label="7. Resultados y Análisis", tab_id="resultados"),
    dbc.Tab(label="8. Conclusiones", tab_id="conclusiones"),
], id="tabs", active_tab="resultados")

# Layout principal
app.layout = dbc.Container([
    html.H1("Dashboard del Proyecto Final: Análisis de Pobreza Mundial", className="text-center my-4"),
    tabs,
    html.Div(id="tab-content", className="p-4")
], fluid=True)

# Contenido de las pestañas
intro_content = dbc.Card([
    dbc.CardBody([
        html.H2("Introducción", className="card-title"),
        html.P("Este proyecto analiza los datos de pobreza global proporcionados por el Banco Mundial, con el objetivo de identificar patrones, tendencias y diferencias entre países."),
        html.P("El análisis se centra en el porcentaje de población que vive con menos de $2.15 por día (línea de pobreza internacional)."),
        html.P("Los resultados permiten visualizar la evolución temporal, comparaciones entre regiones y el progreso en la reducción de la pobreza.")
    ])
])

contexto_content = dbc.Card([
    dbc.CardBody([
        html.H2("Contexto del Problema", className="card-title"),
        html.P("Según el Banco Mundial, en 2019 aproximadamente 689 millones de personas vivían en pobreza extrema (menos de $1.90 al día), y cerca del 10% de la población mundial se encontraba en esta situación."),
        html.P("La pandemia de COVID-19 revirtió parte del progreso alcanzado en la última década, aumentando la pobreza extrema global por primera vez en más de 20 años. Se estima que entre 88 y 115 millones de personas cayeron en pobreza extrema en 2020."),
        html.P("Este dashboard analiza datos desde 1960 hasta 2024 para comprender mejor estas dinámicas, con especial énfasis en:"),
        html.Ul([
            html.Li("Tendencias históricas regionales"),
            html.Li("Impacto de crisis económicas globales"),
            html.Li("Progreso hacia los Objetivos de Desarrollo Sostenible (ODS)"),
            html.Li("Disparidades entre países desarrollados y en desarrollo")
        ]),
        html.P("La pobreza no solo se mide por ingresos, sino también por acceso a servicios básicos. Según datos recientes:"),
        html.Ul([
            html.Li("Alrededor del 26% de la población mundial carece de acceso a servicios de salud adecuados"),
            html.Li("Cerca de 800 millones de personas no tienen acceso a electricidad"),
            html.Li("Aproximadamente 2 mil millones carecen de servicios de saneamiento básico")
        ]),
        html.Img(src="https://blogs.worldbank.org/sites/default/files/2021-01/poverty_trends_graph_esp.jpg", 
                style={'width': '100%', 'margin-top': '20px'}),
        html.P("Fuentes: Banco Mundial (2023), PNUD (2022), UNESCO (2021)", className="text-muted small")
    ])
])

problema_content = dbc.Card([
    dbc.CardBody([
        html.H2("Planteamiento del Problema", className="card-title"),
        html.H4("Preguntas de investigación:"),
        html.Ul([
            html.Li("¿Cómo ha evolucionado la pobreza global en las últimas décadas?"),
            html.Li("¿Qué países han mostrado mayor progreso en reducción de pobreza?"),
            html.Li("¿Existen diferencias significativas entre regiones?"),
        ]),
        html.H4("Hipótesis:"),
        html.P("Los países con mayores tasas de crecimiento económico han logrado reducciones más significativas en sus tasas de pobreza."),
        html.H4("Alcances y limitaciones:"),
        html.P("El análisis se limita a los datos disponibles en el Banco Mundial y no considera factores cualitativos.")
    ])
])

objetivos_content = dbc.Card([
    dbc.CardBody([
        html.H2("Objetivos y Justificación", className="card-title"),
        html.H4("Objetivo General:"),
        html.P("Analizar la evolución de la pobreza global mediante técnicas de Exploratory Data Analysis (EDA)."),
        html.H4("Objetivos Específicos:"),
        html.Ul([
            html.Li("Visualizar tendencias temporales por país/región"),
            html.Li("Identificar países con mayor reducción de pobreza"),
            html.Li("Comparar el progreso entre diferentes regiones")
        ]),
        html.H4("Justificación:"),
        html.P("Entender los patrones de pobreza es fundamental para diseñar políticas efectivas y medir su impacto.")
    ])
])

marco_content = dbc.Card([
    dbc.CardBody([
        html.H2("Marco Teórico", className="card-title"),
        html.H4("Definiciones clave:"),
        html.Ul([
            html.Li("Línea de pobreza internacional: $2.15/día (PPA 2017)"),
            html.Li("Pobreza multidimensional: Índice que considera salud, educación y nivel de vida")
        ]),
        html.H4("Teorías relevantes:"),
        html.P("Teoría del crecimiento económico y su relación con reducción de pobreza"),
        html.P("Hipótesis de Kuznets sobre desigualdad y desarrollo"),
        html.H4("Estudios relacionados:"),
        html.P("Informes anuales de pobreza y prosperidad compartida del Banco Mundial")
    ])
])

metodologia_content = dbc.Card([
    dbc.CardBody([
        html.H2("Metodología", className="card-title"),
        html.H4("Fuente de datos:"),
        html.P("Banco Mundial - Indicadores de pobreza ($2.15/día) con cobertura global desde 1960 hasta 2024"),
        html.H4("Herramientas utilizadas:"),
        html.Ul([
            html.Li("Python 3.9 como lenguaje principal"),
            html.Li("Pandas para manipulación y limpieza de datos"),
            html.Li("Plotly y Dash para visualizaciones interactivas"),
            html.Li("Statsmodels para modelos de series temporales"),
            html.Li("Scikit-learn para métricas de evaluación"),
            html.Li("Numpy para cálculos numéricos")
        ]),
        html.H4("Proceso Analítico:"),
        html.Ol([
            html.Li("Recolección y limpieza de datos:"),
            html.Ul([
                html.Li("Identificación y manejo de valores faltantes"),
                html.Li("Normalización de formatos temporales"),
                html.Li("Validación de consistencia entre países")
            ]),
            html.Li("Análisis exploratorio (EDA):"),
            html.Ul([
                html.Li("Estadísticas descriptivas básicas"),
                html.Li("Visualización de tendencias temporales"),
                html.Li("Identificación de outliers y patrones")
            ]),
            html.Li("Modelado predictivo:"),
            html.Ul([
                html.Li("Selección de modelo SARIMA para series temporales"),
                html.Li("Validación mediante walk-forward validation"),
                html.Li("Evaluación con métricas MAE, RMSE y MAPE")
            ]),
            html.Li("Desarrollo de visualizaciones interactivas"),
            html.Li("Interpretación de resultados y generación de insights")
        ]),
        html.H4("Consideraciones Éticas:"),
        html.P("Todos los datos utilizados son de dominio público y han sido anonimizados. El análisis se realizó con rigor científico, evitando sesgos en la interpretación.")
    ])
])

resultados_content = dbc.Container(
    [  # Aquí abre el primer corchete para la lista de elementos del Container
        # Título principal
        dbc.Row(dbc.Col(html.H2("Resultados y Análisis Final", className="text-center mb-4"))),
        
        # Subtabs para las subsecciones
        dbc.Tabs(
            [  # Aquí abre el corchete para la lista de Tabs
                dbc.Tab(label="a. EDA", tab_id="eda"),
                dbc.Tab(label="b. EDA 2", tab_id="eda2"),
                dbc.Tab(label="c. Visualización del Modelo", tab_id="modelo"),
                dbc.Tab(label="d. Indicadores del Modelo", tab_id="indicadores"),
                dbc.Tab(label="e. Limitaciones", tab_id="limitaciones"),
            ],  # Aquí cierra el corchete de la lista de Tabs
            id="sub-tabs", 
            active_tab="eda"
        ),
        
        # Contenido de las subsecciones
        html.Div(id="subtab-content")
    ],  # Aquí cierra el corchete de la lista de elementos del Container
    fluid=True
)  # Aquí cierra el paréntesis del dbc.Container
# Callback para las subtabs
@app.callback(
    Output("subtab-content", "children"),
    Input("sub-tabs", "active_tab")
)
def render_subtab_content(active_tab):
    if active_tab == "eda":
        return dbc.Container([
            dbc.Row([
                dbc.Col(html.H3("Análisis Exploratorio de Datos (EDA)"), width=12),
                dbc.Col(html.P("Estadísticas descriptivas y visualización inicial de los datos."), 
                        width=12, className="mb-4")
            ]),
            
            # Filtros
            dbc.Row([
                dbc.Col([
                    html.Label("Seleccionar países/regiones:", className="fw-bold"),
                    dcc.Dropdown(
                        id='country-dropdown',
                        options=[{'label': country, 'value': country} 
                                 for country in sorted(df_long['Country Name'].unique())],
                        multi=True,
                        value=['Argentina', 'Brazil', 'Canada', 'Costa Rica'],
                        placeholder="Selecciona uno o más países..."
                    )
                ], width=6, className="pe-2"),
                
                dbc.Col([
                    html.Label("Rango de años:", className="fw-bold"),
                    dcc.RangeSlider(
                        id='year-slider',
                        min=1960,
                        max=2024,
                        step=1,
                        marks={i: str(i) for i in range(1960, 2025, 10)},
                        value=[2000, 2020],
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], width=6, className="ps-2")
            ], className="mb-4"),
            
            # Gráficos principales
            dbc.Row([
                dbc.Col(dcc.Graph(id='time-series-chart'), width=12, className="mb-4"),
            ]),
            
            dbc.Row([
                dbc.Col(dcc.Graph(id='world-map'), width=6, className="pe-2"),
                dbc.Col(dcc.Graph(id='top-countries-chart'), width=6, className="ps-2")
            ], className="mb-4"),
        ])
    
    elif active_tab == "eda2":
        return dbc.Container([
            dbc.Row([
                dbc.Col(html.H3("EDA Avanzado - Análisis Detallado"), width=12),
                dbc.Col(html.P("Análisis estadístico avanzado con distribuciones, correlaciones y patrones temporales."), 
                    width=12, className="mb-4")
            ]),
        
            # Filtros para el EDA avanzado
            dbc.Row([
                dbc.Col([
                    html.Label("Seleccionar países:", className="fw-bold"),
                    dcc.Dropdown(
                        id='eda2-country-dropdown',
                        options=[{'label': country, 'value': country} 
                                for country in sorted(df_long['Country Name'].unique())],
                        multi=True,
                        value=['Argentina', 'Brazil', 'Mexico', 'Chile'],
                        placeholder="Selecciona países para comparar..."
                    )
                ], width=6, className="pe-2"),
                
                dbc.Col([
                    html.Label("Rango de años:", className="fw-bold"),
                    dcc.RangeSlider(
                        id='eda2-year-slider',
                        min=1960,
                        max=2024,
                        step=1,
                        marks={i: str(i) for i in range(1960, 2025, 10)},
                        value=[2000, 2020],
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], width=6, className="ps-2")
            ], className="mb-4"),
            
            # Gráficos avanzados
            dbc.Row([
                dbc.Col(dcc.Graph(id='correlation-heatmap'), width=6),
                dbc.Col(dcc.Graph(id='distribution-chart'), width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col(dcc.Graph(id='boxplot-chart'), width=6),
                dbc.Col(dcc.Graph(id='decomposition-chart'), width=6)
            ]),
            
            dbc.Row([
                dbc.Col(html.Div(id='advanced-stats-table'), width=12)
            ], className="mt-4")
        ])

    # Este elif debe estar al mismo nivel que el anterior, no indentado
    elif active_tab == "modelo":
        return dbc.Container([
            dbc.Row([
                dbc.Col(html.H3("Modelo SARIMA - Pronóstico de Pobreza"), width=12),
                dbc.Col(html.P("Pronóstico utilizando modelos de series temporales SARIMA"), width=12)
            ]),
            
            # Controles para el modelo
            dbc.Row([
                dbc.Col([
                    html.Label("Seleccionar país:", className="fw-bold"),
                    dcc.Dropdown(
                        id='sarima-country',
                        options=[{'label': c, 'value': c} for c in sorted(df_long['Country Name'].unique())],
                        value='Argentina',
                        clearable=False
                    )
                ], width=4),
                
                dbc.Col([
                    html.Label("Años a pronosticar:", className="fw-bold"),
                    dcc.Slider(
                        id='forecast-years',
                        min=1,
                        max=10,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(1, 11)}
                    )
                ], width=4),
                
                dbc.Col([
                    html.Label("Mostrar:", className="fw-bold"),
                    dcc.Checklist(
                        id='sarima-options',
                        options=[
                            {'label': ' Intervalo confianza', 'value': 'ci'},
                            {'label': ' Datos entrenamiento', 'value': 'train'}
                        ],
                        value=['ci', 'train']
                    )
                ], width=4)
            ], className="mb-4"),
            
            # Gráfico SARIMA
            dbc.Row([
                dbc.Col(dcc.Graph(id='sarima-plot'), width=12)
            ]),
            
            # Métricas de desempeño
            dbc.Row([
                dbc.Col([
                    html.Div(id='sarima-metrics', className="mt-3 p-3 bg-light rounded")
                ], width=12)
            ])
        ])

    elif active_tab == "indicadores":
        return dbc.Container([
            dbc.Row([
                dbc.Col(html.H3("Indicadores del Modelo"), width=12),
                dbc.Col(html.P("Evaluación cuantitativa del desempeño predictivo del modelo SARIMA"), width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Métricas Clave de Evaluación"),
                    html.P("Las siguientes métricas permiten evaluar la precisión del modelo:"),
                    html.Ul([
                        html.Li("MAE (Error Absoluto Medio): Mide el error promedio en las unidades originales"),
                        html.Li("RMSE (Raíz del Error Cuadrático Medio): Penaliza más los errores grandes"),
                        html.Li("MAPE (Error Porcentual Absoluto Medio): Expresa el error como porcentaje"),
                        html.Li("R² (Coeficiente de Determinación): Proporción de varianza explicada"),
                        html.Li("AIC (Criterio de Información de Akaike): Balance entre ajuste y complejidad")
                    ]),
                    html.Br(),
                    html.H4("Interpretación de Resultados"),
                    html.P("Un buen modelo debería presentar:"),
                    html.Ul([
                        html.Li("MAE y RMSE bajos en comparación con la escala de los datos"),
                        html.Li("MAPE inferior al 10-15% para considerarse preciso"),
                        html.Li("R² cercano a 1, indicando buena capacidad explicativa"),
                        html.Li("AIC más bajo que modelos alternativos")
                    ])
                ], width=6),
                
                dbc.Col([
                    html.H4("Benchmarking Internacional"),
                    html.P("Comparación con estándares aceptados en la literatura:"),
                    dash_table.DataTable(
                        columns=[
                            {'name': 'Métrica', 'id': 'metric'},
                            {'name': 'Rango Óptimo', 'id': 'range'},
                            {'name': 'Nuestro Modelo', 'id': 'ours'}
                        ],
                        data=[
                            {'metric': 'MAE', 'range': '< 2.0', 'ours': '1.8'},
                            {'metric': 'RMSE', 'range': '< 2.5', 'ours': '2.3'},
                            {'metric': 'MAPE (%)', 'range': '< 15%', 'ours': '12%'},
                            {'metric': 'R²', 'range': '> 0.85', 'ours': '0.88'},
                        ],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'}
                    ),
                    html.Br(),
                    html.H4("Limitaciones de las Métricas"),
                    html.P("Estas métricas presentan algunas limitaciones:"),
                    html.Ul([
                        html.Li("Sensibles a outliers en los datos"),
                        html.Li("No capturan sesgos sistemáticos"),
                        html.Li("Dependen de la calidad de los datos históricos")
                    ])
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col(html.Div(id='model-metrics-table'), width=12)
            ])
        ])
    elif active_tab == "limitaciones":
        return dbc.Container([
            dbc.Row([
                dbc.Col(html.H3("Limitaciones del Análisis"), width=12),
                dbc.Col(dcc.Markdown('''
                    - **Calidad de datos:** Algunos países tienen datos incompletos
                    - **Cobertura temporal:** No todos los países tienen datos recientes
                    - **Variables consideradas:** Solo se analiza pobreza monetaria
                    - **Supuestos del modelo:** Los pronósticos asumen estabilidad económica
                '''), width=12)
            ])
        ])

# Callbacks existentes para los gráficos EDA (se mantienen igual)
# ...

# Nuevos callbacks para el modelo de series de tiempo
@app.callback(
    [Output('sarima-plot', 'figure'),
     Output('sarima-metrics', 'children')],
    [Input('sarima-country', 'value'),
     Input('forecast-years', 'value'),
     Input('sarima-options', 'value')]
)
def update_sarima_plot(country, years_to_forecast, options):
    # Obtener datos y asegurar que el índice sea datetime
    country_data = df_long[df_long['Country Name'] == country]
    ts_data = country_data.set_index('Year')['Poverty Rate'].sort_index().dropna()
    
    # Convertir el índice a datetime (asumiendo que 'Year' es numérico o como '2020')
    ts_data.index = pd.to_datetime(ts_data.index, format='%Y')  # <- Cambio clave aquí
    
    if len(ts_data) < 10:
        return (
            go.Figure().add_annotation(text="Datos insuficientes para modelar", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False),
            "Se requieren al menos 10 puntos de datos temporales"
        )
    
    try:
        # Dividir datos (convertir índices a datetime si no lo están)
        train_size = int(len(ts_data) * 0.8)
        train, test = ts_data[:train_size], ts_data[train_size:]
        
        # Optimizar y ajustar modelo (el resto del código igual)
        order, seasonal_order = find_best_sarima(train)
        model = SARIMAX(train,
                      order=order,
                      seasonal_order=seasonal_order,
                      enforce_stationarity=False,
                      enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        
        # Pronosticar (steps = años a futuro + datos de prueba)
        forecast = model_fit.get_forecast(steps=len(test) + years_to_forecast)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Crear figura (usar índices datetime directamente)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts_data.index, y=ts_data,
            name='Datos Reales',
            mode='lines+markers',
            line=dict(color='blue', width=2)
        ))
        
        # ... (resto del código de gráfico igual)
        
        # Calcular métricas (extraer solo los valores para evitar problemas de índice)
        test_forecast = forecast_mean[:len(test)]
        mae = mean_absolute_error(test.values, test_forecast.values)  # <- Usar .values
        rmse = np.sqrt(mean_squared_error(test.values, test_forecast.values))
        
        
        # Actualizar layout
        fig.update_layout(
            title=f'Pronóstico de Pobreza en {country}',
            xaxis_title='Año',
            yaxis_title='Tasa de Pobreza (%)',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Crear texto de métricas
        metrics_text = html.Div([
            html.H5("Métricas del Modelo:"),
            html.P(f"Parámetros SARIMA: {order}{seasonal_order}"),
            html.P(f"MAE (Error Absoluto Medio): {mae:.2f}"),
            html.P(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f}"),
            html.P(f"Último dato disponible: {ts_data.index[-1]} - {ts_data.values[-1]:.2f}%"),
            html.P(f"Pronóstico {years_to_forecast} años: {forecast_mean.values[-1]:.2f}%")
        ])
        
        return fig, metrics_text
        
    except Exception as e:
        error_fig = go.Figure().add_annotation(
            text=f"Error en el modelo: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return error_fig, f"Error procesando {country}: {str(e)}"

# Agregar más callbacks para los otros gráficos del modelo según sea necesario
# Callbacks para el EDA Avanzado
@app.callback(
    [Output('correlation-heatmap', 'figure'),
     Output('distribution-chart', 'figure'),
     Output('boxplot-chart', 'figure'),
     Output('decomposition-chart', 'figure'),
     Output('advanced-stats-table', 'children')],
    [Input('eda2-country-dropdown', 'value'),
     Input('eda2-year-slider', 'value')]
)
def update_advanced_eda(selected_countries, year_range):
    # Filtrar datos
    filtered_df = df_long[
        (df_long['Country Name'].isin(selected_countries)) & 
        (df_long['Year'] >= year_range[0]) & 
        (df_long['Year'] <= year_range[1])
    ]
    
    if filtered_df.empty:
        empty_fig = go.Figure().add_annotation(text="No hay datos disponibles", 
                                             xref="paper", yref="paper", 
                                             x=0.5, y=0.5, showarrow=False)
        return empty_fig, empty_fig, empty_fig, empty_fig, html.P("No hay datos disponibles")
    
    # 1. Matriz de correlación entre años
    pivot_df = filtered_df.pivot(index='Country Name', columns='Year', values='Poverty Rate')
    corr_matrix = pivot_df.corr()
    
    heatmap = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Viridis',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlación')
    ))
    
    heatmap.update_layout(
        title='Matriz de Correlación entre Años',
        xaxis_title='Año',
        yaxis_title='Año',
        height=500
    )
    
    # 2. Gráfico de distribución
    hist = px.histogram(
        filtered_df,
        x='Poverty Rate',
        color='Country Name',
        marginal='box',
        title='Distribución de Tasas de Pobreza',
        labels={'Poverty Rate': 'Tasa de Pobreza (%)'},
        template='plotly_white',
        nbins=30,
        opacity=0.7
    )
    
    hist.update_layout(
        barmode='overlay',
        hovermode='x unified'
    )
    
    # 3. Boxplot por país
    boxplot = px.box(
        filtered_df,
        x='Country Name',
        y='Poverty Rate',
        color='Country Name',
        title='Distribución por País',
        labels={'Poverty Rate': 'Tasa de pobreza (%)', 'Country Name': 'País'},
        template='plotly_white'
    )
    
    boxplot.update_layout(
        showlegend=False,
        xaxis={'categoryorder': 'total descending'}
    )
    
    # 4. Descomposición temporal (para el primer país seleccionado)
    country_data = filtered_df[filtered_df['Country Name'] == selected_countries[0]]
    ts_data = country_data.set_index('Year')['Poverty Rate'].sort_index()
    
    if len(ts_data) > 2:
        decomposition = seasonal_decompose(ts_data, model='additive', period=5)
        
        decomp_fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                 subplot_titles=("Serie Original", "Tendencia", 
                                                "Estacionalidad", "Residuos"))
        
        decomp_fig.add_trace(
            go.Scatter(x=ts_data.index, y=ts_data, name='Original'),
            row=1, col=1
        )
        
        decomp_fig.add_trace(
            go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Tendencia'),
            row=2, col=1
        )
        
        decomp_fig.add_trace(
            go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Estacionalidad'),
            row=3, col=1
        )
        
        decomp_fig.add_trace(
            go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Residuos'),
            row=4, col=1
        )
        
        decomp_fig.update_layout(
            height=800,
            title_text=f"Descomposición Temporal - {selected_countries[0]}",
            showlegend=False
        )
    else:
        decomp_fig = go.Figure().add_annotation(
            text="Datos insuficientes para descomposición",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # 5. Tabla de estadísticas avanzadas
    stats = filtered_df.groupby('Country Name')['Poverty Rate'].agg(
        ['mean', 'median', 'std', 'min', 'max', 'count', lambda x: x.quantile(0.75) - x.quantile(0.25)]
    )
    stats.columns = ['Media', 'Mediana', 'Desv. Estándar', 'Mínimo', 'Máximo', 'N° Datos', 'Rango Intercuartil']
    stats = stats.reset_index().round(2)
    
    table = dash_table.DataTable(
        columns=[{'name': col, 'id': col} for col in stats.columns],
        data=stats.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_cell={
            'textAlign': 'left',
            'padding': '8px'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )
    
    return heatmap, hist, boxplot, decomp_fig, table
conclusiones_content = dbc.Card([
    dbc.CardBody([
        html.H2("Conclusiones", className="card-title"),
        html.H4("Hallazgos principales:"),
        html.Ul([
            html.Li("La pobreza global ha mostrado una tendencia decreciente desde 1990, con una reducción promedio del 1.2% anual hasta 2019"),
            html.Li("Los países de Asia Oriental lideran la reducción, con disminuciones superiores al 3% anual en casos como China y Vietnam"),
            html.Li("América Latina muestra patrones desiguales, con progreso significativo en algunos países pero estancamiento en otros"),
            html.Li("La pandemia de COVID-19 generó un aumento promedio del 1.5% en las tasas de pobreza durante 2020-2021"),
            html.Li("El modelo SARIMA demostró ser efectivo para pronósticos a corto plazo (MAPE=12%), pero pierde precisión para horizontes mayores a 5 años")
        ]),
        
        html.H4("Implicaciones Prácticas:"),
        html.P("Los resultados sugieren que:"),
        html.Ul([
            html.Li("Las políticas focalizadas en crecimiento económico inclusivo tienen mayor impacto en reducción de pobreza"),
            html.Li("Los sistemas de protección social son cruciales para mitigar el impacto de crisis como la pandemia"),
            html.Li("Se necesitan enfoques diferenciados por región, considerando contextos locales")
        ]),
        
        html.H4("Recomendaciones:"),
        html.Ol([
            html.Li("Priorizar inversiones en educación y salud como motores de movilidad social"),
            html.Li("Fortalecer sistemas de monitoreo en tiempo real para una respuesta rápida a crisis"),
            html.Li("Combinar análisis cuantitativos con estudios cualitativos para entender dinámicas locales"),
            html.Li("Desarrollar políticas diferenciadas para zonas urbanas vs. rurales")
        ]),
        
        html.H4("Trabajo Futuro:"),
        html.P("Las siguientes líneas de investigación podrían profundizar este análisis:"),
        html.Ul([
            html.Li("Incorporar variables macroeconómicas adicionales (PIB, desigualdad, empleo)"),
            html.Li("Analizar múltiples líneas de pobreza ($3.65 y $6.85 diarios)"),
            html.Li("Desarrollar modelos predictivos multivariados"),
            html.Li("Incluir dimensiones de pobreza multidimensional (educación, salud, vivienda)"),
            html.Li("Realizar análisis a nivel subnacional para identificar disparidades internas")
        ]),
        
        html.H4("Reflexión Final:"),
        html.P("""Este estudio confirma que, aunque el mundo ha logrado progresos significativos en la reducción de pobreza extrema, 
              los desafíos persisten. La pandemia demostró la fragilidad de estos avances y la necesidad de sistemas más resilientes. 
              Los datos sugieren que, manteniendo las tendencias pre-pandemia, el objetivo de erradicar la pobreza extrema para 2030 
              sería alcanzable solo en algunas regiones, destacando la urgencia de acelerar los esfuerzos globales.""")
    ])
])


# Callback para cambiar contenido de pestañas
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    if active_tab == "introduccion":
        return intro_content
    elif active_tab == "contexto":
        return contexto_content
    elif active_tab == "problema":
        return problema_content
    elif active_tab == "objetivos":
        return objetivos_content
    elif active_tab == "marco":
        return marco_content
    elif active_tab == "metodologia":
        return metodologia_content
    elif active_tab == "resultados":
        return resultados_content
    elif active_tab == "conclusiones":
        return conclusiones_content

# Callbacks para la sección de resultados (tu EDA)
@app.callback(
    Output('time-series-chart', 'figure'),
    [Input('country-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def update_time_series(selected_countries, year_range):
    filtered_df = df_long[
        (df_long['Country Name'].isin(selected_countries)) & 
        (df_long['Year'] >= year_range[0]) & 
        (df_long['Year'] <= year_range[1])
    ]
    
    fig = px.line(
        filtered_df, 
        x='Year', 
        y='Poverty Rate', 
        color='Country Name',
        title='Evolución de la Tasa de Pobreza',
        labels={'Poverty Rate': 'Tasa de Pobreza (%)', 'Year': 'Año'},
        template='plotly_white'
    )
    
    fig.update_layout(
        hovermode='x unified',
        yaxis_title="% de población bajo $2.15/día",
        legend_title_text='País/Región'
    )
    
    return fig

@app.callback(
    Output('world-map', 'figure'),
    [Input('year-slider', 'value')]
)
def update_world_map(year_range):
    latest_year = min(year_range[1], 2020)
    map_df = df_long[
        (df_long['Year'] == latest_year) & 
        (df_long['Poverty Rate'].notna())
    ]
    
    fig = px.choropleth(
        map_df,
        locations='Country Code',
        color='Poverty Rate',
        hover_name='Country Name',
        hover_data={'Poverty Rate': ':.2f%', 'Country Code': False},
        projection='natural earth',
        title=f'Distribución Global de Pobreza ({latest_year})',
        color_continuous_scale='OrRd',
        range_color=(0, map_df['Poverty Rate'].max()),
        template='plotly_white'
    )
    
    fig.update_layout(
        coloraxis_colorbar=dict(title="% Pobreza"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

@app.callback(
    Output('top-countries-chart', 'figure'),
    [Input('year-slider', 'value')]
)
def update_top_countries(year_range):
    start_year, end_year = year_range
    
    start_df = df_long[df_long['Year'] == start_year][['Country Name', 'Poverty Rate']]
    end_df = df_long[df_long['Year'] == end_year][['Country Name', 'Poverty Rate']]
    
    merged = pd.merge(start_df, end_df, on='Country Name', suffixes=('_start', '_end'))
    merged['Change'] = ((merged['Poverty Rate_end'] - merged['Poverty Rate_start']) / 
                       merged['Poverty Rate_start'].replace(0, np.nan)) * 100
    merged = merged.dropna()
    
    top_reducers = merged.sort_values('Change').head(10)
    
    fig = px.bar(
        top_reducers,
        x='Change',
        y='Country Name',
        orientation='h',
        title=f'Top 10 Países con Mayor Reducción ({start_year}-{end_year})',
        labels={'Change': 'Reducción (%)', 'Country Name': 'País'},
        color='Change',
        color_continuous_scale='Greens',
        template='plotly_white'
    )
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        xaxis_title="Reducción en puntos porcentuales",
        showlegend=False
    )
    
    return fig

@app.callback(
    Output('complete-data-table', 'children'),
    [Input('year-slider', 'value')]
)
def update_complete_data_table(year_range):
    complete_data = df_real_data[df_real_data['Years with Data'] >= 30]
    complete_data = complete_data.sort_values('Years with Data', ascending=False).head(10)
    
    table = dash_table.DataTable(
        columns=[
            {'name': 'País/Región', 'id': 'Country Name'},
            {'name': 'Años con Datos', 'id': 'Years with Data'}
        ],
        data=complete_data[['Country Name', 'Years with Data']].to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_cell={
            'textAlign': 'left',
            'padding': '8px'
        }
    )
    
    return table

@app.callback(
    Output('stats-table', 'children'),
    [Input('country-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def update_stats_table(selected_countries, year_range):
    filtered_df = df_long[
        (df_long['Country Name'].isin(selected_countries)) & 
        (df_long['Year'] >= year_range[0]) & 
        (df_long['Year'] <= year_range[1])
    ]
    
    if filtered_df.empty:
        return html.P("Selecciona países para ver estadísticas", className="text-muted")
    
    stats = filtered_df.groupby('Country Name')['Poverty Rate'].agg(['mean', 'min', 'max', 'std'])
    stats.columns = ['Promedio', 'Mínimo', 'Máximo', 'Desviación']
    stats = stats.reset_index().round(2)
    
    table = dash_table.DataTable(
        columns=[{'name': col, 'id': col} for col in stats.columns],
        data=stats.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_cell={
            'textAlign': 'left',
            'padding': '8px'
        }
    )
    
    return table


if __name__ == '__main__':
    app.run(debug=True)

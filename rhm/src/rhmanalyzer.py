import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats.distributions import chi2
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from math import pi
from datetime import date

## Bokeh imports

from bokeh.plotting import figure, output_notebook, show, ColumnDataSource
from bokeh.io import push_notebook
from bokeh.layouts import gridplot
from bokeh.models import Range1d, LinearAxis, CustomJS, DateRangeSlider
from bokeh.embed import components
from bokeh.models import HoverTool
from bokeh.models.glyphs import Text
from bokeh.layouts import column
from bokeh.tile_providers import get_provider, WIKIMEDIA, CARTODBPOSITRON, STAMEN_TERRAIN, STAMEN_TONER, ESRI_IMAGERY, OSM
from bokeh.models import DatetimeTickFormatter
from pyproj import Proj, transform
from bokeh.layouts import row 
from bokeh.plotting import figure
from bokeh.transform import cumsum
from bokeh.models.glyphs import Text

import warnings

warnings.filterwarnings("ignore")
pd.set_option("max_columns", 30)

DataPath = "C:/Users/agh/OneDrive - Spartan Controls Ltd/Documents/Projects/Project1-Regulator Health Monitoring/RHMDashboard/rhm/static/rhm/data/"

def processData():
        # Loading data
        Flow = pd.read_csv(DataPath+'Flow.csv')
        OP   = pd.read_csv(DataPath+'OP.csv')
        IP   = pd.read_csv(DataPath+'IP.csv')

        # Preprocessing data

        # Step 1: Renaming and processing columns
        Flow = Flow[['Time', 'Value', 'Average', 'Minimum', 'Maximum']]
        Flow['Time'] = pd.to_datetime(Flow['Time'])
        Flow['Value'] = 41.67*Flow['Value']
        Flow.rename(columns={'Value': 'F (m3/h)', 'Average': 'F_Avg', 'Minimum': 'F_Min', 'Maximum': 'F_Max'}, inplace=True)

        OP = OP[['Time', 'Value', 'Average', 'Minimum', 'Maximum']]
        OP['Time'] = pd.to_datetime(OP['Time'])
        OP.rename(columns={'Value': 'OP (kPa)', 'Average': 'OP_Avg', 'Minimum': 'OP_Min', 'Maximum': 'OP_Max'}, inplace=True)

        IP = IP[['Time', 'Value', 'Average', 'Minimum', 'Maximum']]
        IP['Time'] = pd.to_datetime(IP['Time'])
        IP.rename(columns={'Value': 'IP (kPa)', 'Average': 'IP_Avg', 'Minimum': 'IP_Min', 'Maximum': 'IP_Max'}, inplace=True)

        # Step 2: Merge flow, op, ip data
        # Preparing 2018 - 2020 data, include only outlet pressure & inlet pressure. No flow data available
        # Prparing 2020 data, include outlet pressure, inlet pressure, and flow

        Data_All = IP.merge(OP, left_on='Time', right_on='Time')
        Data_2020 = Flow.merge(IP, left_on='Time', right_on='Time')
        Data_2020 = Data_2020.merge(OP, left_on='Time', right_on='Time')

        print(Data_All.head())
        print(Data_2020.head())

        plot1 = make_trendplots_2020(Data_2020)
        plot2 = make_trendplots_all(Data_All)
        plot3 = make_scatterplots(Data_2020)

        Data_2020_Updated = calculate_healthscore(Data_2020)
        ## Clean final dataset

        Data_2020_Updated.loc[Data_2020_Updated['IP (kPa)'] < 3000, 'T2 Score'] = 0
        Data_2020_Updated.loc[Data_2020_Updated['F (m3/h)'] < 25, 'T2 Score'] = 0
        Data_2020_Updated.loc[Data_2020_Updated['IP (kPa)'] < 3000, 'Anomaly Score (SPE) %'] = 0
        Data_2020_Updated.loc[Data_2020_Updated['F (m3/h)'] < 25, 'Anomaly Score (SPE) %'] = 0

        plot4 = make_anomaly_plots(Data_2020_Updated)
        
        plot5 = make_combination_plots(Data_2020)

        plot6 =  make_multi_yaxis_plot(Data_2020_Updated)
        
        return plot1, plot2, plot3, plot4, plot5, plot6

def make_trendplots_2020(Data_2020):
        Tooltips1=[("Index", "$index"),
          ("Time:", "$x"),
          ("Flow: ", "$y")]

        plot1 = figure(title='Flow Rate (m3/h)', y_axis_label = "F (m3/h)", plot_width=330, plot_height=320, tooltips=Tooltips1, sizing_mode='scale_width')
        plot1.line(Data_2020['Time'], Data_2020['F (m3/h)'], line_alpha=0.8, line_width=2, color = "#118DFF")
        plot1.xaxis.formatter=DatetimeTickFormatter(
        hours=["%d %b %Y"],
        days=["%d %b %Y"],
        months=["%d %b %Y"],
        years=["%d %b %Y"],)
        plot1.xgrid.visible = False
        plot1.ygrid.visible = False
        plot1.xaxis.minor_tick_line_color = None
        plot1.yaxis.minor_tick_line_color = None
        plot1.outline_line_color = None
        plot1.xaxis.major_label_orientation = math.pi/4

        Tooltips2=[("Index", "$index"),
          ("Time:", "$x"),
          ("Flow: ", "$y")]

        plot2 = figure(title='Inlet Pressure (kPa)', x_range=plot1.x_range, y_axis_label = "IP (kPa)", plot_width=330, plot_height=320, tooltips=Tooltips2, sizing_mode='scale_width')
        plot2.line(Data_2020['Time'], Data_2020['IP (kPa)'], line_alpha=0.8, line_width=2, color = "#118DFF")
        plot2.xaxis.formatter=DatetimeTickFormatter(
        hours=["%d %b %Y"],
        days=["%d %b %Y"],
        months=["%d %b %Y"],
        years=["%d %b %Y"],)
        plot2.xgrid.visible = False
        plot2.ygrid.visible = False
        plot2.xaxis.minor_tick_line_color = None
        plot2.yaxis.minor_tick_line_color = None
        plot2.outline_line_color = None
        plot2.xaxis.major_label_orientation = math.pi/4

        Tooltips3=[("Index", "$index"),
          ("Time:", "$x"),
          ("Flow: ", "$y")]

        plot3 = figure(title='Outlet Pressure (kPa)', x_range=plot1.x_range,  y_axis_label = "OP (kPa)", plot_width=330, plot_height=320, tooltips=Tooltips3, sizing_mode='scale_width')
        plot3.line(Data_2020['Time'], Data_2020['OP (kPa)'], line_alpha=0.8, line_width=2, color="#118DFF")
        plot3.background_fill_color = None
        plot3.xaxis.formatter=DatetimeTickFormatter(
        hours=["%d %b %Y"],
        days=["%d %b %Y"],
        months=["%d %b %Y"],
        years=["%d %b %Y"],)
        plot3.xgrid.visible = False
        plot3.ygrid.visible = False
        plot3.xaxis.minor_tick_line_color = None
        plot3.yaxis.minor_tick_line_color = None
        plot3.outline_line_color = None
        plot3.xaxis.major_label_orientation = math.pi/4

        p = gridplot([[plot1, plot2, plot3]], toolbar_location='right', sizing_mode='stretch_both')


        callback1 = CustomJS(args=dict(plot=plot1), code="""
        var a = cb_obj.value;
        plot.x_range.start = a[0];
        plot.x_range.end = a[1];
        """)

        slider = DateRangeSlider(start=date(2020, 1, 1), end=date(2020, 9, 30), value=(date(2020, 1, 1), date(2020, 9, 30)), step=30, format="%d, %b, %Y")
        slider.js_on_change('value', callback1)


        layout = column(p,slider)

        return layout

def make_anomaly_plots(Data_2020_Updated):

        ## Generate plot
        Tooltips1=[("Index", "$index"),
          ("Time:", "$x"),
          ("Flow: ", "$y")]

        plot = figure(title='Anomaly trend for 2020', x_axis_label = "Time", y_axis_label = "Anomaly Score (SPE) %", plot_width=500, plot_height=220, tooltips=Tooltips1, toolbar_location="below")
        plot.line(Data_2020_Updated['Time'], Data_2020_Updated['Anomaly Score (SPE) %'], line_alpha=0.8, legend_label='Flow trend per minute', line_width=2, color = "orange")

        return plot

def make_trendplots_all(Data_All):
        Tooltips1=[("Index", "$index"),
          ("Time:", "$x"),
          ("Flow: ", "$y")]

        plot1 = figure(title='RHM all inlet pressure trend', x_axis_label = "Time", y_axis_label = "IP (kPa)", plot_width=450, plot_height=375, tooltips=Tooltips1, toolbar_location="below")
        plot1.line(Data_All['Time'], Data_All['IP (kPa)'], line_alpha=0.8, legend_label='Inlet pressure trend', line_width=2, color = "blue")

        Tooltips2=[("Index", "$index"),
          ("Time:", "$x"),
          ("Flow: ", "$y")]

        plot2 = figure(title='RHM all outlet pressure trend', x_axis_label = "Time", y_axis_label = "OP (kPa)", plot_width=450, plot_height=375, tooltips=Tooltips2, toolbar_location="below")
        plot2.line(Data_All['Time'], Data_All['OP (kPa)'], line_alpha=0.8, legend_label='Outlet pressure trend', line_width=2, color = "green")

        plots = column(plot1, plot2)

        return plots

def make_scatterplots(Data_2020):
        # Preprocessing

        Data_2020['Month'] = Data_2020['Time'].dt.month
        Data_2020 = Data_2020[Data_2020['IP (kPa)'] >= 3000] # Removing line shutdown data
        Data_2020 = Data_2020[Data_2020['F (m3/h)'] >= 25] # Removing line shutdown data
        Data_2020 = Data_2020[Data_2020['Month'] < 10] # Retaining data till regulator failure

        # Drawing the charts

        colormap = {1: 'blue', 2: 'orange', 3: 'green', 4: 'red', 5: 'purple', 6: 'brown', 9: 'pink'}
        colors = [colormap[x] for x in Data_2020['Month']]
        labels = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 9: 'September'}
        annotations = [labels[z] for z in Data_2020['Month'].ravel()]

        Tooltips=[("Index", "$index"),
        ("Time:", "$x"),
        ("Flow: ", "$y")]

        source1 = ColumnDataSource(
                data = dict(
                x= Data_2020['IP (kPa)'],
                y = Data_2020['OP (kPa)'],
                label = annotations,
                all_colors = colors
                )
        )

        plot1 = figure(title='Inlet vs Outlet pressure data', x_axis_label = "IP (kPa)", y_axis_label = "OP (kPa)", plot_width=600, plot_height=450, tooltips=Tooltips)
        plot1.circle('x', 'y', fill_color='all_colors', fill_alpha=0.8, line_color='white', legend='label', size=7, source=source1)
        plot1.legend.location = 'left'
        plot1.background_fill_color = None

        source2 = ColumnDataSource(
                data = dict(
                x= Data_2020['F (m3/h)'],
                y = Data_2020['OP (kPa)'],
                label = annotations,
                all_colors = colors
                )
        )

        plot2 = figure(title='Flow vs Outlet pressure data', x_axis_label = "F (m3/h)", y_axis_label = "OP (kPa)", plot_width=600, plot_height=250, tooltips=Tooltips)
        plot2.circle('x', 'y', fill_color='all_colors', line_color='white', size=5, legend='label', source=source2)

        source3 = ColumnDataSource(
                data = dict(
                x= Data_2020['F (m3/h)'],
                y = Data_2020['IP (kPa)'],
                label = annotations,
                all_colors = colors
                )
        )

        plot3 = figure(title='Flow vs Inlet pressure data', x_axis_label = "F (m3/h)", y_axis_label = "IP (kPa)", plot_width=600, plot_height=250, tooltips=Tooltips)
        plot3.circle('x', 'y', fill_color='all_colors', line_color='white', size=5, legend='label', source=source3)

        #plots = column(plot1, plot2, plot3)

        return plot1

def calculate_healthscore(Data_2020):
        # Building PCA model with 3 months of data for outlier detection in the rest of the data
        Data_2020_Original = Data_2020.copy()
        Data = Data_2020[['Time', 'Month', 'F (m3/h)', 'IP (kPa)', 'OP (kPa)']]
        Data_Training = Data[Data['Month'] <= 3]
        Data_Testing = Data_2020_Original

        Data_Training = Data_Training[['F (m3/h)', 'IP (kPa)', 'OP (kPa)']].to_numpy()
        Data_Testing = Data_Testing[['F (m3/h)', 'IP (kPa)', 'OP (kPa)']].to_numpy()

        Data_Training = Data_Training[~np.isnan(Data_Training.sum(axis=1)), :]
        Data_mean = np.nanmean(Data_Training, axis=0)


        Data_Training = Data_Training - Data_mean
        Data_Testing = Data_Testing - Data_mean


        covariance = np.cov(Data_Training.transpose())

        u, s, vh = np.linalg.svd(covariance, full_matrices=True)

        # SPE statistics
        r=2
        data = Data_Testing
        normalizer = np.eye(vh.shape[1]) - np.dot(vh[:r, :].transpose(), vh[:r, :])
        data_n = np.dot(data, normalizer)
        data_square = data * data_n
        alert_score = np.sum(data_square, axis=1)

        t1 = sum(s[r:])
        t2 = sum(s[r:] ** 2)
        t3 = sum(s[r:] ** 3)
        h0 = 1 - (2 * t1 * t3 / (3 * (t2 ** 2)))
        c = 3
        Q_alpha = t1 * ((c * np.sqrt(2 * t2 * (h0 ** 2)) / t1) + 1 + ((t2 * h0 * (h0 - 1)) / (t1 ** 2))) ** (1 / h0)

        alert_score = alert_score - Q_alpha
        alert_score[alert_score < 0] = 0
        alert_score = alert_score * 100 / Q_alpha
        alert_score[alert_score > 100] = 100

        Data_2020_Original['Anomaly Score (SPE) %'] = alert_score

        # T2 statistics
        normalizer = np.diag(1 / s[:r])
        data_n = np.dot(data, vh[:r, :].transpose())
        data_norm = np.dot(data_n, normalizer)
        data_square = data_n * data_norm

        alert_score = np.sum(data_square, axis=1)

        T_alpha = chi2.ppf(0.999, df=r)
        alert_score = alert_score - T_alpha
        alert_score[alert_score < 0] = 0
        alert_score = alert_score * 100 / T_alpha
        alert_score[alert_score > 100] = 100

        Data_2020_Original['T2 Score'] = alert_score

        return Data_2020_Original

def map_geo_locations():

        colors = ["Green", "Green", "Green", "Green", "Red", "Green", "Green", "Green", "Red", "Green","Green", "Green", "Green", "Green", "Red", "Green", "Green", "Green", "Red", "Green"]

        TOOLS = "box_select,lasso_select,box_zoom,wheel_zoom, save,reset,hover,help"

        Locn = pd.read_csv(DataPath+'Locations.csv')

        inProj = Proj(init='epsg:3857')
        outProj = Proj(init='epsg:4326')

        world_lon1, world_lat1 = transform(outProj,inProj,-150,60)
        world_lon2, world_lat2 = transform(outProj,inProj,-70,55)

        cartodb = get_provider(CARTODBPOSITRON)

        rhm_ca = Locn[Locn.Country == "CA"]

        lons, lats = [], []
        for lon, lat in list(zip(rhm_ca["Longitude"], rhm_ca["Latitude"])):
                x, y = transform(outProj,inProj,lon,lat)
                lons.append(x)
                lats.append(y)
        
        rhm_ca["MercatorX"] = lons
        rhm_ca["MercatorY"] = lats

        rhm_ca = rhm_ca.rename(columns={"Site Name":"Name", "State/Province":"State"})

        Tooltips=[("Index", "Regulator "+"$index"),
                        ("State:", "@state"),
                        ("Londitude: ", "@londt"),
                        ("Latitude: ", "@lattd")]

        plot = figure(plot_width=1170, plot_height=755,
           x_range=(world_lon1, world_lon2), y_range=(world_lat1, world_lat2),
           x_axis_type="mercator", y_axis_type="mercator",
        #    tooltips=[
        #             ("State", 'rhm_ca["State"]'), ("(Long, Lat)", "(@Longitude, @Latitude)")
        #             ],
           tooltips=Tooltips,
           tools=TOOLS)

        plot.add_tile(cartodb)
        
        plot.axis.visible = False
        src = ColumnDataSource(
                data=dict(
                        x= rhm_ca["MercatorX"],
                        y= rhm_ca["MercatorY"],
                        state = rhm_ca["State"],
                        londt= rhm_ca["Longitude"],
                        lattd= rhm_ca["Latitude"],
                        color = colors
                )
        )

        plot.circle(x="x", y="y",
                size=15,
                fill_color="color", line_color="grey",
                fill_alpha=0.5,
                source=src,
                )

        return plot

def make_capacity_plot():
        layout = go.Layout(
                margin=go.layout.Margin(
                        l=20, #left margin
                        r=50, #right margin
                        b=0, #bottom margin
                        t=0, #top margin
                )
                )

        fig = go.Figure(go.Indicator(
                domain = {'x': [0, 1], 'y': [0, 1]},
                value = 3575,
                mode = "gauge+number+delta",
                title = {'text': "Capacity"},
                delta = {'reference': 5000},
                gauge = {'axis': {'range': [None, 5000]},
                        'steps' : [
                                {'range': [0,4500], 'color': "lightgreen"},
                                ],
                        'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 4900}}),
                layout=layout)

        return fig

def make_healthscore_plot():

        layout = go.Layout(
                margin=go.layout.Margin(
                        l=0, #left margin
                        r=0, #right margin
                        b=0, #bottom margin
                        t=0, #top margin
                )
                )

        data = pd.DataFrame(['Col', "Val"])
        data['Col'] = ['Healthy','Unhealthy']
        data['Val'] = [15.8, 84.2]
        data['Angle'] = data['Val']/data['Val'].sum() * 2*pi
        data['color'] = ["olivedrab", "lightgrey"]
        txt = ['Health Score \n     84.2%', 'Health Score \n     84.2%']

        src = ColumnDataSource(
        data = dict(
                        color = data['color'],
                        angle=data['Angle'],
                        text=txt
                        )
        )
        p = figure(plot_height=370, x_range=(-.35, .55), background_fill_color = None)

        glyph = Text(x=0,y=1, x_offset=-65, y_offset=35, text="text", text_font_size="18pt", text_color="#707070")

        p.annular_wedge(x=0, y=1, inner_radius=0.15, outer_radius=0.25, direction="clock",
                        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                line_color="white", fill_color='color', source=src)

        p.add_glyph(src, glyph)
        p.outline_line_color = None
        p.axis.axis_label=None
        p.axis.visible=False
        p.grid.grid_line_color = None
        p.background_fill_color = None

        return p

def make_combination_plots(Data_2020):
        Tooltips=[("Index", "$index"),
          ("Time", "$x"),
          ("Flow: ", "$y")]
        #)

        plot1 = figure(title='Flow Vs. Outlet Pressure', x_axis_label = "Time", y_axis_label = "F (m3/h)", plot_width=750, plot_height=350, tooltips=Tooltips)
        plot1.extra_y_ranges = {"foo": Range1d(start=Data_2020['OP (kPa)'].min(), end=Data_2020['OP (kPa)'].max())}
        plot1.add_layout(LinearAxis(y_range_name="foo", axis_label="OP (kPa)"), 'right')
        plot1.line(Data_2020['Time'], Data_2020['F (m3/h)'], line_alpha=0.8, legend_label='Flow', line_width=2, color='orange')
        plot1.line(Data_2020['Time'], Data_2020['OP (kPa)'], line_alpha=0.8, legend_label='Outlet Pressure', line_width=2, color='blue', y_range_name="foo")
        plot1.xaxis.formatter=DatetimeTickFormatter(
        hours=["%d %b %Y"],
        days=["%d %b %Y"],
        months=["%d %b %Y"],
        years=["%d %b %Y"],
    )

        plot2 = figure(title='Inlet Vs. Outlet Pressure', x_axis_label = "Time", y_axis_label = "IP (kPa)", plot_width=700, plot_height=400, tooltips=Tooltips)
        plot2.extra_y_ranges = {"foo": Range1d(start=Data_2020['OP (kPa)'].min(), end=Data_2020['OP (kPa)'].max())}
        plot2.add_layout(LinearAxis(y_range_name="foo", axis_label="OP (kPa)"), 'right')
        plot2.line(Data_2020['Time'], Data_2020['IP (kPa)'], line_alpha=0.8, legend_label='Inlet Pressure', line_width=2, color='orange')
        plot2.line(Data_2020['Time'], Data_2020['OP (kPa)'], line_alpha=0.8, legend_label='Outlet Pressure', line_width=2, color='blue', y_range_name="foo")
        plot2.xaxis.formatter=DatetimeTickFormatter(
        hours=["%d %b %Y"],
        days=["%d %b %Y"],
        months=["%d %b %Y"],
        years=["%d %b %Y"],
    )

        plot3 = figure(title='Flow Vs. Inlet Pressure', x_axis_label = "Time", y_axis_label = "F (m3/h)", plot_width=700, plot_height=400, tooltips=Tooltips)
        plot3.extra_y_ranges = {"foo": Range1d(start=Data_2020['IP (kPa)'].min(), end=Data_2020['IP (kPa)'].max())}
        plot3.add_layout(LinearAxis(y_range_name="foo", axis_label="IP (kPa)"), 'right')
        plot3.line(Data_2020['Time'], Data_2020['F (m3/h)'], line_alpha=0.8, legend_label='Flow', line_width=2, color='orange')
        plot3.line(Data_2020['Time'], Data_2020['IP (kPa)'], line_alpha=0.8, legend_label='Inlet Pressure', line_width=2, color='blue', y_range_name="foo")
        plot3.xaxis.formatter=DatetimeTickFormatter(
        hours=["%d %b %Y"],
        days=["%d %b %Y"],
        months=["%d %b %Y"],
        years=["%d %b %Y"],
    )

        return plot1

def make_multi_yaxis_plot(Data_2020_Original):
        Tooltips=[("Index", "$index"),
          ("Time", "$x"),
          ("Flow: ", "$y")]


        plot = figure(x_axis_label = "Time", y_axis_label = "F (m3/h)", plot_width=880, plot_height=400, tooltips=Tooltips, sizing_mode='scale_width')

        plot.yaxis.axis_line_color = 'orange'
        plot.yaxis.axis_label_text_color = 'orange'
        plot.yaxis.major_label_text_color = 'orange'
        plot.yaxis.major_tick_line_color= 'orange'

        # Setting the second y axis range name and range
        plot.extra_y_ranges = {"foo": Range1d(start=Data_2020_Original['OP (kPa)'].min(), end=Data_2020_Original['OP (kPa)'].max()),
                        "bar": Range1d(start=Data_2020_Original['IP (kPa)'].min(), end=Data_2020_Original['IP (kPa)'].max()),
                        "cat": Range1d(start=Data_2020_Original['Anomaly Score (SPE) %'].min(), end=Data_2020_Original['Anomaly Score (SPE) %'].max())}

        # Adding the second axis to the plot.  
        plot.add_layout(LinearAxis(y_range_name="foo", 
                                axis_label="OP (kPa)", 
                                axis_line_color = 'blue', 
                                axis_label_text_color='blue',
                                major_label_text_color = 'blue',
                                major_tick_line_color= 'blue'), 'left')

        # Adding the second axis to the plot.  
        plot.add_layout(LinearAxis(y_range_name="bar", 
                                axis_label="IP (kPa)", 
                                axis_line_color = 'green', 
                                axis_label_text_color='green',
                                major_label_text_color = 'green',
                                major_tick_line_color= 'green'), 'left')

        # Adding the second axis to the plot.  
        plot.add_layout(LinearAxis(y_range_name="cat", 
                                axis_label="Anomaly (kPa)", 
                                axis_line_color = 'red', 
                                axis_label_text_color='red',
                                major_label_text_color = 'red',
                                major_tick_line_color= 'red'), 'left')

        plot.line(Data_2020_Original['Time'], Data_2020_Original['F (m3/h)'], line_alpha=0.6, legend_label='Flow', line_width=2, color='orange')

        plot.line(Data_2020_Original['Time'], Data_2020_Original['OP (kPa)'], line_alpha=0.6, legend_label='Outlet Pressure', line_width=2, color='blue', y_range_name="foo")

        plot.line(Data_2020_Original['Time'], Data_2020_Original['IP (kPa)'], line_alpha=0.6, legend_label='Inlet Pressure', line_width=2, color='green', y_range_name="bar")

        plot.line(Data_2020_Original['Time'], Data_2020_Original['Anomaly Score (SPE) %'], line_alpha=0.8, legend_label='Anomaly', line_width=2, color='red', y_range_name="cat")

        plot.add_layout(plot.legend[0], 'right')

        plot.xaxis.formatter=DatetimeTickFormatter(
                hours=["%d %b %Y"],
                days=["%d %b %Y"],
                months=["%d %b %Y"],
                years=["%d %b %Y"],)

        callback1 = CustomJS(args=dict(plot=plot), code="""
        var a = cb_obj.value;
        plot.x_range.start = a[0];
        plot.x_range.end = a[1];
        """)

        slider = DateRangeSlider(start=date(2020, 1, 1), end=date(2020, 9, 30), value=(date(2020, 1, 1), date(2020, 9, 30)), step=30, format="%d, %b, %Y")
        slider.js_on_change('value', callback1)

        layout = column(plot,slider)

        return layout


# class RhmAnalyzer:

#     def processData():
#         Flow = pd.read_csv('./static/data/Flow.csv', nrows=5)
#         # OP   = pd.read_csv('./data/OP.csv')
#         # IP   = pd.read_csv('./data/IP.csv')

#         Flow = Flow[['Time', 'Value', 'Average', 'Minimum', 'Maximum']]
#         print(Flow.head())
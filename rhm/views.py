## Django imports
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt

## bokeh imports

from bokeh.embed import components

## My file imports

from .src.rhmanalyzer import processData, map_geo_locations, make_healthscore_plot, make_capacity_plot

# Create your views here.
@csrf_exempt
def home(request):

    '''
    An alternative home page
    '''

    plot = map_geo_locations()
    script_map, div_map = components((plot))

    return render(request, 'rhm/locations.html', 
                            {'script_map': script_map, 
                            'div_map' : div_map})

@csrf_exempt
def get_details(request):
    plot1, plot2, plot3, plot4, plot7, plot8 = processData()
    plot5= make_healthscore_plot()

    script_tp20, div_tp20 = components((plot1))
    script_tpa, div_tpa = components((plot2))
    script_scp, div_scp = components((plot3))
    script_an, div_an = components((plot4))
    script_hs, div_hs = components((plot5))
    script_cmb, div_cmb = components((plot7))
    script_mlty, div_mlty = components((plot8))

    plot6= make_capacity_plot()
    graph = plot6.to_html(full_html=False, default_height=368, default_width=450)
    

    return render(request, 'rhm/dashboard.html',{
                        'script_tp20': script_tp20, 
                        'div_tp20' : div_tp20, 
                        'script_tpa' : script_tpa, 
                        'div_tpa' : div_tpa, 
                        'script_scp' : script_scp, 
                        'div_scp' : div_scp , 
                        'script_an' : script_an, 
                        'div_an' : div_an,
                        'script_hs': script_hs, 
                        'div_hs': div_hs,
                        'graph': graph,
                        'script_cmb' : script_cmb, 
                        'div_cmb' : div_cmb,
                        'script_mlty' : script_mlty, 
                        'div_mlty' : div_mlty})

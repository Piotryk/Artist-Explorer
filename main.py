import os
import spotipy
import webbrowser
import numpy as np
import PySimpleGUI as sg
import contextlib
with contextlib.redirect_stdout(None):
    from pygame import mixer

#from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_CLIENT_URL
import artist_graph as artist_graph_lib
import artist_graph_functions
import window

#TODO:
'''
Check if temp folder exists before downloading icons and music!!!!!! 

expand more?
Month listeners - Spotify API does not provide that info
'''

main_window = window.make_window(debug=True)
graph = main_window['GRAPH']

SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_CLIENT_URL = artist_graph_functions.get_credentials('config.py')
SCOPE = 'playlist-modify-public playlist-modify-private user-library-read user-library-modify'
auth_manager = spotipy.oauth2.SpotifyOAuth(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET, redirect_uri=SPOTIFY_CLIENT_URL, scope=SCOPE, open_browser=True)
spotify = spotipy.Spotify(auth_manager=auth_manager)

artists_graph = artist_graph_lib.Graph()
mixer.init()
music = None

half_size = np.array(window.size_graph) / 2

previous_event = previous_event_value = None
dragging = node_flag = False
start_point = end_point = None
x = y = prev_x = prev_y = 0
no_children = 9
empty_state = True
search_state = False
top_songs = []


while True:
    event, values = main_window.read()

    if event == sg.WIN_CLOSED:
        main_window.close()
        break

    # killing double GRAPH event created by enable_events and drag_submits
    if event == 'GRAPH' and previous_event == 'GRAPH' and values['GRAPH'] == previous_event_value:
        continue

    if event == 'MouseWheel:Up':
        artist_graph_functions.zoom_in(main_window, graph, artists_graph, half_size)

    if event == 'MouseWheel:Down':
        artist_graph_functions.zoom_out(main_window, graph, artists_graph, half_size)

    if event == "GRAPH":
        x, y = values["GRAPH"]
        dragging, start_point, end_point, prev_x, prev_y = artist_graph_functions.drag(graph, artists_graph, x, y, dragging, start_point, end_point, prev_x, prev_y)

    if event.endswith('+UP'):  # is elif needed ?
        clicked_figs = graph.get_figures_at_location((x, y))
        dragging = False

        if end_point:   # if there is no end point that means it was just click
            start_point = end_point = None
            continue
        start_point = end_point = None

        if empty_state or not clicked_figs:
            continue

        if search_state:
            artists_graph = artist_graph_functions.artists_graph_from_search_graph(
                main_window, graph, artists_graph, clicked_figs)
            search_state = False
            clicked_figs = [artists_graph.start_node.frame_id]

        top_songs = artist_graph_functions.on_click(main_window, graph, artists_graph, spotify, clicked_figs, values['SLIDER_CHILD_NO'])

    if event in ['SEARCH', 'SEARCH_ENTER']:
        artist_graph_functions.clear_artist_info(main_window)
        artist_graph_functions.clear_top_tracks(main_window)
        artists_graph, empty_state, search_state = artist_graph_functions.search(main_window, graph, artists_graph, spotify, values)

    if 'SONG_PLAY_' in event:
        music = artist_graph_functions.play_preview(main_window, mixer, event, values)

    if event == 'VOLUME' and music:
        music.set_volume(values['VOLUME'] / 100)

    if event == 'STOP':
        mixer.stop()
        window.debug(main_window, 'Ready')
        music = None

    if event == 'SLIDER_CHILD_NO':
        no_children = int(values['SLIDER_CHILD_NO'])

    if event == 'CLEAR':
        artist_graph_functions.clear_artist_info(main_window)
        artist_graph_functions.clear_top_tracks(main_window)
        artists_graph = artist_graph_lib.Graph()
        graph.erase()
        mixer.stop()
        empty_state = True
        music = None

    if "SONG_ADD" in event:
        artist_graph_functions.add_song_to_playlist(main_window, spotify, event, top_songs)

    if event == "INFO_URL":
        webbrowser.open(main_window['INFO_URL'].get_text())

    #if event == 'Left:37':
    if 'Left' in event:
        no_children = int(values['SLIDER_CHILD_NO'])
        if no_children >= 2:
            main_window['SLIDER_CHILD_NO'].update(no_children - 1)

    #if event == 'Right:39':
    if 'Right' in event:
        no_children = int(values['SLIDER_CHILD_NO'])
        if no_children <= 18:
            main_window['SLIDER_CHILD_NO'].update(no_children + 1)

    main_window.refresh()
    previous_event = event
    previous_event_value = values['GRAPH']

main_window.close()

# Clear temp files
for temp_file in os.listdir('assets/temp'):
    os.remove(os.path.join('assets/temp', temp_file))

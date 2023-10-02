import base64
import PySimpleGUI as sg
from PIL import Image
from io import BytesIO


def base64_image_import(path):
    image = Image.open(path)
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue())
    return b64


def debug(main_window, text):
    main_window['DEBUG'].update(text)
    main_window.refresh()


sg.theme('DarkAmber')
graph_background_color = '#dddddd'
ico_path = 'assets/spotify_ico.png'

album_ico_size = (32, 32)
song_name_size = (27, 1)

size_graph = (1080, 890)
size_control = (442, 890)
default_win_size = (1520, 895)
default_win_size_debug = (1520, 870)


top_songs_frame = [
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY_1', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_1', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_1', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Add to playlist', key='SONG_ADD_1', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY_2', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_2', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_2', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Add to playlist', key='SONG_ADD_2', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY_3', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_3', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_3', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Add to playlist', key='SONG_ADD_3', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY_4', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_4', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_4', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Add to playlist', key='SONG_ADD_4', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY_5', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_5', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_5', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Add to playlist', key='SONG_ADD_5', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY_6', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_6', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_6', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Add to playlist', key='SONG_ADD_6', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY_7', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_7', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_7', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Add to playlist', key='SONG_ADD_7', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY_8', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_8', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_8', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Add to playlist', key='SONG_ADD_8', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY_9', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_9', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_9', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Add to playlist', key='SONG_ADD_9', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY_10', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_10', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_10', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Add to playlist', key='SONG_ADD_10', visible=False)],
]

graph = sg.Graph(canvas_size=size_graph, graph_bottom_left=(0, 0), graph_top_right=size_graph,
                 background_color=graph_background_color,
                 enable_events=True, drag_submits=True, expand_x=True, expand_y=True, key="GRAPH")

control_col = sg.Column(layout=[
    [sg.Push(), sg.Image('assets/spotify_ico.png'), sg.Push()],
    [sg.Input(key='SEARCH_NAME', enable_events=True, expand_x=True),
        sg.Button('', bind_return_key=True, visible=False, key='SEARCH_ENTER'),
        sg.Button("Search", key='SEARCH')],
    [sg.Frame('Selected artist info', size=(430, 100), expand_x=True, layout=[
        [sg.Text('Artist: ', size=(5, 1)), sg.Text('', key='INFO_ARTIST_NAME')],
        [sg.Text('Genres: ', size=(5, 1)), sg.Text('', key='INFO_GENRES')],
        [sg.Text('URL: ', size=(5, 1)), sg.Button('', visible=False, key='INFO_URL')],
    ])],
    [sg.Frame('Top 10:', size=(430, 427), expand_x=True, layout=top_songs_frame)],
    [sg.Frame(' ', size=(430, 70), expand_x=True, border_width=0, layout=[
        [sg.Button('', image_data=base64_image_import('assets/pause.png'), border_width=1, mouseover_colors=('#2c2825', '#2c2825'), key='STOP'), sg.Push(),
         sg.Button('Clear graph', visible=True, key='CLEAR'), ]
        ])],
    # [sg.Text('', size=(1, 1), expand_y=False, key='FILLER2')],
    [sg.Frame('Settings', expand_x=True, layout=[
        [sg.Slider(range=(0, 100), default_value=50, orientation='h', key='VOLUME', enable_events=True),
            sg.Text('\nVolume')],
        [sg.Slider(range=(1, 19), default_value=10, orientation='h', key='SLIDER_CHILD_NO', enable_events=True),
            sg.Text('\nNo of related artists to show')],
        [sg.Checkbox('Show already viewed artists', default=False, enable_events=True, key='DUPES')],
        [sg.Checkbox('Collapsing expanded artist', default=False, enable_events=True, key='COLLAPSE')],
    ])],
    [sg.Text('DEBUG INFO: ', visible=False, key='DEBUG_I'),
     sg.Image('assets/dial_green.png', size=(16, 16), background_color=graph_background_color, visible=True, key='DIAL'),
     sg.Text('Ready', size=(50, 1), expand_x=False, justification='left', visible=True, key='DEBUG')],
], size=size_control, pad=1, vertical_alignment='top', key='CCOL')

layout = [[control_col, graph]]


def make_window(size_mode=None, debug=False):
    s = default_win_size
    if size_mode == '4k':
        pass
    if debug:
        s = default_win_size
    window = sg.Window('Artist Explorer', layout, resizable=True, finalize=True, size=s, margins=(2, 2), return_keyboard_events=True, disable_close=False)
    window.bind('<Configure>', "RESIZE")  # ????

    return window


def dial_busy_state(window, busy):
    if busy:
        window['DIAL'].update('assets/dial_red.png')
    else:
        window['DIAL'].update('assets/dial_green.png')


def debug_draw_ref(graph):
    graph.draw_circle((0, 0), 50)
    graph.draw_circle((0, 0), 10)
    graph.draw_circle((0, 0), 1)

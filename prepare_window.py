import PySimpleGUI as sg
from PIL import Image
from io import BytesIO
import base64

ico_path = 'assets/spotify_ico.png'
graph_background_color = 'white'
ico_size = (64, 64)
album_ico_size = (32, 32)
name_size = (12, 3)
song_name_size = (22, 1)
history_name_size = (12, 1)
history_ico_size = (24, 24)
filler_size = (1, 2)
children_no = 10
history_size = 30
child_icos = child_names = ['', '', '', '', '', '', '', '', '', '']
sg.theme('DarkAmber')
hardcoded_slip = 'Hardcoded Slipknot'


def base64_image_import(path):
    image = Image.open(path)
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue())
    return b64


top_songs_frame = [
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY1', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_1', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_1', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Dodaj do pomidorka', key='SONG_ADD1', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY2', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_2', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_2', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Dodaj do pomidorka', key='SONG_ADD2', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY3', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_3', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_3', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Dodaj do pomidorka', key='SONG_ADD3', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY4', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_4', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_4', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Dodaj do pomidorka', key='SONG_ADD4', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY5', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_5', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_5', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Dodaj do pomidorka', key='SONG_ADD5', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY6', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_6', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_6', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Dodaj do pomidorka', key='SONG_ADD6', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY7', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_7', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_7', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Dodaj do pomidorka', key='SONG_ADD7', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY8', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_8', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_8', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Dodaj do pomidorka', key='SONG_ADD8', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY9', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_9', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_9', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Dodaj do pomidorka', key='SONG_ADD9', visible=False)],
        [sg.Button(image_data=base64_image_import('assets/play.ico'), key='SONG_PLAY10', visible=False),
            sg.Image('', size=album_ico_size, enable_events=True, key='SONG_ALBUM_ICO_10', background_color=graph_background_color, visible=False),
            sg.Text('', key='SONG_NAME_10', enable_events=True, visible=False, size=song_name_size),
            sg.Button('Dodaj do pomidorka', key='SONG_ADD10', visible=False)],
        [sg.VPush()],
]


parent_row1 = [sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='PARENT_ICO_1'),
               sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='PARENT_NAME_1'),
               sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='PARENT_ICO_2'),
               sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='PARENT_NAME_2'),
               sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='PARENT_ICO_3'),
               sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='PARENT_NAME_3'),
               sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='PARENT_ICO_4'),
               sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='PARENT_NAME_4'),
               sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='PARENT_ICO_5'),
               sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='PARENT_NAME_5'),
]

parent_row2 = [sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='PARENT_ICO_6'),
               sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='PARENT_NAME_6'),
               sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='PARENT_ICO_7'),
               sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='PARENT_NAME_7'),
               sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='PARENT_ICO_8'),
               sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='PARENT_NAME_8'),
               sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='PARENT_ICO_9'),
               sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='PARENT_NAME_9'),
               sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='PARENT_ICO_10'),
               sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='PARENT_NAME_10'),
]

current_row = [sg.Push(background_color=graph_background_color),
               sg.Image('', size=ico_size, background_color=graph_background_color, enable_events=True, key='CURRENT_ICO'),
               sg.Button(hardcoded_slip, size=name_size, button_color='#FFFDD0', border_width=2, key='CURRENT_NAME'),
               sg.Push(background_color=graph_background_color)
]

child_row1 = [sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='CHILD_ICO_1'),
              sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='CHILD_NAME_1'),
              sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='CHILD_ICO_2'),
              sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='CHILD_NAME_2'),
              sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='CHILD_ICO_3'),
              sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='CHILD_NAME_3'),
              sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='CHILD_ICO_4'),
              sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='CHILD_NAME_4'),
              sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='CHILD_ICO_5'),
              sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='CHILD_NAME_5'),
]

child_row2 = [sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='CHILD_ICO_6'),
              sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='CHILD_NAME_6'),
              sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='CHILD_ICO_7'),
              sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='CHILD_NAME_7'),
              sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='CHILD_ICO_8'),
              sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='CHILD_NAME_8'),
              sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='CHILD_ICO_9'),
              sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='CHILD_NAME_9'),
              sg.Image('', size=ico_size, enable_events=True, background_color=graph_background_color, key='CHILD_ICO_10'),
              sg.Button('', button_color=graph_background_color, border_width=0, size=name_size, key='CHILD_NAME_10'),
]


history_col1 = sg.Column(layout=[
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_01'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_01'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_02'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_02'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_03'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_03'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_04'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_04'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_05'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_05'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_06'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_06'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_07'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_07'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_08'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_08'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_09'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_09'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_10'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_10'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_11'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_11'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_12'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_12'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_13'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_13'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_14'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_14'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_15'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_15'), ],
])


history_col2 = sg.Column(layout=[
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_16'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_16'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_17'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_17'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_18'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_18'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_19'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_19'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_20'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_20'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_21'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_21'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_22'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_22'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_23'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_23'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_24'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_24'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_25'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_25'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_26'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_26'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_27'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_27'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_28'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_28'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_29'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_29'), ],
    [sg.Image('', size=history_ico_size, enable_events=True, visible=False, background_color=sg.theme_background_color(), key='HISTORY_ICO_30'),
     sg.Button('', size=history_name_size, visible=False, key='HISTORY_NAME_30'), ],
])


control_col = sg.Column(layout=[
    [sg.Frame('Selected artist info', size=(410, 100), expand_x=True, layout=[
        [sg.Text('Artist: '), sg.Text('', key='INFO_ARTIST_NAME')],
        [sg.Text('Genres: '), sg.Text('', key='INFO_ARTIST_GENRE')],
        [sg.Text('URL: '), sg.Button('', visible=False, key='INFO_ARTIST_URL')],
    ])],
    [sg.Frame('Top 10:', size=(410, 430), expand_x=True, layout=top_songs_frame)],
    [sg.Push(), sg.Button('', image_data=base64_image_import('assets/pause.png'), key='STOP'), sg.Push(),
     sg.Button('Dupcia', key='DUPCIA'), sg.Push(),
     sg.Button('Clear viewed', visible=True, key='CLEAR_VIEWED'), sg.Push()],
    [sg.VPush()],
])


artists_col = sg.Column(background_color=graph_background_color, size=(940, 565), layout=[
    [sg.Button("Search: ", key='SEARCH'), sg.Input(key='SEARCH_NAME', enable_events=True, expand_x=True),
        sg.Button('', bind_return_key=True, visible=False, key='SEARCH_ENTER')],
    [sg.Push()],
    [sg.Text('Current: ', text_color='black', size=(20, 1), background_color=graph_background_color, expand_x=True, justification='center', key='FILLER_2')],
    current_row,
    [sg.Push()],
    [sg.Text('', text_color='black', size=(20, 1), background_color=graph_background_color, expand_x=True, justification='center', visible=True, key='FILLER_3')],
    child_row1,
    child_row2,
    [sg.Text('', text_color='white', size=(6, 4), background_color=graph_background_color)],
    [sg.Push()],
    [sg.Text('', text_color='white', size=(7, 1), background_color=graph_background_color),
     sg.Text('Previously related to:', text_color='black', background_color=graph_background_color, expand_x=True, visible=False, justification='right', key='FILLER_1'),
     sg.Text('', text_color='black', background_color=graph_background_color, expand_x=True, visible=False, justification='left', key='FILLER_PARENT_NAME'),
     sg.Checkbox('Keep', enable_events=True, visible=False, key='KEEP_PARENTS')],
    parent_row1,
    parent_row2,
])


history_col = sg.Column(layout=[
    [sg.Frame('Settings', size=(320, 95), layout=[
        [sg.Slider(range=(0, 100), default_value=50, orientation='h', key='VOLUME', enable_events=True),
            sg.Text('\nMaster volume')],
        [sg.Checkbox('Show viewed artists', default=True, key='DUPES'),
         sg.Push(),
         sg.Button('Clear history', visible=False, key='CLEAR_HISTORY'), sg.Push()],
    ])],
    [sg.Frame('History', size=(320, 475), layout=[[history_col1, history_col2]])],
    [sg.VPush()],
])


layout = [
    [sg.Push(), sg.Text('Dupa :D'), sg.Image('assets/spotify_ico.png'), sg.Push()],
    [history_col, artists_col, control_col],
    [sg.Text('DEBUG INFO: '), sg.Text('', key='DEBUG', size=(50, 1), justification='left')],
]


def hide_children(window):
    for i in range(children_no):
        child_name_key = 'CHILD_NAME_' + str(i + 1)
        child_ico_key = 'CHILD_ICO_' + str(i + 1)
        window[child_ico_key].update(visible=False)
        window[child_name_key].update(visible=False)
        window[child_ico_key].update('')
        window[child_name_key].update('')


def clear_history(window):
    for i in range(history_size):
        if i < 9:
            history_name_key = 'HISTORY_NAME_0' + str(i + 1)
            history_ico_key = 'HISTORY_ICO_0' + str(i + 1)
        else:
            history_name_key = 'HISTORY_NAME_' + str(i + 1)
            history_ico_key = 'HISTORY_ICO_' + str(i + 1)
        window[history_ico_key].update('')
        window[history_name_key].update('')
        window[history_ico_key].update(visible=False)
        window[history_name_key].update(visible=False)
        window['CLEAR_HISTORY'].update(visible=False)
    return 1


def clear_parents(window):
    for i in range(10):
        parent_name_key = 'PARENT_NAME_' + str(i + 1)
        parent_ico_key = 'PARENT_ICO_' + str(i + 1)
        window[parent_name_key].update('')
        window[parent_ico_key].update('')
        window[parent_name_key].update(visible=False)
        window[parent_ico_key].update(visible=False)
    window['FILLER_1'].update(visible=False)
    window['KEEP_PARENTS'].update(visible=False)
    window['FILLER_PARENT_NAME'].update(visible=False)
    window['FILLER_PARENT_NAME'].update('')
    return 1

import PySimpleGUI as sg
from PIL import Image, ImageDraw
from pygame import mixer
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests
import webbrowser
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
#import time

import prepare_window
from prepare_window import layout
from config import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_CLIENT_URL, SCOPE_MASTER
from config import pomidorek_id

max_child = 10
no_of_top = 10
max_history = 10


spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                                    client_secret=SPOTIPY_CLIENT_SECRET,
                                                    redirect_uri=SPOTIPY_CLIENT_URL,
                                                    scope=SCOPE_MASTER))
mixer.init()
s = requests.session()


def clear_top():
    for i in range(10):
        song_play_key = 'SONG_PLAY' + str(i + 1)
        song_name_key = 'SONG_NAME_' + str(i + 1)
        song_ico_key = 'SONG_ALBUM_ICO_' + str(i + 1)
        song_pomidorek = 'SONG_ADD' + str(i + 1)
        window[song_name_key].update('')
        window[song_ico_key].update('')
        window[song_name_key].update(visible=False)
        window[song_ico_key].update(visible=False)
        window[song_play_key].update(visible=False)
        window[song_pomidorek].update(visible=False)
    return 1


def download_shit(url, name):
    if not url or url == '':
        return 0
    preview = s.get(url)
    with open(str(name), "wb") as f:
        f.write(preview.content)

    # resize everything but mp3
    if name[-1] != '3':
        img = Image.open(name).resize(prepare_window.album_ico_size)
        img.save(name)


def genres_string(genres):
    text = ''
    for genre in genres:
        text += genre
        text += ', '
    return text


def make_artist_ico(url, ico_name):
    if url == '':
        return 0
    preview = s.get(url)
    with open(str(ico_name), "wb") as f:
        f.write(preview.content)
    img = Image.open(ico_name).resize(prepare_window.ico_size).convert("RGB")
    h, w = img.size
    alpha = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice(((0, 0), (h, w)), 0, 360, fill=255)
    # noinspection PyTypeChecker
    npimg = np.dstack((np.array(img), np.array(alpha)))
    Image.fromarray(npimg).save(ico_name)


def children_update(artist):
    related = spotify.artist_related_artists(artist['id'])['artists']
    related_to_show = []
    for ar in related:
        if ar['id'] not in Viewed:
            a = [ar['name'], ar['id'], ar['images'][0]['url']]
            related_to_show.append(a)
            if not window['DUPES'].get():
                Viewed.append(ar['id'])

    while len(related_to_show) <= max_child:
        related_to_show.append(['', '', ''])

    urls = []
    names = []

    for i in range(max_child):
        ico_name = 'assets/temp/c_ico' + str(i+1) + '.png'
        urls.append(related_to_show[i][2])
        names.append(ico_name)

    pool = ThreadPool(30)  # play with ``processes`` for best results
    pool.starmap(make_artist_ico, zip(urls, names))  # This line blocks, look at map_async
    pool.close()  # the process pool no longer accepts new tasks

    for i in range(max_child):
        ico_name = 'assets/temp/c_ico' + str(i+1) + '.png'
        ico_key = 'CHILD_ICO_' + str(i+1)
        ico_key_name = 'CHILD_NAME_' + str(i+1)
        if related_to_show[i][0] == '':
            window[ico_key].update('')
            window[ico_key_name].update('')
            window[ico_key].update(visible=False)
            window[ico_key_name].update(visible=False)
        else:
            window[ico_key].update(ico_name)
            window[ico_key_name].update(related_to_show[i][0])
            window[ico_key].update(visible=True)
            window[ico_key_name].update(visible=True)
            child_ids[i] = related_to_show[i][1]


def top_tracks(artist):
    top_songs = spotify.artist_top_tracks(artist['id'])['tracks']
    urls = []
    names = []

    for nr in range(len(top_songs)):
        if top_songs[nr]['preview_url']:
            preview_name = 'assets/temp/preview' + str(nr + 1) + '.mp3'
            ico_name = 'assets/temp/top_ico' + str(nr + 1) + '.png'
            urls.append(top_songs[nr]['preview_url'])
            urls.append(top_songs[nr]['album']['images'][0]['url'])
            names.append(preview_name)
            names.append(ico_name)

    pool = ThreadPool(2*no_of_top)  # play with ``processes`` for best results
    pool.starmap(download_shit, zip(urls, names))  # This line blocks, look at map_async
    pool.close()  # the process pool no longer accepts new tasks

    bugged = 0
    for pff in range(len(top_songs)):
        nr = pff - bugged
        play_key = 'SONG_PLAY' + str(nr + 1)
        album_ico_key = 'SONG_ALBUM_ICO_' + str(nr + 1)
        title_key = 'SONG_NAME_' + str(nr + 1)
        ico_name = 'assets/temp/top_ico' + str(nr + 1) + '.png'
        add_key = 'SONG_ADD' + str(nr + 1)
        if top_songs[pff]['preview_url']:
            window[play_key].update(visible=True)
            window[album_ico_key].update(ico_name)
            window[album_ico_key].update(visible=True)
            window[title_key].update(top_songs[nr]['name'])
            window[title_key].update(visible=True)
            window[add_key].update(visible=True)
        else:
            bugged += 1
            window[album_ico_key].update('')
            window[title_key].update('')
            window[play_key].update(visible=False)
            window[album_ico_key].update(visible=False)
            window[title_key].update(visible=False)
            window[add_key].update(visible=False)


def dummy():
    print('dummy')


def search_by_name(name):
    current_artist = spotify.search(name, type='artist')
    return current_artist['artists']['items'][0]['id']


def copy_to_parents():
    for i in range(max_child):
        child_ico_name = 'assets/temp/c_ico' + str(i + 1) + '.png'
        parent_ico_name = 'assets/temp/parent_ico' + str(i + 1) + '.png'
        child_name_key = 'CHILD_NAME_' + str(i + 1)
        parent_ico_key = 'PARENT_ICO_' + str(i + 1)
        parent_name_key = 'PARENT_NAME_' + str(i + 1)
        parent_ids[i] = child_ids[i]
        if window[child_name_key].get_text() == '':
            window[parent_name_key].update('')
            window[parent_ico_key].update('')
            window[parent_ico_key].update(visible=False)
            window[parent_name_key].update(visible=False)

        else:
            Image.open(str(child_ico_name)).save(str(parent_ico_name))
            window[parent_name_key].update(window[child_name_key].get_text())
            window[parent_ico_key].update(parent_ico_name)
            window[parent_ico_key].update(visible=True)
            window[parent_name_key].update(visible=True)


def update(current_artist):
    #start_time = time.time()
    current_artist_name = current_artist['name']
    try:
        make_artist_ico(current_artist['images'][0]['url'], 'assets/temp/current.png')
        window['CURRENT_ICO'].update('assets/temp/current.png')
    except:
        window['CURRENT_ICO'].update('assets/empty_ico.png')

    window['CURRENT_NAME'].update(current_artist['name'])
    window['INFO_ARTIST_NAME'].update(current_artist_name)
    window['INFO_ARTIST_GENRE'].update(genres_string(current_artist['genres']))
    window['INFO_ARTIST_URL'].update(current_artist['external_urls']['spotify'])
    window['INFO_ARTIST_URL'].update(visible=True)
    children_update(current_artist)
    if window['CHILD_NAME_1'] == '':
        debug('Może warto przeczyścić historię oglądanych?')
    clear_top()
    top_tracks(current_artist)
    #print('Update took', time.time() - start_time, ' s')


def debug(text):
    window['DEBUG'].update(text)


def add_current_to_history():
    if history_index > prepare_window.history_size:
        return history_index + 1

    current_artist_name = window['CURRENT_NAME'].get_text()
    if current_artist_name in history:
        return history_index

    history.append(current_artist_name)
    hi = history_index
    if hi < 10:
        hi = '0' + str(hi)
    else:
        hi = str(hi)
    history_name_key = 'HISTORY_NAME_' + hi
    history_ico_key = 'HISTORY_ICO_' + hi
    history_ico_path = 'assets/temp/hist_ico' + hi + '.png'
    img = Image.open('assets/temp/current.png').resize(prepare_window.history_ico_size)
    img.save(history_ico_path)

    window[history_name_key].update(current_artist_name)
    window[history_ico_key].update(history_ico_path)
    window[history_ico_key].update(visible=True)
    window[history_name_key].update(visible=True)
    window['CLEAR_HISTORY'].update(visible=True)
    history_ids[history_index - 1] = current_artist_id
    return history_index + 1


def get_nr(event):
    nr = int(event[-1])
    if nr == 0:
        nr = 10
    return nr


current_artist_name = 'ERROR'
current_artist_id = 'ERROR'
current_artist = 'ERROR'
music = 'ERROR'
child_ids = ['', '', '', '', '', '', '', '', '', '']
parent_ids = ['', '', '', '', '', '', '', '', '', '']
history_ids = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
pomidorek_events = ['SONG_ADD1', 'SONG_ADD2', 'SONG_ADD3', 'SONG_ADD4', 'SONG_ADD5', 'SONG_ADD6', 'SONG_ADD7', 'SONG_ADD8', 'SONG_ADD9', 'SONG_ADD10']
play_events = ['SONG_PLAY1', 'SONG_PLAY2', 'SONG_PLAY3', 'SONG_PLAY4', 'SONG_PLAY5', 'SONG_PLAY6', 'SONG_PLAY7', 'SONG_PLAY8', 'SONG_PLAY9', 'SONG_PLAY10']
child_ico_events = ['CHILD_ICO_1', 'CHILD_ICO_2', 'CHILD_ICO_3', 'CHILD_ICO_4', 'CHILD_ICO_5', 'CHILD_ICO_6', 'CHILD_ICO_7', 'CHILD_ICO_8', 'CHILD_ICO_9', 'CHILD_ICO_10']
child_name_events = ['CHILD_NAME_1', 'CHILD_NAME_2', 'CHILD_NAME_3', 'CHILD_NAME_4', 'CHILD_NAME_5', 'CHILD_NAME_6', 'CHILD_NAME_7', 'CHILD_NAME_8', 'CHILD_NAME_9', 'CHILD_NAME_10']
parent_ico_events = ['PARENT_ICO_1', 'PARENT_ICO_2', 'PARENT_ICO_3', 'PARENT_ICO_4', 'PARENT_ICO_5', 'PARENT_ICO_6', 'PARENT_ICO_7', 'PARENT_ICO_8', 'PARENT_ICO_9', 'PARENT_ICO_10']
parent_name_events = ['PARENT_NAME_1', 'PARENT_NAME_2', 'PARENT_NAME_3', 'PARENT_NAME_4', 'PARENT_NAME_5', 'PARENT_NAME_6', 'PARENT_NAME_7', 'PARENT_NAME_8', 'PARENT_NAME_9', 'PARENT_NAME_10']
history_ico_events = ['HISTORY_ICO_01', 'HISTORY_ICO_02', 'HISTORY_ICO_03', 'HISTORY_ICO_04', 'HISTORY_ICO_05', 'HISTORY_ICO_06', 'HISTORY_ICO_07', 'HISTORY_ICO_08', 'HISTORY_ICO_09', 'HISTORY_ICO_10', 'HISTORY_ICO_11', 'HISTORY_ICO_12', 'HISTORY_ICO_13', 'HISTORY_ICO_14', 'HISTORY_ICO_15', 'HISTORY_ICO_16', 'HISTORY_ICO_17', 'HISTORY_ICO_18', 'HISTORY_ICO_19', 'HISTORY_ICO_20', 'HISTORY_ICO_21', 'HISTORY_ICO_22', 'HISTORY_ICO_23', 'HISTORY_ICO_24', 'HISTORY_ICO_25', 'HISTORY_ICO_26', 'HISTORY_ICO_27', 'HISTORY_ICO_28', 'HISTORY_ICO_29', 'HISTORY_ICO_30']
history_name_events = ['HISTORY_NAME_01', 'HISTORY_NAME_02', 'HISTORY_NAME_03', 'HISTORY_NAME_04', 'HISTORY_NAME_05', 'HISTORY_NAME_06', 'HISTORY_NAME_07', 'HISTORY_NAME_08', 'HISTORY_NAME_09', 'HISTORY_NAME_10', 'HISTORY_NAME_11', 'HISTORY_NAME_12', 'HISTORY_NAME_13', 'HISTORY_NAME_14', 'HISTORY_NAME_15', 'HISTORY_NAME_16', 'HISTORY_NAME_17', 'HISTORY_NAME_18', 'HISTORY_NAME_19', 'HISTORY_NAME_20', 'HISTORY_NAME_21', 'HISTORY_NAME_22', 'HISTORY_NAME_23', 'HISTORY_NAME_24', 'HISTORY_NAME_25', 'HISTORY_NAME_26', 'HISTORY_NAME_27', 'HISTORY_NAME_28', 'HISTORY_NAME_29', 'HISTORY_NAME_30']
window = sg.Window('Artist penetrator', layout)
event, values = window.read(timeout=10)
flag_expanded = 0
history_index = 1
history = []
Viewed = []


while True:
    event, values = window.read()  # (timeout=50) #timeout -> po 50ms i tak leci pętla

    if event == sg.WIN_CLOSED:
        break

    if event == 'DUPCIA':
        if window['CURRENT_NAME'].get_text() != 'DUPCIA :D':
            history_index = add_current_to_history()
        window['CURRENT_NAME'].update('DUPCIA :D')
        window['DEBUG'].update('Dupcia')
        Viewed = []
        print('Dupcia')

    if event in ['CURRENT_NAME', 'CURRENT_ICO']:
        current_artist_name = window['CURRENT_NAME'].get_text()
        if window['DUPES'].get():
            Viewed = []
        if current_artist_name in [prepare_window.hardcoded_slip, 'DUPCIA :D']:
            window['SEARCH_NAME'].update('Slipknot')
            event = 'SEARCH'
            values['SEARCH_NAME'] = 'Slipknot'
        elif flag_expanded == 1:
            debug('Po co to robisz?')
        elif current_artist != 'ERROR':
            update(current_artist)
            # Image.open('assets/temp/current.png').save('assets/temp/hist_ico1.png')
            # window['HISTORY_ICO_1'].update('assets/temp/hist_ico1.png')
            # window['HISTORY_NAME_1'].update(current_artist['name'])
            flag_expanded = 1
            window['FILLER_3'].update('Related: ')
            window['FILLER_2'].update('Current: ')
            window['SEARCH_NAME'].update('')
            current_artist_id = current_artist['id']

    if event in ['SEARCH', 'SEARCH_ENTER']:
        if values['SEARCH_NAME'] != window['CURRENT_NAME'].get_text():
            prepare_window.hide_children(window)
            if not values['KEEP_PARENTS']:
                prepare_window.clear_parents(window)
            # prepare_window.clear_history(window)
            flag_expanded = 0
            if window['DUPES'].get():
                Viewed = []
            if values['SEARCH_NAME']:
                if window['CURRENT_NAME'].get_text() not in [prepare_window.hardcoded_slip, 'DUPCIA :D', '']:
                    history_index = add_current_to_history()
                current_artist = spotify.artist(search_by_name(values['SEARCH_NAME']))
                current_artist_id = current_artist['id']
                window['CURRENT_NAME'].update(current_artist['name'])
                try:
                    make_artist_ico(current_artist['images'][0]['url'], 'assets/temp/current.png')
                    window['CURRENT_ICO'].update('assets/temp/current.png')
                except:
                    window['CURRENT_ICO'].update('assets/empty_ico.png')
                for i in range(max_child):
                    ico_key = 'CHILD_ICO_' + str(i + 1)
                    ico_key_name = 'CHILD_NAME_' + str(i + 1)
                    window[ico_key].update('')
                    window[ico_key_name].update('')
                    window[ico_key].update(visible=False)
                    window[ico_key_name].update(visible=False)
                    window['FILLER_3'].update('')
                    window['FILLER_2'].update('Search results: ')
            else:
                window['CURRENT_NAME'].update('Wpisz coś, a nie...')
                debug('No i czemu taki jesteś?')
        else:
            debug('')
            debug('Po co szukasz czegoś co już masz?')

    if event in child_ico_events:
        nr = get_nr(event)
        event = 'CHILD_NAME_' + str(nr)

    if event in child_name_events:
        nr = get_nr(event) - 1
        history_index = add_current_to_history()
        #current_artist_name = window[event].get_text()
        current_artist_id = child_ids[nr]
        if current_artist_name != '':
            if not values['KEEP_PARENTS']:
                window['FILLER_1'].update(visible=True)
                window['FILLER_PARENT_NAME'].update(visible=True)
                window['KEEP_PARENTS'].update(visible=True)
                window['FILLER_PARENT_NAME'].update(window['CURRENT_NAME'].get_text())
                copy_to_parents()
            current_artist = spotify.artist(current_artist_id)
            update(current_artist)
            window['DEBUG'].update('')

    if event in parent_ico_events:
        nr = get_nr(event)
        event = 'PARENT_NAME_' + str(nr)

    if event in parent_name_events:
        nr = get_nr(event) - 1
        #current_artist_name = window[event].get_text()
        history_index = add_current_to_history()
        if parent_ids[nr] == '':
            debug('Tu byłby srogi crash')
            print('Tu byłby srogi crash')
        elif parent_ids[nr] == current_artist['id']:
            debug('Już to masz przecież...')
        else:
            #current_artist = spotify.artist(search_by_name(current_artist_name))
            current_artist = spotify.artist(parent_ids[nr])
            current_artist_id = current_artist['id']
            update(current_artist)
            window['DEBUG'].update('')

    if event in history_ico_events:
        nr = event[-2] + event[-1]
        event = 'HISTORY_NAME_' + str(nr)

    if event in history_name_events:
        nr = int(event[-2] + event[-1]) - 1
        #current_artist_name = window[event].get_text()
        if history_ids[nr] == '':
            debug('Tu byłby srogi crash')
            print('Tu byłby srogi crash')
        elif history_ids[nr] == current_artist['id']:
            debug('Już to masz przecież...')
        else:
            current_artist = spotify.artist(history_ids[nr])
            current_artist_id = current_artist['id']
            update(current_artist)

    if event in play_events:
        mixer.stop()
        nr = get_nr(event)
        song_path = 'assets/temp/preview' + str(nr) + '.mp3'
        music = mixer.Sound(song_path)
        music.play()
        music.set_volume(values['VOLUME'] / 100)

    if event == 'STOP':
        mixer.stop()
        music = 'ERROR'

    if event == 'VOLUME':
        if music != 'ERROR':
            music.set_volume(values['VOLUME'] / 100)

    if event == 'INFO_ARTIST_URL':
        webbrowser.open(window['INFO_ARTIST_URL'].get_text())

    if event == 'KEEP_PARENTS':
        current_artist_id = current_artist_id

    if event == 'CLEAR_HISTORY':
        history_index = prepare_window.clear_history(window)

    if event == 'CLEAR_VIEWED':
        Viewed = []
        flag_expanded = 0

    if event in pomidorek_events:
        # noinspection PyTypeChecker
        top_songs = spotify.artist_top_tracks(current_artist['id'])['tracks']
        nr = get_nr(event) - 1
        playlist = spotify.playlist_items(pomidorek_id)['items']
        if not nr > len(top_songs):
            track = top_songs[nr]['id']
            playlist_songs = []
            for song in playlist:
                playlist_songs.append(song['track']['id'])
            if track not in playlist_songs:
                spotify.playlist_add_items(pomidorek_id, items=[track])

window.close()

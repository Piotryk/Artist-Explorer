import os
import time
import requests
import numpy as np
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool

import artist_graph as artist_graph_lib
import window
from window import debug, dial_busy_state

max_search_artists = 5
s = requests.session()


def get_credentials(filepath):
    f = open(filepath, 'r')
    spotify_id = f.readline()
    spotify_secret = f.readline()
    spotify_url = f.readline()
    f.close()

    spotify_id = spotify_id.strip('SPOTIFY_CLIENT_ID')
    spotify_id = spotify_id.strip('\r\n\t \'\"=')
    spotify_secret = spotify_secret.replace('SPOTIFY_CLIENT_SECRET', '')
    spotify_secret = spotify_secret.strip('\r\n\t \'\"=')
    spotify_url = spotify_url.replace('SPOTIFY_CLIENT_URL', '')
    spotify_url = spotify_url.strip('\r\n\t \'\"=')

    return spotify_id, spotify_secret, spotify_url


def add_song_to_playlist(main_window, spotify, event, top_songs):
    nr = int(event[-1])
    if nr == 0:
        nr = 10
    if len(top_songs) > nr:
        spotify.current_user_saved_tracks_add(tracks=[top_songs[nr - 1]])
        song_name = main_window['SONG_NAME_' + str(nr)].get()
        debug(main_window, f'{song_name} added to Liked Songs')


def download_top_song(url, name):
    if not url or url == '':
        return None
    preview = s.get(url)
    with open(str(name), "wb") as f:
        f.write(preview.content)

    # resize ico
    if name[-1] != '3':  # but do not resize .mp3 file
        img = Image.open(name).resize(window.album_ico_size)
        img.save(name)


def make_top_tracks(main_window, spotify, spotify_id):
    clear_top_tracks(main_window)
    top_songs = spotify.artist_top_tracks(spotify_id)['tracks']
    urls = []
    names = []
    uris = []

    if not os.path.exists("assets/temp/"):
        os.makedirs("assets/temp/")

    for nr in range(0, len(top_songs)):
        if top_songs[nr]['preview_url']:
            preview_name = f'assets/temp/preview{nr + 1}.mp3'
            ico_name = f'assets/temp/top_ico{nr + 1}.png'
            urls.append(top_songs[nr]['preview_url'])
            if top_songs[nr]['album']:
                if top_songs[nr]['album']['images']:
                    urls.append(top_songs[nr]['album']['images'][-1]['url'])
            names.append(preview_name)
            names.append(ico_name)

    # Downloading top tracks asynchronously to speed up the process
    # from 2 - 2.5 s to .2 - .3 s
    pool = ThreadPool(2 * 10)
    pool.starmap(download_top_song, zip(urls, names))
    pool.close()
    pool.join()

    bugged = 0
    for no in range(1, len(top_songs) + 1):
        nr = no - bugged
        play_key = f'SONG_PLAY_{nr}'
        album_ico_key = f'SONG_ALBUM_ICO_{nr}'
        title_key = f'SONG_NAME_{nr}'
        ico_name = f'assets/temp/top_ico{nr}.png'
        add_key = f'SONG_ADD_{nr}'
        if top_songs[no - 1]['preview_url']:
            uris.append(top_songs[no - 1]['uri'])
            main_window[play_key].update(visible=True)
            main_window[album_ico_key].update(ico_name)
            main_window[album_ico_key].update(visible=True)
            main_window[title_key].update(top_songs[no - 1]['name'])
            main_window[title_key].update(visible=True)
            main_window[add_key].update(visible=True)
        else:
            bugged += 1
            main_window[album_ico_key].update('')
            main_window[title_key].update('')
            main_window[play_key].update(visible=False)
            main_window[album_ico_key].update(visible=False)
            main_window[title_key].update(visible=False)
            main_window[add_key].update(visible=False)
    return uris


def clear_top_tracks(main_window):
    for nr in range(1, 11):
        play_key = f'SONG_PLAY_{nr}'
        album_ico_key = f'SONG_ALBUM_ICO_{nr}'
        title_key = f'SONG_NAME_{nr}'
        add_key = f'SONG_ADD_{nr}'
        main_window[album_ico_key].update('')
        main_window[title_key].update('')
        main_window[play_key].update(visible=False)
        main_window[album_ico_key].update(visible=False)
        main_window[title_key].update(visible=False)
        main_window[add_key].update(visible=False)


def play_preview(main_window, mixer, event, values):
    mixer.stop()
    nr = int(event[-1])
    if nr == 0:
        nr = 10
    song_path = f'assets/temp/preview{nr}.mp3'
    music = mixer.Sound(song_path)
    music.play()
    music.set_volume(values['VOLUME'] / 100)
    song_name = main_window['SONG_NAME_' + str(nr)].get()
    debug(main_window, f'Playing {song_name}')
    return music


def make_genres_string(genres):
    text = ''
    for genre in genres:
        if text:
            text += ', '
        text += genre[0].upper() + genre[1:]
    return text


def update_artist_info(main_window, name, genres, url):
    main_window['INFO_ARTIST_NAME'].update(name)
    main_window['INFO_GENRES'].update(make_genres_string(genres))
    main_window['INFO_URL'].update(url)
    main_window['INFO_URL'].update(visible=True)
    main_window['SEARCH_NAME'].update('')
    main_window.refresh()


def clear_artist_info(main_window):
    main_window['INFO_ARTIST_NAME'].update('')
    main_window['INFO_GENRES'].update('')
    main_window['INFO_URL'].update('')
    main_window['SEARCH_NAME'].update('')
    main_window['INFO_URL'].update(visible=False)
    main_window.refresh()


def expand_artist(main_window, graph, artists_graph, spotify, clicked_figs, no_children):
    top_songs = []
    for node in artists_graph.nodes.values():
        if node.frame_id in clicked_figs:
            # Get the response
            dial_busy_state(main_window, True)
            debug(main_window, 'Getting related artists from spotify')
            main_window.refresh()

            related = spotify.artist_related_artists(node.spotify_id)
            added = artists_graph.add_nodes_from_spotify_response(related['artists'], max_size=no_children, parent_id=node.id, show_duplicates=main_window['DUPES'].get())
            if added == 0:
                if not main_window['DUPES'].get():
                    debug(main_window, 'All similar artists are already shown')
                else:
                    debug(main_window, 'Nothing found :c')
                dial_busy_state(main_window, False)
                break
            # Update UI
            update_artist_info(main_window, node.name, node.genres, node.url)
            top_songs = make_top_tracks(main_window, spotify, node.spotify_id)

            # Draw
            # artists_graph.draw(graph, main_window)
            artists_graph.draw_animate(graph, main_window)
            debug(main_window, 'Ready')
            dial_busy_state(main_window, False)
            main_window.refresh()
            break
    return top_songs


def collapse_artist(main_window, graph, artists_graph, id):
    dial_busy_state(main_window, True)
    debug(main_window, f'Collapsing node {artists_graph.nodes[id].name}')

    artists_graph.active_node_id = id
    artists_graph.draw_animate(graph, main_window, collapse=True, collapse_id=id)
    artists_graph.collapse_node_v2(id)
    artists_graph.draw_animate(graph, main_window)
    debug(main_window, 'Ready')
    dial_busy_state(main_window, False)
    main_window.refresh()


def on_click(main_window, graph, artists_graph, spotify, clicked_figs, no_children):
    top_songs = []
    for node in artists_graph.nodes.values():
        if node.frame_id in clicked_figs:
            artists_graph.active_node = node.id
            if node.expanded and main_window['COLLAPSE'].get():
                top_songs = make_top_tracks(main_window, spotify, node.spotify_id)
                collapse_artist(main_window, graph, artists_graph, node.id)
                #clear_artist_info(main_window)
                #clear_top_tracks(main_window)
            elif node.expanded and not main_window['COLLAPSE'].get():            
                dial_busy_state(main_window, True)
                debug(main_window, f'Getting {node.name}\'s top tracks')
                top_songs = make_top_tracks(main_window, spotify, node.spotify_id)

                debug(main_window, 'Ready')
                dial_busy_state(main_window, False)
            else:
                top_songs = expand_artist(main_window, graph, artists_graph, spotify, clicked_figs, no_children)
            break
    return top_songs


def artists_graph_from_search_graph(main_window, graph, artists_graph, clicked_figs):
    for node in artists_graph.nodes.values():
        if node.frame_id in clicked_figs:
            debug(main_window, 'Getting related artists from spotify')
            dial_busy_state(main_window, True)

            starting_artist_node = node
            new_offset = starting_artist_node.old_position + artists_graph.offset
            scale = artists_graph.scale_level
            starting_artist_node.old_position = [0, 0]
            starting_artist_node.id = 0
            artists_graph.clear(graph)
            del artists_graph  # is this necessary?

            artists_graph = artist_graph_lib.Graph(starting_artist_node, offset=new_offset, scale_level=scale)
            artists_graph.draw(graph, main_window)

            y = (window.size_graph[1] - artist_graph_lib.image_sizes[artists_graph.scale_level]) / 2
            destination = np.array([100, y])
            anim_len = 20
            anim_len_s = 0.3
            diff = (artists_graph.offset - destination) / anim_len
            for i in range(anim_len, -1, -1):
                artists_graph.offset -= diff
                graph.move(-diff[0], -diff[1])
                main_window.refresh()
                time.sleep(anim_len_s / anim_len)

            #related = spotify.artist_related_artists(node.spotify_id)
            #artists_graph.add_nodes_from_spotify_response(related['artists'], max_size=no_children, parent_id=node.id)
            #artists_graph.draw_animate(graph, main_window)

            debug(main_window, 'Ready')
            main_window.refresh()
            dial_busy_state(main_window, False)
            break
    return artists_graph


def search(main_window, graph, artists_graph, spotify, values):
    if values['SEARCH_NAME'] == '':
        debug(main_window, 'Please provide artist name.')
        debug(main_window, 'Czemu taki jesteś?')
        main_window.refresh()
        return artists_graph, False, False
    if values['SEARCH_NAME'] == main_window['INFO_ARTIST_NAME']:
        debug(main_window, 'Po co szukasz czegoś co już masz?')
        main_window.refresh()
        return artists_graph, False, False

    empty_state = False
    search_state = True
    current_artist = spotify.search(values['SEARCH_NAME'], type='artist')
    if not current_artist['artists']['items']:
        debug(main_window, 'Nothing found :c')
        return artists_graph, True, False

    search_offset = [100, 666]  # 198
    artists_graph = artist_graph_lib.SearchGraph(current_artist['artists']['items'], max_search_artists,
                                                 offset=search_offset)
    artists_graph.draw(graph, main_window, center=True)
    return artists_graph, empty_state, search_state


def drag(graph, artists_graph, x, y, start_point, end_point, dragging, prev_x, prev_y):
    if not dragging:
        start_point = (x, y)
        dragging = True
        prev_x = x
        prev_y = y
    else:
        end_point = (x, y)

    dx = x - prev_x
    dy = y - prev_y
    prev_x = x
    prev_y = y
    if start_point and end_point:
        graph.move(dx, dy)
        artists_graph.offset += np.array([dx, dy])

    return dragging, start_point, end_point, prev_x, prev_y


def zoom_in(main_window, graph, artists_graph, center):
    dial_busy_state(main_window, True)
    debug(main_window, f'Zooming in')

    if artists_graph.scale_level >= artist_graph_lib.max_scale_level:
        return
    scale = 4 / 3
    if artists_graph.scale_level % 2 == 0:
        scale = 1.5
    d = (artists_graph.offset - center) * (1 - scale)

    artists_graph.scale_level += 1
    artists_graph.offset = artists_graph.offset - d
    artists_graph.draw(graph, main_window)

    debug(main_window, 'Ready')
    dial_busy_state(main_window, False)


def zoom_out(main_window, graph, artists_graph, center):
    dial_busy_state(main_window, True)
    debug(main_window, f'Zooming out')
    if artists_graph.scale_level <= artist_graph_lib.min_scale_level:
        return
    scale = 2 / 3
    if artists_graph.scale_level % 2 == 0:
        scale = 3 / 4

    d = (artists_graph.offset - center) * (1 - scale)

    artists_graph.scale_level -= 1
    artists_graph.offset = artists_graph.offset - d
    artists_graph.draw(graph, main_window)

    debug(main_window, 'Ready')
    dial_busy_state(main_window, False)

import networkx as nx
import time
import os
import PySimpleGUI as sg
import numpy as np
import base64
import requests
from multiprocessing.dummy import Pool as ThreadPool
from io import BytesIO
from PIL import Image, ImageDraw

font = 'arial'
node_width_factor = 3.3
image_sizes = np.array([8, 12, 16, 24, 32, 48, 64, 96, 128, 192, ])
image_pads = [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, ]
node_sizes = np.array([[node_width_factor * (image_sizes[i] + 2 * image_pads[i]), image_sizes[i] + 2 * image_pads[i]] for i in range(len(image_sizes))], dtype=int)
font_sizes = (image_sizes / 4).astype(int)
max_name_len = 16

min_scale_level = 0
max_scale_level = len(image_sizes) - 1

animation_len = 10
animation_len_s = 0.3


def down_artist_ico(images):
    url = ''
    path = 'assets/empty_ico.png'
    if images:
        url = images[-1]['url']

    if url != '':
        session = requests.session()
        path = 'assets/temp/' + url[24:] + '.png'
        preview = session.get(url)
        with open(str(path), "wb") as f:
            f.write(preview.content)
    w = h = 128
    img = Image.open(path).resize((w, h)).convert("RGB")
    alpha = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice(((0, 0), (h, w)), 0, 360, fill=255)
    npimg = np.dstack((np.array(img), np.array(alpha)))
    img = Image.fromarray(npimg)
    return img


def make_artist_icos(img):
    results = []
    for img_size in image_sizes:
        w = h = img_size
        image = img.resize((w, h))
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        results.append(base64.b64encode(buffer.getvalue()))
    return results


# Depreceated
def make_artist_icons(images):
    url = ''
    results = []
    path = 'assets/empty_ico.png'
    session = requests.session()

    if images:
        url = images[-1]['url']
        path = 'assets/temp/' + url[24:] + '.png'

    if url != '':
        preview = session.get(url)
        with open(str(path), "wb") as f:
            f.write(preview.content)

    w = h = 128
    img = Image.open(path).resize((w, h)).convert("RGB")
    alpha = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice(((0, 0), (h, w)), 0, 360, fill=255)
    npimg = np.dstack((np.array(img), np.array(alpha)))
    img = Image.fromarray(npimg)
    #img.save('assets/temp/' + url[24:] + '.png')

    for img_size in image_sizes:
        w = h = img_size
        image = img.resize((w, h))
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        results.append(base64.b64encode(buffer.getvalue()))
    return results


def unify_art_name_len(name, max_lenght):
    leng = max_lenght
    for c in name:
        if c in ['m', 'w', ' ', '&']:
            leng -= 1
        if leng <= max_name_len / 2:
            break
    if len(name) > leng:
        name = name[:leng] + '..'
    return name


class Node:
    def __init__(self, id, name, column=0, parent=None, all_parents=None, spotify_id=None, images=None, genres=None, url=None, old_position=None):
        if images is None:
            images = []
        if all_parents is None:
            all_parents = []
        self.id = id
        self.name = name
        self.spotify_id = spotify_id
        self.genres = genres
        self.url = url
        self.column = column
        self.images = images
        self.active = False
        self.expanded = False

        self.children = []
        self.parent = parent
        self.all_parents = all_parents.copy()
        if parent is not None:
            self.all_parents.append(parent)

        self.dfs_id = -1
        self.old_position = old_position
        if old_position is None:
            self.old_position = np.array([0, 0])

        self.frame_id = None
        self.ico_id = None
        self.text_id = None
        self.connector_id = None    # deprecated
        self.connector_id_p1 = None
        self.connector_id_p2 = None
        self.connector_id_p3 = None

    def print(self):
        #print(self.id, self.name, self.column, self.row, f'dfs_id: {self.dfs_id}', 'p: ',self.parent, self.all_parents, )
        print(self.id, self.name, f'col: {self.column},', f'parent: {self.parent}')


class Graph:
    def __init__(self, node=None, offset=None, scale_level=5):
        if offset is None:
            offset = [0, 0]
        self.start_node = node
        self.nodes = {0: node}
        self.active_node_id = 0

        self.width = 1
        self.scale_level = scale_level
        self.offset = np.array(offset, dtype=float)

    def add_nodes_from_spotify_response(self, spotify_response, max_size=10, parent_id=None, show_duplicates=True):
        if not spotify_response:
            return 0

        if not os.path.exists("assets/temp/"):
            os.makedirs("assets/temp/")

        added = 0
        icons_urls = [artist['images'] for artist in spotify_response]
        pool = ThreadPool(len(icons_urls))
        results = pool.starmap(down_artist_ico, zip(icons_urls))
        pool.close()
        images = {artist['id']: results[i] for i, artist in enumerate(spotify_response, start=0)}

        all_artists_in_graph = [node.spotify_id for node in self.nodes.values()]
        for artist in spotify_response:
            if added >= max_size:
                break

            if not show_duplicates and artist['id'] in all_artists_in_graph:
                continue
            added += 1
            node_id = max(self.nodes.keys()) + 1
            node_name = artist['name']
            #node_images = make_artist_icons(artist['images'])
            node_images = make_artist_icos(images[artist['id']])
            node_genres = artist['genres']
            node_url = artist['external_urls']['spotify']
            node_column = self.nodes[parent_id].column + 1
            node_old_pos = self.nodes[parent_id].old_position
            new_node = Node(node_id, node_name, node_column,
                            parent=parent_id, spotify_id=artist['id'], images=node_images,
                            url=node_url, genres=node_genres, old_position=node_old_pos)

            self.nodes[node_id] = new_node
            self.active_node_id = parent_id

            self.nodes[parent_id].children.append(node_id)
            self.nodes[parent_id].active = True
            self.nodes[parent_id].expanded = True

        self.dfs()
        return added

    def draw(self, graph, window):
        if self.start_node is None:
            return None

        node_size = node_sizes[self.scale_level]
        text_offset = np.array([image_pads[self.scale_level] * 2 + image_sizes[self.scale_level], node_size[1] / 2])
        font_size = font_sizes[self.scale_level]
        connector_offset = np.array([0, node_size[1] / 2])
        connector_parent_offset = np.array([node_size[0], node_size[1] / 2])
        y_spacing = 5 * image_sizes[self.scale_level] / 64  #default dot.exe spacing is [72,72]
        x_spacing = 1.25 * image_sizes[self.scale_level] / 64

        # Utilizing networkx and graphviz to get the node positioning
        g = nx.DiGraph()
        nodes = [n for n in self.nodes.values()]
        nodes.sort(key=lambda x: x.dfs_id)
        for node in nodes:
            g.add_node(node.id, column=node.column)
        for node in nodes:
            for child in node.children:
                g.add_edge(node.id, self.nodes[child].id)
        positions = nx.nx_pydot.graphviz_layout(g, prog=os.getcwd() + '/graphviz/dot.exe')

        positions = {node: (-y * y_spacing, -x * x_spacing) for (node, (x, y)) in positions.items()}
        positions = {node: (x - positions[0][0], y - positions[0][1]) for (node, (x, y)) in positions.items()}

        graph.erase()
        for node in self.nodes.values():
            base_pos = np.array(positions[node.id])
            pos = base_pos + self.offset

            # Draw node frame
            node.frame_id = graph.draw_rectangle(pos, pos + node_size, line_width=2, fill_color='#dddddd')

            # Draw text
            text = unify_art_name_len(node.name, max_name_len)
            node.text_id = graph.draw_text(text, font=(font, font_size), location=tuple(pos + text_offset), text_location=sg.TEXT_LOCATION_LEFT)

            # Draw artist icons
            img = node.images[self.scale_level]
            image_pos = pos + [image_pads[self.scale_level], image_pads[self.scale_level] + image_sizes[self.scale_level]]
            node.img_id = graph.draw_image(data=img, location=tuple(image_pos))

            # Draw connectors from parent to children
            if not node.children:
                continue

            no_children_above = len([child_id for child_id in node.children if positions[child_id][1] > positions[node.id][1]])
            no_children_below = len([child_id for child_id in node.children if positions[child_id][1] < positions[node.id][1]])
            if no_children_above + no_children_below == len(node.children):
                no_children_above -= 0.5

            for x, child_id in enumerate(node.children, start=1):
                child_base_pos = np.array(positions[child_id])
                line_start = pos + connector_parent_offset
                line_end = child_base_pos + self.offset + connector_offset

                if line_start[0] > line_end[0]:
                    continue

                z = abs(x - no_children_above - 1)
                half_pos_offset = 3 / 4 - z / len(node.children) * 0.66
                x_offset_left = (line_end[0] - line_start[0]) * half_pos_offset
                x_offset_right = line_end[0] - line_start[0] - x_offset_left
                y_offset = (line_end[1] - line_start[1]) / 2

                thr = 4

                if abs(y_offset) > thr:
                    if y_offset > 0:
                        node.connector_id_p1 = graph.draw_arc((line_start[0] - x_offset_left, line_end[1]), (line_start[0] + x_offset_left, line_start[1]), 91, 270,  line_width=1, arc_color='black', style='arc')
                        node.connector_id_p2 = graph.draw_arc((line_end[0] - x_offset_right, line_end[1]), (line_end[0] + x_offset_right, line_start[1]), 91, 90, line_width=1, arc_color='black', style='arc')
                    else:
                        node.connector_id_p1 = graph.draw_arc((line_start[0] - x_offset_left, line_start[1]), (line_start[0] + x_offset_left, line_end[1]), 91, 0,  line_width=1, arc_color='black', style='arc')
                        node.connector_id_p2 = graph.draw_arc((line_end[0] - x_offset_right, line_start[1]), (line_end[0] + x_offset_right, line_end[1]), 91, 180, line_width=1, arc_color='black', style='arc')
                else:
                    node.connector_id_p1 = graph.draw_line(tuple(line_start), (line_start[0] + x_offset_left, line_start[1]), width=1)
                    node.connector_id_p1 = graph.draw_line((line_start[0] + x_offset_left, line_start[1]), (line_start[0] + x_offset_left, line_end[1]), width=1)
                    node.connector_id_p1 = graph.draw_line((line_start[0] + x_offset_left, line_end[1]), (line_end[0], line_end[1]), width=1)

            graph.bring_figure_to_front(node.frame_id)
            graph.bring_figure_to_front(node.img_id)
            graph.bring_figure_to_front(node.text_id)

        window.refresh()
        for node in self.nodes.values():
            node.old_position = np.copy(np.array(positions[node.id]))
            node.active = False

    def draw_animate(self, graph, window, collapse=False, collapse_id=0):
        if self.start_node is None:
            return None

        node_size = node_sizes[self.scale_level]
        text_offset = np.array([image_pads[self.scale_level] * 2 + image_sizes[self.scale_level], node_size[1] / 2])
        font_size = font_sizes[self.scale_level]
        connector_offset = np.array([0, node_size[1] / 2])
        connector_parent_offset = np.array([node_size[0], node_size[1] / 2])
        y_spacing = 5 * image_sizes[self.scale_level] / 64  #default dot.exe spacing is [72,72]
        x_spacing = 1.25 * image_sizes[self.scale_level] / 64

        # Utilizing networkx and graphviz to get the node positioning
        g = nx.DiGraph()
        nodes = [n for n in self.nodes.values()]
        nodes.sort(key=lambda x: x.dfs_id)
        for node in nodes:
            g.add_node(node.id, column=node.column)
        for node in nodes:
            for child in node.children:
                g.add_edge(node.id, self.nodes[child].id)
        positions = nx.nx_pydot.graphviz_layout(g, prog=os.getcwd() + '/graphviz/dot.exe')

        positions = {node: (-y * y_spacing, -x * x_spacing) for (node, (x, y)) in positions.items()}
        positions = {node: (x - positions[0][0], y - positions[0][1]) for (node, (x, y)) in positions.items()}

        if collapse:
            child_ids = self.get_all_children(collapse_id, [])
            if child_ids:
                for child_id in child_ids:
                    positions[child_id] = positions[collapse_id]

        '''
        if (no_children_above % 2) == no_children_below % 2:
            no_children_above += 1
        else:
            no_children_above += 0.5
        '''

        for i in range(animation_len, -1, -1):
            st = time.time()
            graph.erase()
            for node in self.nodes.values():
                base_pos = np.array(positions[node.id])
                diff = (base_pos - node.old_position) / animation_len
                pos = (base_pos - i * diff) + self.offset

                # Draw node frame
                node.frame_id = graph.draw_rectangle(pos, pos + node_size, line_width=2, fill_color='#dddddd')

                # Draw text
                text = unify_art_name_len(node.name, max_name_len)
                node.text_id = graph.draw_text(text, font=(font, font_size), location=tuple(pos + text_offset), text_location=sg.TEXT_LOCATION_LEFT)

                # Draw artist icons
                img = node.images[self.scale_level]
                image_pos = pos + [image_pads[self.scale_level], image_pads[self.scale_level] + image_sizes[self.scale_level]]
                node.img_id = graph.draw_image(data=img, location=tuple(image_pos))

                # Draw connectors from parent to children
                if not node.children:
                    continue

                no_children_above = len([child_id for child_id in node.children if positions[child_id][1] > positions[node.id][1]])
                no_children_below = len([child_id for child_id in node.children if positions[child_id][1] < positions[node.id][1]])
                if no_children_above + no_children_below == len(node.children):
                    no_children_above -= 0.5

                for x, child_id in enumerate(node.children, start=1):
                    child_base_pos = np.array(positions[child_id])
                    child_diff = (child_base_pos - self.nodes[child_id].old_position) / animation_len

                    line_start = pos + connector_parent_offset
                    line_end = (child_base_pos - i * child_diff) + self.offset + connector_offset

                    if line_start[0] > line_end[0]:
                        continue

                    z = abs(x - no_children_above - 1)
                    half_pos_offset = 3 / 4 - z / len(node.children) * 0.66
                    x_offset_left = (line_end[0] - line_start[0]) * half_pos_offset
                    x_offset_right = line_end[0] - line_start[0] - x_offset_left
                    y_offset = (line_end[1] - line_start[1]) / 2

                    thr = 4
                    if abs(y_offset) > thr:
                        if y_offset > 0:
                            node.connector_id_p1 = graph.draw_arc((line_start[0] - x_offset_left, line_end[1]), (line_start[0] + x_offset_left, line_start[1]), 91, 270, line_width=1, arc_color='black', style='arc')
                            node.connector_id_p2 = graph.draw_arc((line_end[0] - x_offset_right, line_end[1]), (line_end[0] + x_offset_right, line_start[1]), 91, 90, line_width=1, arc_color='black', style='arc')
                        else:
                            node.connector_id_p1 = graph.draw_arc((line_start[0] - x_offset_left, line_start[1]), (line_start[0] + x_offset_left, line_end[1]), 91, 0, line_width=1, arc_color='black', style='arc')
                            node.connector_id_p2 = graph.draw_arc((line_end[0] - x_offset_right, line_start[1]), (line_end[0] + x_offset_right, line_end[1]), 91, 180, line_width=1, arc_color='black', style='arc')
                    else:
                        node.connector_id_p1 = graph.draw_line(tuple(line_start), (line_start[0] + x_offset_left, line_start[1]), width=1)
                        node.connector_id_p1 = graph.draw_line((line_start[0] + x_offset_left, line_start[1]), (line_start[0] + x_offset_left, line_end[1]), width=1)
                        node.connector_id_p1 = graph.draw_line((line_start[0] + x_offset_left, line_end[1]), (line_end[0], line_end[1]), width=1)

                graph.bring_figure_to_front(node.frame_id)
                graph.bring_figure_to_front(node.img_id)
                graph.bring_figure_to_front(node.text_id)

            active_diff = (np.array(positions[self.active_node_id]) - self.nodes[self.active_node_id].old_position) / animation_len
            self.offset -= active_diff
            graph.move(-active_diff[0], -active_diff[1])
            window.refresh()
            sleep_time = (time.time() - st)
            sleep_time = animation_len_s / animation_len - sleep_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        for node in self.nodes.values():
            node.old_position = np.copy(np.array(positions[node.id]))
            node.active = False

    def dfs(self):
        for node in self.nodes.values():
            node.dfs_id = -1

        def VisitNode(id, no):
            if self.nodes[id].dfs_id == -1:
                self.nodes[id].dfs_id = no
                no += 1

            for child_id in self.nodes[id].children:
                no = VisitNode(child_id, no)

            return no
        VisitNode(self.start_node.id, 0)

    def print(self):
        print('\n\n')
        print(len(self.nodes), self.nodes)
        for node in self.nodes.values():
            if node is None:
                return
            print(node.id, node.name, f'col: {node.column},', f'parent: {node.parent}')
            if node.children:
                for child_id in node.children:
                    print(f'    {self.nodes[child_id].id} {self.nodes[child_id].name}')
            #node.print()

    def clear(self, graph):
        for node in self.nodes.values():
            graph.delete_figure(node.frame_id)
            graph.delete_figure(node.text_id)
            graph.delete_figure(node.connector_id)
            del node

    def get_all_children(self, id, child_ids):
        if self.nodes[id].children:
            for child_id in self.nodes[id].children:
                child_ids.append(child_id)
                child_ids = self.get_all_children(child_id, child_ids)
        return child_ids

    def collapse_node(self, id):
        if self.nodes[id].children:
            child_ids = self.nodes[id].children.copy()
            for child_id in child_ids:
                self.collapse_node(child_id)
            self.nodes[id].children = []
            self.nodes[id].expanded = False
        else:
            del self.nodes[id]

    def collapse_node_v2(self, id):
        child_ids = self.get_all_children(id, [])
        self.nodes[id].children = []
        self.nodes[id].expanded = False
        self.nodes[id].active = True
        self.active_node_id = id
        for child_id in child_ids:
            self.nodes[child_id].children = []
            self.nodes[child_id].expanded = False
            if child_id == id:
                continue
            del self.nodes[child_id]


class SearchGraph(Graph):
    def __init__(self, spotify_response, max_size, offset=None):
        super().__init__(offset)
        if offset is None:
            offset = [0, 0]
        self.nodes = {}
        self.width = 1
        self.scale = 1
        self.offset = np.array(offset, dtype=float)
        self.add_nodes_from_spotify_response(spotify_response, max_size=max_size)

    def add_nodes_from_spotify_response(self, spotify_response, max_size=10, **kwargs):
        if not os.path.exists("assets/temp/"):
            os.makedirs("assets/temp/")

        for id, artist in enumerate(spotify_response):
            node_name = unify_art_name_len(artist['name'], max_name_len)

            node_images = make_artist_icons(artist['images'])
            node_genres = artist['genres']
            node_url = artist['external_urls']['spotify']
            new_node = Node(id, node_name, 0, spotify_id=artist['id'], images=node_images,
                            url=node_url, genres=node_genres)

            self.nodes[id] = new_node
            id +=+ 1
            if len(self.nodes) == max_size:
                break

    def draw(self, graph, window, center=False):
        node_size = node_sizes[self.scale_level]
        text_offset = np.array([image_pads[self.scale_level] * 2 + image_sizes[self.scale_level], node_size[1] / 2])
        font_size = font_sizes[self.scale_level]
        y_spacing = 5 * image_sizes[self.scale_level] / 64  # default dot.exe spacing is [72,72]
        x_spacing = 1.75 * image_sizes[self.scale_level] / 64

        # Utilizing networkx and graphviz to get the node positioning
        g = nx.DiGraph()
        nodes = [n for n in self.nodes.values()]
        nodes.sort(key=lambda x: x.dfs_id)
        for node in nodes:
            g.add_node(node.id, column=node.column)
        for node in nodes:
            for child in node.children:
                g.add_edge(node.id, self.nodes[child].id)
        positions = nx.nx_pydot.graphviz_layout(g, prog=os.getcwd() + '/graphviz/dot.exe')

        positions = {node: (-y * y_spacing, -x * x_spacing) for (node, (x, y)) in positions.items()}
        positions = {node: (x - positions[0][0], y - positions[0][1]) for (node, (x, y)) in positions.items()}

        if center:
            y_poitions = [positions[node.id][1] for node in self.nodes.values()]
            half_width = (max(y_poitions) - min(y_poitions) + node_sizes[self.scale_level][1]) / 2
            self.offset = [100, 445 + half_width - node_sizes[self.scale_level][1]]

        graph.erase()
        for node in self.nodes.values():
            base_pos = np.array(positions[node.id])
            pos = base_pos + self.offset

            # Draw node frame
            node.frame_id = graph.draw_rectangle(pos, pos + node_size, line_width=1, fill_color='#dddddd')

            # Draw text
            text = unify_art_name_len(node.name, max_name_len)
            node.text_id = graph.draw_text(text, font=(font, font_size), location=tuple(pos + text_offset), text_location=sg.TEXT_LOCATION_LEFT)

            # Draw artist icons
            img = node.images[self.scale_level]
            image_pos = pos + [image_pads[self.scale_level], image_pads[self.scale_level] + image_sizes[self.scale_level]]
            node.img_id = graph.draw_image(data=img, location=tuple(image_pos))

        #window.refresh()
        for node in self.nodes.values():
            node.old_position = np.copy(np.array(positions[node.id]))
            node.active = False

    def draw_animate(self, graph, window, **kwargs):
        node_size = node_sizes[self.scale_level]
        text_offset = np.array([image_pads[self.scale_level] * 2 + image_sizes[self.scale_level], node_size[1] / 2])
        font_size = font_sizes[self.scale_level]
        y_spacing = 5 * image_sizes[self.scale_level] / 64  # default dot.exe spacing is [72,72]
        x_spacing = 1.75 * image_sizes[self.scale_level] / 64

        # Utilizing networkx and graphviz to get the node positioning
        g = nx.DiGraph()
        nodes = [n for n in self.nodes.values()]
        nodes.sort(key=lambda x: x.dfs_id)
        for node in nodes:
            g.add_node(node.id, column=node.column)
        for node in nodes:
            for child in node.children:
                g.add_edge(node.id, self.nodes[child].id)
        positions = nx.nx_pydot.graphviz_layout(g, prog=os.getcwd() + '/graphviz/dot.exe')

        positions = {node: (-y * y_spacing, -x * x_spacing) for (node, (x, y)) in positions.items()}
        positions = {node: (x - positions[0][0], y - positions[0][1]) for (node, (x, y)) in positions.items()}

        for i in range(animation_len, -1, -1):
            graph.erase()
            for node in self.nodes.values():
                base_pos = np.array(positions[node.id])
                node.old_position = np.array([-100, base_pos[1]])
                diff = (base_pos - node.old_position) / animation_len
                pos = (base_pos - i * diff) + self.offset

                # Draw node frame
                node.frame_id = graph.draw_rectangle(pos, pos + node_size, line_width=1, fill_color='#dddddd')

                # Draw text
                text = unify_art_name_len(node.name, max_name_len)
                node.text_id = graph.draw_text(text, font=(font, font_size), location=tuple(pos + text_offset),
                                               text_location=sg.TEXT_LOCATION_LEFT)

                # Draw artist icons
                img = node.images[self.scale_level]
                image_pos = pos + [image_pads[self.scale_level],
                                   image_pads[self.scale_level] + image_sizes[self.scale_level]]
                node.img_id = graph.draw_image(data=img, location=tuple(image_pos))
            active_diff = (np.array(positions[self.active_node_id]) - self.nodes[self.active_node_id].old_position) / animation_len
            self.offset -= active_diff
            graph.move(-active_diff[0], -active_diff[1])
            window.refresh()
            time.sleep(animation_len_s / animation_len)
        for node in self.nodes.values():
            node.old_position = np.copy(np.array(positions[node.id]))
            node.active = False

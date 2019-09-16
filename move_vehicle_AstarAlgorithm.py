from viewer import *
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import numpy as np


def get_union_polygon(polygon, prev_centroid, current_centroid):
    coords = list(polygon.exterior.coords)

    vector1 = np.subtract(prev_centroid, polygon.centroid.coords[0])
    prev_polygon = Polygon([tuple(np.add(coord, vector1)) for coord in coords])

    vector2 = np.subtract(current_centroid, polygon.centroid.coords[0])
    current_polygon = Polygon([tuple(np.add(coord, vector2)) for coord in coords])

    return cascaded_union([prev_polygon, current_polygon])


def get_move_polygon_centroid(polygon, centroid):
    coords = list(polygon.exterior.coords)
    vector = np.subtract(centroid, polygon.centroid.coords[0])
    move_polygon_coords = [tuple(np.add(coord, vector)) for coord in coords]
    return Polygon(move_polygon_coords)


class Obstacle:
    def __init__(self, polygon):
        self.polygon = polygon

    def get_view_object(self):
        return ViewObject(list(self.polygon.exterior.coords))


class LoadableArea:
    def __init__(self, polygon):
        self.polygon = polygon

    def get_view_object(self):
        return ViewObject(list(self.polygon.exterior.coords))


class Node:
    """A node class for A* Pathfinding"""
    def __init__(self, direction, parent=None, position=None):
        self.direction = direction
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position and self.direction == other.direction


class PathManager:
    def __init__(self, loadable_area_list, obstacle_list):
        self.loadable_area_list = loadable_area_list
        self.obstacle_list = obstacle_list
        self.interval = 1

    def is_valid(self, polygon):
        # Make sure polygon is contained by loadable_area
        for loadable_area in self.loadable_area_list:
            if not loadable_area.polygon.contains(polygon):
                return False
        # Make sure polygon do not intersect with obstacle
        for obstacle in self.obstacle_list:
            if obstacle.polygon.intersects(polygon):
                return False
        return True

    def get_path_finding(self, car, s_point, d_point, s_direction, d_direction, interval=1.0):
        """
        시작점과 도착점 정보를 받아, loadable area 안에서 장애물을 회피하여 도착점까지 갈 수 있는 경로를 반환함
        A* algorithm 사용
        :param car:
        :param s_point: 시작점
        :param s_direction: 시작방향
        :param d_point: 도착점
        :param d_direction: 도착방향
        :return: [bezier_pts_list, beizer_pts에 대한 가중치 list, 총 가중치]
        """
        # Create start and end node
        self.interval = interval
        s_direction = tuple(np.multiply((interval, interval), s_direction))
        start_node = Node(s_direction, None, s_point)
        start_node.g = start_node.h = start_node.f = 0
        d_direction = tuple(np.multiply((interval, interval), d_direction))
        end_node = Node(d_direction, None, d_point)
        end_node.g = end_node.h = end_node.f = 0

        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Add the start node
        open_list.append(start_node)

        # Loop until you find the end
        while len(open_list) > 0:
            # Get the current node (lowest F)
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # print(current_node.position)
            # Pop current off open list, add to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)

            # Found the goal
            if current_node == end_node:
                prev = current_node
                current = current_node
                total = 0
                sequence = 0
                time = []
                node_list = [current]
                while current is not None:
                    total += interval
                    if prev.direction == current.direction:
                        sequence += 1
                    else:
                        node_list.append(current)
                        time.append(sequence*interval)
                        sequence = 1
                    prev = current
                    current = current.parent
                node_list.append(prev)
                time.append(sequence*interval)
                return [node_list[::-1], time[::-1], total]

            # Generate children
            directions = [current_node.direction]
            if current_node.direction[0] * current_node.direction[1]:
                directions.append((current_node.direction[0]-current_node.direction[0], current_node.direction[1]))
                directions.append((current_node.direction[0], current_node.direction[1]-current_node.direction[1]))
            else:
                swapped_direction = (current_node.direction[1], current_node.direction[0])
                directions.append(tuple(np.add(current_node.direction, swapped_direction)))
                directions.append(tuple(np.subtract(current_node.direction, swapped_direction)))

            children = []
            for new_direction in directions:
                new_position = tuple(np.add(current_node.position, new_direction))
                # print(new_position)
                move_polygon = get_union_polygon(car.polygon, current_node.position, new_position)
                if not self.is_valid(move_polygon):
                    # print(move_polygon)
                    continue
                new_node = Node(direction=new_direction, parent=current_node, position=new_position)
                children.append(new_node)

            for child in children:
                # Child is on the closed list => ignore
                in_closed_list = False
                for closed_child in closed_list:
                    if child == closed_child:
                        in_closed_list = True
                        break
                if in_closed_list:
                    continue

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h

                # Child is already in the open list
                in_open_list = False
                for open_node in open_list:
                    if child == open_node and child.g >= open_node.g:
                        in_open_list = True
                        break
                if in_open_list:
                    continue

                # Add the child to the open list
                open_list.append(child)

        return [[], [], 0]  # no way

    def get_path_planning(self, car, node_list, weight, total, s_time, d_time):
        car_view_object = car.get_view_object()
        if total == 0:
            return car_view_object
        period = (d_time - s_time) / total
        last_node_index = len(node_list) - 1
        bezier_pts = [node_list[0].position]
        interval = self.interval
        for i in range(1, last_node_index - 1):
            time = s_time
            if weight[i - 1] > interval:
                point = tuple(np.subtract(node_list[i].position, node_list[i].direction))
                bezier_pts.append(point)
                weight[i - 1] -= interval
                # print(bezier_pts, 'time', weight[i - 1])
                car_view_object.set_move(bezier_pts=bezier_pts, start_time=time,
                                         finish_time=time + period * weight[i - 1])
                time = time + period * weight[i - 1]
                bezier_pts = [point, node_list[i].position]

                if weight[i] > interval:
                    weight[i] -= interval
                    point = tuple(np.add(node_list[i].position, node_list[i + 1].direction))
                    bezier_pts.append(point)
                    # print(bezier_pts, 'time', 2 * interval)
                    car_view_object.set_move(bezier_pts=bezier_pts, start_time=time,
                                             finish_time=time + period * 2 * interval)
                    time = time + period * 2 * interval
                    bezier_pts = [point]
            else:
                bezier_pts.append(node_list[i].position)
                if weight[i] > interval:
                    point = tuple(np.add(node_list[i].position, node_list[i + 1].direction))
                    weight[i] -= interval
                    bezier_pts.append(point)
                    # print(bezier_pts, 'time', (len(bezier_pts) - 1) * interval)
                    car_view_object.set_move(bezier_pts=bezier_pts, start_time=time,
                                             finish_time=time + period * (len(bezier_pts) - 1) * interval)
                    time = time + period * (len(bezier_pts) - 1) * interval
                    bezier_pts = [point]
        bezier_pts.append(node_list[last_node_index].position)
        if len(bezier_pts) == 2:
            # print(bezier_pts, 'time', weight[i - 1])
            car_view_object.set_move(bezier_pts=bezier_pts, start_time=time, finish_time=time + period * weight[i - 1])
        else:
            # print(bezier_pts, 'time', (len(bezier_pts) - 1) * interval)
            car_view_object.set_move(bezier_pts=bezier_pts, start_time=time,
                                     finish_time=time + period * (len(bezier_pts) - 1) * interval)
        return car_view_object


class Car:
    def __init__(self, polygon, direction, radius, path_manager, color=(255, 255, 255, 0), is_filled=True):
        self.polygon = polygon
        self.direction = direction
        self.radius = radius
        self.path_manager = path_manager
        self.color = color
        self.is_filled = is_filled

    def get_view_object(self):
        return ViewObject(list(self.polygon.exterior.coords), color=self.color, is_filled=self.is_filled)

    def get_path(self, vector, d_direction):
        return self.path_manager.get_path(car=self, vector=vector, d_direction=d_direction)

    def get_path_finding(self, s_point, d_point, s_direction, d_direction, interval=1.0):
        return self.path_manager.get_path_finding(car=self, s_point=s_point, d_point=d_point, s_direction=s_direction,
                                           d_direction=d_direction, interval=interval)

    def get_path_planning(self, node_list, weight, total, s_time, d_time):
        return self.path_manager.get_path_planning(self, node_list, weight, total, s_time, d_time)


if __name__ == "__main__":
    window_width = 1280
    window_height = 720

    obstacle_list = []
    obstacle1 = Obstacle(polygon=Polygon([[0, 8], [6, 8], [6, 11], [0, 11]]))
    obstacle_list.append(obstacle1)
    obstacle2 = Obstacle(polygon=Polygon([[3, 3], [8, 3], [8, 6], [3, 6]]))
    # obstacle_list.append(obstacle2)
    obstacle3 = Obstacle(polygon=Polygon([[15, 1], [17, 1], [17, 16], [15, 16]]))
    obstacle_list.append(obstacle3)
    obstacle3 = Obstacle(polygon=Polygon([[20, 15], [24, 15], [24, 17], [20, 17]]))
    obstacle_list.append(obstacle3)
    obstacle4 = Obstacle(polygon=Polygon([[-5, 10], [-3, 10], [-3, 15], [-5, 15]]))
    obstacle_list.append(obstacle4)

    # obstacle5 = Obstacle(polygon=Polygon([[-10, 0], [-1, 0], [-1, 1], [-10, 1]]))
    # obstacle_list.append(obstacle5)
    # obstacle6 = Obstacle(polygon=Polygon([[3, 0], [5, 0], [5, 1], [3, 1]]))
    # obstacle_list.append(obstacle6)

    loadable_area_list = []
    loadable_area1 = LoadableArea(polygon=Polygon([[-50, -100], [50, -100], [50, 100], [-50, 100]]))
    loadable_area_list.append(loadable_area1)

    path_manager = PathManager(obstacle_list=obstacle_list, loadable_area_list=loadable_area_list)

    # //[0, 0], [4, 0], [4, 2], [0, 2]
    car = Car(polygon=Polygon([[0, 0], [4, 0], [4, 2], [0, 2]]), direction=(1, 0), radius=2, path_manager=path_manager, color=(255, 111, 111, 0))

    path_finding = car.get_path_finding(s_point=(2, 1), d_point=(2, 1), s_direction=(1, 0), d_direction=(-1, 0), interval=1)

    view_object_list = []
    if path_finding[0]:
        car_view_object = car.get_path_planning(path_finding[0], path_finding[1], path_finding[2], 3, 8)
        view_object_list.append(car_view_object)

    obstacle_list = [obstacle.get_view_object() for obstacle in obstacle_list]
    loadable_area_list = [loadable_area.get_view_object() for loadable_area in loadable_area_list]

    viewer = Viewer(window=pyglet.window.Window(window_width, window_height),
                    object_list=view_object_list + obstacle_list + loadable_area_list, color=(0, 0, 0))
    viewer.init_viewer()
    pyglet.app.run()

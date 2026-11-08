"""The small street graph shared by the #20 and #31 tests.

Lawnmarket (two-way, motor_vehicle=no), Castlehill, Bank Street (motor_vehicle=yes beats
access=no), Upper Bow (one-way 3 -> 5) and Victoria Street (pedestrian).
"""

from engine.orchid import street_network as sn


def build_graph() -> sn.RoadGraph:
    g = sn.RoadGraph()
    for nid, lon in enumerate([-3.2010, -3.2000, -3.1990, -3.1980, -3.1970]):
        g.add_node(nid, lat=55.9490, lon=lon)
    g.add_node(5, lat=55.9500, lon=-3.1980)
    g.add_node(6, lat=55.9510, lon=-3.1980)
    no_cars = {"way_id": 1, "motor_vehicle": "no"}
    g.add_edge(1, 2, oneway=False, street_name="Lawnmarket", osm=no_cars)                 # 0, 1
    g.add_edge(2, 3, oneway=False, street_name="Lawnmarket", osm=no_cars)                 # 2, 3
    g.add_edge(0, 1, oneway=False, street_name="Castlehill", osm={"way_id": 2})           # 4, 5
    g.add_edge(3, 4, oneway=False, street_name="Bank Street",
               osm={"way_id": 3, "motor_vehicle": "yes", "access": "no"})                 # 6, 7
    g.add_edge(3, 5, street_name="Upper Bow", osm={"way_id": 4, "oneway": "yes"})         # 8
    g.add_edge(5, 6, oneway=False, street_name="Victoria Street", highway_class="pedestrian",
               osm={"way_id": 5})                                                         # 9, 10
    return g

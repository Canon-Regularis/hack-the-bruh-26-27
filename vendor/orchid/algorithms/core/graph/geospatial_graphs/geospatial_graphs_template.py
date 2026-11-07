"""
algorithms/core/graph/geospatial_graphs/geospatial_graphs_template.py

Pure-Python, zero-dependency geospatial-graph toolkit.

A self-contained, competition-ready library for graphs embedded on the Earth:
geographic distance (great-circle / ellipsoidal), a KD-tree spatial index over
earth-centred coordinates (so Euclidean nearest == great-circle nearest),
geohashing, proximity-graph construction (k-NN / radius / Gabriel / relative-
neighbourhood), shortest-path routing with an admissible great-circle A*
heuristic, isochrones, computational geometry (convex hull, point-in-polygon,
spherical polygon area, Douglas-Peucker simplification), density clustering
(DBSCAN with a metric eps), Web-Mercator projection, and GeoJSON I/O.  No
geopy / shapely / pyproj / scipy / numpy required.

Coordinates & distance
----------------------
  GeoPoint                   - validated (lat, lon) on Earth
  haversine / law_of_cosines / equirectangular / vincenty / geodesic
  bearing / final_bearing / destination / midpoint
  cross_track_distance / along_track_distance

Projection & encoding
---------------------
  to_web_mercator / from_web_mercator
  geohash_encode / geohash_decode / geohash_bbox / geohash_neighbors

Spatial index
-------------
  GeoIndex                   - nearest / k_nearest / within_radius / bbox query
                               (KD-tree over unit-sphere ECEF coordinates)

Graph
-----
  GeoGraph                   - nodes at coordinates, geodesic-weighted edges
                               build: knn_graph / radius_graph / gabriel_graph /
                               relative_neighborhood_graph / complete_graph
                               route: dijkstra / shortest_path / astar /
                               reachable_within (isochrone)
                               tour:  nearest_neighbor_tour / two_opt

Geometry
--------
  bounding_box / convex_hull / point_in_polygon / spherical_polygon_area /
  polygon_perimeter / centroid / douglas_peucker

Clustering
----------
  dbscan                     - density clustering with eps in metres (haversine)

I/O / payloads
--------------
  to_geojson / build_geo_payload / build_a2ui_payload

Run as a script for an end-to-end demo:
  python geospatial_graphs_template.py
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import (
    Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Set,
    Tuple, Union,
)

__version__ = "1.0.0"

__all__ = [
    "GeoPoint",
    "EARTH_RADIUS_M", "WGS84_A", "WGS84_B", "WGS84_F",
    "haversine", "law_of_cosines", "equirectangular", "vincenty", "geodesic",
    "bearing", "final_bearing", "destination", "midpoint",
    "cross_track_distance", "along_track_distance",
    "to_web_mercator", "from_web_mercator",
    "geohash_encode", "geohash_decode", "geohash_bbox", "geohash_neighbors",
    "GeoIndex", "GeoGraph",
    "bounding_box", "convex_hull", "point_in_polygon",
    "spherical_polygon_area", "polygon_perimeter", "centroid", "douglas_peucker",
    "dbscan",
    "to_geojson", "build_geo_payload", "build_a2ui_payload",
]

NodeId = Hashable

# Mean Earth radius (IUGG) in metres, and WGS84 ellipsoid parameters.
EARTH_RADIUS_M = 6371008.8
WGS84_A = 6378137.0                       # semi-major axis (m)
WGS84_F = 1.0 / 298.257223563             # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)       # semi-minor axis (m)

_EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# GeoPoint
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GeoPoint:
    """A validated geographic position (decimal degrees).

    ``lat`` in [-90, 90], ``lon`` in [-180, 180].  Longitudes are normalised
    into [-180, 180] on construction (so 200 -> -160) but out-of-range
    latitudes are rejected (they are almost always a lat/lon swap)."""
    lat: float
    lon: float

    def __post_init__(self) -> None:
        lat = float(self.lat)
        lon = float(self.lon)
        if not math.isfinite(lat) or not math.isfinite(lon):
            raise ValueError("lat/lon must be finite, got (%r, %r)"
                             % (self.lat, self.lon))
        if not -90.0 <= lat <= 90.0:
            raise ValueError(
                "latitude %r out of range [-90, 90] (lat/lon swapped?)" % lat)
        if not -180.0 <= lon <= 180.0:
            # wrap into [-180, 180]
            lon = (lon + 180.0) % 360.0 - 180.0
        object.__setattr__(self, "lat", lat)
        object.__setattr__(self, "lon", lon)

    def __iter__(self):
        yield self.lat
        yield self.lon

    @cached_property
    def _rad(self) -> Tuple[float, float]:
        return (math.radians(self.lat), math.radians(self.lon))

    @cached_property
    def ecef_unit(self) -> Tuple[float, float, float]:
        """Position on the unit sphere (x, y, z).  Euclidean distance between
        two such points is monotone in their great-circle distance, so a KD-tree
        over these coordinates answers great-circle nearest-neighbour queries
        exactly."""
        rlat, rlon = self._rad
        cl = math.cos(rlat)
        return (cl * math.cos(rlon), cl * math.sin(rlon), math.sin(rlat))


def _as_point(p: Union[GeoPoint, Sequence[float]]) -> GeoPoint:
    return p if isinstance(p, GeoPoint) else GeoPoint(p[0], p[1])


# ─────────────────────────────────────────────────────────────────────────────
# Distance & navigation
# ─────────────────────────────────────────────────────────────────────────────

def haversine(a: Union[GeoPoint, Sequence[float]],
              b: Union[GeoPoint, Sequence[float]],
              radius: float = EARTH_RADIUS_M) -> float:
    """Great-circle distance (metres) via the haversine formula.  Numerically
    stable for small distances and correct across the antimeridian."""
    a, b = _as_point(a), _as_point(b)
    rlat1, rlon1 = a._rad
    rlat2, rlon2 = b._rad
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    h = (math.sin(dlat * 0.5) ** 2
         + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon * 0.5) ** 2)
    return 2.0 * radius * math.asin(min(1.0, math.sqrt(h)))


def law_of_cosines(a: Union[GeoPoint, Sequence[float]],
                   b: Union[GeoPoint, Sequence[float]],
                   radius: float = EARTH_RADIUS_M) -> float:
    """Great-circle distance via the spherical law of cosines (clamped)."""
    a, b = _as_point(a), _as_point(b)
    rlat1, rlon1 = a._rad
    rlat2, rlon2 = b._rad
    c = (math.sin(rlat1) * math.sin(rlat2)
         + math.cos(rlat1) * math.cos(rlat2) * math.cos(rlon2 - rlon1))
    return radius * math.acos(max(-1.0, min(1.0, c)))


def equirectangular(a: Union[GeoPoint, Sequence[float]],
                    b: Union[GeoPoint, Sequence[float]],
                    radius: float = EARTH_RADIUS_M) -> float:
    """Fast equirectangular approximation (good for short distances)."""
    a, b = _as_point(a), _as_point(b)
    rlat1, rlon1 = a._rad
    rlat2, rlon2 = b._rad
    # Wrap the longitude difference into [-pi, pi] so a short hop across the
    # antimeridian is not mistaken for a ~360-degree journey the long way round.
    dlon = (rlon2 - rlon1 + math.pi) % (2 * math.pi) - math.pi
    x = dlon * math.cos((rlat1 + rlat2) * 0.5)
    y = rlat2 - rlat1
    return radius * math.hypot(x, y)


def vincenty(a: Union[GeoPoint, Sequence[float]],
             b: Union[GeoPoint, Sequence[float]],
             *, max_iter: int = 200, tol: float = 1e-12) -> float:
    """Vincenty inverse: geodesic distance (metres) on the WGS84 ellipsoid -
    accurate to ~0.5 mm.  Falls back to :func:`haversine` for near-antipodal
    points where the iteration does not converge."""
    a, b = _as_point(a), _as_point(b)
    if a.lat == b.lat and a.lon == b.lon:
        return 0.0
    rlat1, rlon1 = a._rad
    rlat2, rlon2 = b._rad
    L = rlon2 - rlon1
    U1 = math.atan((1 - WGS84_F) * math.tan(rlat1))
    U2 = math.atan((1 - WGS84_F) * math.tan(rlat2))
    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)
    lam = L
    for _ in range(max_iter):
        sinLam, cosLam = math.sin(lam), math.cos(lam)
        sinSigma = math.sqrt((cosU2 * sinLam) ** 2
                             + (cosU1 * sinU2 - sinU1 * cosU2 * cosLam) ** 2)
        if sinSigma == 0:
            return 0.0
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLam
        sigma = math.atan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * sinLam / sinSigma
        cos2Alpha = 1 - sinAlpha ** 2
        cos2SigmaM = (cosSigma - 2 * sinU1 * sinU2 / cos2Alpha
                      if cos2Alpha != 0 else 0.0)
        C = WGS84_F / 16 * cos2Alpha * (4 + WGS84_F * (4 - 3 * cos2Alpha))
        lam_prev = lam
        lam = L + (1 - C) * WGS84_F * sinAlpha * (
            sigma + C * sinSigma * (cos2SigmaM + C * cosSigma
                                    * (-1 + 2 * cos2SigmaM ** 2)))
        if abs(lam - lam_prev) < tol:
            u2 = cos2Alpha * (WGS84_A ** 2 - WGS84_B ** 2) / WGS84_B ** 2
            A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
            B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
            dSigma = B * sinSigma * (cos2SigmaM + B / 4 * (
                cosSigma * (-1 + 2 * cos2SigmaM ** 2)
                - B / 6 * cos2SigmaM * (-3 + 4 * sinSigma ** 2)
                * (-3 + 4 * cos2SigmaM ** 2)))
            return WGS84_B * A * (sigma - dSigma)
    return haversine(a, b)        # antipodal: did not converge


def geodesic(a: Union[GeoPoint, Sequence[float]],
             b: Union[GeoPoint, Sequence[float]],
             method: str = "haversine") -> float:
    """Distance dispatcher: ``"haversine"`` (default), ``"vincenty"``,
    ``"cosines"`` or ``"equirectangular"``."""
    if method == "haversine":
        return haversine(a, b)
    if method == "vincenty":
        return vincenty(a, b)
    if method == "cosines":
        return law_of_cosines(a, b)
    if method == "equirectangular":
        return equirectangular(a, b)
    raise ValueError("unknown distance method %r" % (method,))


def bearing(a: Union[GeoPoint, Sequence[float]],
            b: Union[GeoPoint, Sequence[float]]) -> float:
    """Initial bearing (degrees, 0=N, clockwise) from ``a`` to ``b``."""
    a, b = _as_point(a), _as_point(b)
    rlat1, rlon1 = a._rad
    rlat2, rlon2 = b._rad
    dlon = rlon2 - rlon1
    y = math.sin(dlon) * math.cos(rlat2)
    x = (math.cos(rlat1) * math.sin(rlat2)
         - math.sin(rlat1) * math.cos(rlat2) * math.cos(dlon))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def final_bearing(a: Union[GeoPoint, Sequence[float]],
                  b: Union[GeoPoint, Sequence[float]]) -> float:
    """Bearing (degrees) at which the geodesic arrives at ``b``."""
    return (bearing(b, a) + 180.0) % 360.0


def destination(a: Union[GeoPoint, Sequence[float]], bearing_deg: float,
                distance_m: float, radius: float = EARTH_RADIUS_M) -> GeoPoint:
    """The point reached by travelling ``distance_m`` from ``a`` along the
    great circle with initial bearing ``bearing_deg``."""
    a = _as_point(a)
    rlat1, rlon1 = a._rad
    theta = math.radians(bearing_deg)
    delta = distance_m / radius
    sin_lat2 = (math.sin(rlat1) * math.cos(delta)
                + math.cos(rlat1) * math.sin(delta) * math.cos(theta))
    lat2 = math.asin(max(-1.0, min(1.0, sin_lat2)))
    y = math.sin(theta) * math.sin(delta) * math.cos(rlat1)
    x = math.cos(delta) - math.sin(rlat1) * sin_lat2
    lon2 = rlon1 + math.atan2(y, x)
    return GeoPoint(math.degrees(lat2), math.degrees(lon2))


def midpoint(a: Union[GeoPoint, Sequence[float]],
             b: Union[GeoPoint, Sequence[float]]) -> GeoPoint:
    """Great-circle midpoint of ``a`` and ``b``."""
    a, b = _as_point(a), _as_point(b)
    rlat1, rlon1 = a._rad
    rlat2, rlon2 = b._rad
    dlon = rlon2 - rlon1
    bx = math.cos(rlat2) * math.cos(dlon)
    by = math.cos(rlat2) * math.sin(dlon)
    lat3 = math.atan2(math.sin(rlat1) + math.sin(rlat2),
                      math.sqrt((math.cos(rlat1) + bx) ** 2 + by ** 2))
    lon3 = rlon1 + math.atan2(by, math.cos(rlat1) + bx)
    return GeoPoint(math.degrees(lat3), math.degrees(lon3))


def cross_track_distance(p: Union[GeoPoint, Sequence[float]],
                         path_start: Union[GeoPoint, Sequence[float]],
                         path_end: Union[GeoPoint, Sequence[float]],
                         radius: float = EARTH_RADIUS_M) -> float:
    """Signed distance (metres) of ``p`` from the great circle through
    ``path_start``/``path_end`` (positive = right of the path direction, the
    Aviation Formulary convention).  For a degenerate path (start == end) the
    bearing is undefined, so the plain distance from ``p`` to that point is
    returned."""
    p, s, e = _as_point(p), _as_point(path_start), _as_point(path_end)
    if s.lat == e.lat and s.lon == e.lon:
        return haversine(p, s, radius)
    d13 = law_of_cosines(s, p, radius=1.0)          # angular distance
    theta13 = math.radians(bearing(s, p))
    theta12 = math.radians(bearing(s, e))
    return math.asin(max(-1.0, min(1.0,
                     math.sin(d13) * math.sin(theta13 - theta12)))) * radius


def along_track_distance(p: Union[GeoPoint, Sequence[float]],
                         path_start: Union[GeoPoint, Sequence[float]],
                         path_end: Union[GeoPoint, Sequence[float]],
                         radius: float = EARTH_RADIUS_M) -> float:
    """Signed distance (metres) from ``path_start`` to the projection of ``p``
    onto the great circle through ``path_start``/``path_end``.  It is **negative**
    when the projection falls *before* the start (i.e. ``p`` is "behind" the
    path direction) and exceeds the path length when ``p`` projects beyond the
    end.  A degenerate path (start == end) returns 0."""
    p, s, e = _as_point(p), _as_point(path_start), _as_point(path_end)
    if s.lat == e.lat and s.lon == e.lon:
        return 0.0
    d13 = law_of_cosines(s, p, radius=1.0)
    xt = cross_track_distance(p, s, e, radius=1.0)
    cos_d13 = math.cos(d13)
    cos_xt = math.cos(xt)
    if abs(cos_xt) < _EPS:
        return 0.0
    at = math.acos(max(-1.0, min(1.0, cos_d13 / cos_xt)))
    # acos only gives the magnitude; the projection is behind the start when the
    # bearing to p differs from the path bearing by more than 90 degrees.
    theta12 = math.radians(bearing(s, e))
    theta13 = math.radians(bearing(s, p))
    if math.cos(theta13 - theta12) < 0:
        at = -at
    return at * radius


# ─────────────────────────────────────────────────────────────────────────────
# Projection
# ─────────────────────────────────────────────────────────────────────────────

_MERC_MAX_LAT = 85.05112878            # Web-Mercator latitude clamp


def to_web_mercator(p: Union[GeoPoint, Sequence[float]],
                    radius: float = WGS84_A) -> Tuple[float, float]:
    """Project to Web Mercator (EPSG:3857) metres (x east, y north).  Latitude
    is clamped to +/-85.05 deg (the standard valid range)."""
    p = _as_point(p)
    lat = max(-_MERC_MAX_LAT, min(_MERC_MAX_LAT, p.lat))
    x = radius * math.radians(p.lon)
    y = radius * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return (x, y)


def from_web_mercator(x: float, y: float, radius: float = WGS84_A) -> GeoPoint:
    """Inverse Web-Mercator projection back to (lat, lon)."""
    lon = math.degrees(x / radius)
    lat = math.degrees(2 * math.atan(math.exp(y / radius)) - math.pi / 2)
    return GeoPoint(lat, lon)


# ─────────────────────────────────────────────────────────────────────────────
# Geohash
# ─────────────────────────────────────────────────────────────────────────────

_GEOHASH_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_GEOHASH_DECODE = {c: i for i, c in enumerate(_GEOHASH_B32)}


def geohash_encode(lat: float, lon: float, precision: int = 12) -> str:
    """Encode (lat, lon) as a geohash string of ``precision`` base-32 chars."""
    if precision < 1:
        raise ValueError("precision must be >= 1, got %r" % (precision,))
    p = _as_point(GeoPoint(lat, lon))
    lat, lon = p.lat, p.lon
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    gh: List[str] = []
    bit = 0
    ch = 0
    even = True            # start with longitude
    while len(gh) < precision:
        if even:
            mid = (lon_lo + lon_hi) / 2
            if lon >= mid:
                ch = (ch << 1) | 1
                lon_lo = mid
            else:
                ch = ch << 1
                lon_hi = mid
        else:
            mid = (lat_lo + lat_hi) / 2
            if lat >= mid:
                ch = (ch << 1) | 1
                lat_lo = mid
            else:
                ch = ch << 1
                lat_hi = mid
        even = not even
        bit += 1
        if bit == 5:
            gh.append(_GEOHASH_B32[ch])
            bit = 0
            ch = 0
    return "".join(gh)


def geohash_bbox(gh: str) -> Tuple[float, float, float, float]:
    """Decode a geohash to its bounding box (min_lat, min_lon, max_lat,
    max_lon)."""
    if not gh:
        raise ValueError("empty geohash")
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    even = True
    for c in gh:
        try:
            cd = _GEOHASH_DECODE[c]
        except KeyError:
            raise ValueError("invalid geohash character %r" % (c,))
        for mask in (16, 8, 4, 2, 1):
            if even:
                mid = (lon_lo + lon_hi) / 2
                if cd & mask:
                    lon_lo = mid
                else:
                    lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if cd & mask:
                    lat_lo = mid
                else:
                    lat_hi = mid
            even = not even
    return (lat_lo, lon_lo, lat_hi, lon_hi)


def geohash_decode(gh: str) -> GeoPoint:
    """Decode a geohash to the centre of its cell."""
    lat_lo, lon_lo, lat_hi, lon_hi = geohash_bbox(gh)
    return GeoPoint((lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2)


def geohash_neighbors(gh: str) -> Dict[str, str]:
    """The 8 neighbouring geohashes (keys n, ne, e, se, s, sw, w, nw) at the
    same precision."""
    lat_lo, lon_lo, lat_hi, lon_hi = geohash_bbox(gh)
    lat_c, lon_c = (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2
    dlat, dlon = lat_hi - lat_lo, lon_hi - lon_lo
    prec = len(gh)
    out: Dict[str, str] = {}
    for name, (dla, dlo) in {
        "n": (1, 0), "ne": (1, 1), "e": (0, 1), "se": (-1, 1),
        "s": (-1, 0), "sw": (-1, -1), "w": (0, -1), "nw": (1, -1),
    }.items():
        nlat = max(-90.0, min(90.0, lat_c + dla * dlat))
        nlon = lon_c + dlo * dlon
        out[name] = geohash_encode(nlat, nlon, prec)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# KD-tree + spatial index
# ─────────────────────────────────────────────────────────────────────────────

class _KDNode:
    __slots__ = ("coord", "payload", "axis", "left", "right")

    def __init__(self, coord, payload, axis):
        self.coord = coord
        self.payload = payload
        self.axis = axis
        self.left: Optional["_KDNode"] = None
        self.right: Optional["_KDNode"] = None


class _KDTree:
    """A static k-d tree over Euclidean coordinates with nearest / k-nearest /
    radius queries (median-split build, branch-pruned search)."""

    def __init__(self, items: Sequence[Tuple[Sequence[float], Any]]) -> None:
        items = list(items)
        self.dim = len(items[0][0]) if items else 0
        self.root = self._build(items, 0)

    def _build(self, items, depth):
        if not items:
            return None
        axis = depth % self.dim
        items.sort(key=lambda it: it[0][axis])
        mid = len(items) // 2
        node = _KDNode(items[mid][0], items[mid][1], axis)
        node.left = self._build(items[:mid], depth + 1)
        node.right = self._build(items[mid + 1:], depth + 1)
        return node

    @staticmethod
    def _d2(a, b):
        return sum((a[i] - b[i]) ** 2 for i in range(len(a)))

    def nearest(self, q):
        best = [None, math.inf]      # [payload, dist2]

        def rec(node):
            if node is None:
                return
            d2 = self._d2(q, node.coord)
            if d2 < best[1]:
                best[0], best[1] = node.payload, d2
            ax = node.axis
            diff = q[ax] - node.coord[ax]
            near, far = ((node.left, node.right) if diff < 0
                         else (node.right, node.left))
            rec(near)
            if diff * diff < best[1]:
                rec(far)

        rec(self.root)
        return best[0], best[1]

    def k_nearest(self, q, k):
        heap: List[Tuple[float, int, Any]] = []     # max-heap by -dist2
        counter = 0

        def rec(node):
            nonlocal counter
            if node is None:
                return
            d2 = self._d2(q, node.coord)
            if len(heap) < k:
                heapq.heappush(heap, (-d2, counter, node.payload))
                counter += 1
            elif d2 < -heap[0][0]:
                heapq.heapreplace(heap, (-d2, counter, node.payload))
                counter += 1
            ax = node.axis
            diff = q[ax] - node.coord[ax]
            near, far = ((node.left, node.right) if diff < 0
                         else (node.right, node.left))
            rec(near)
            if len(heap) < k or diff * diff < -heap[0][0]:
                rec(far)

        rec(self.root)
        out = [(-nd2, payload) for nd2, _, payload in heap]
        out.sort(key=lambda t: t[0])
        return out                                  # [(dist2, payload), ...]

    def within_radius(self, q, r):
        r2 = r * r
        found: List[Tuple[float, Any]] = []

        def rec(node):
            if node is None:
                return
            d2 = self._d2(q, node.coord)
            if d2 <= r2:
                found.append((d2, node.payload))
            ax = node.axis
            diff = q[ax] - node.coord[ax]
            near, far = ((node.left, node.right) if diff < 0
                         else (node.right, node.left))
            rec(near)
            if diff * diff <= r2:
                rec(far)

        rec(self.root)
        return found


def _chord(angle: float) -> float:
    """Unit-sphere chord length for a central angle (radians)."""
    return 2.0 * math.sin(min(math.pi, angle) / 2.0)


def _chord_to_metres(chord: float, radius: float) -> float:
    return radius * 2.0 * math.asin(max(0.0, min(1.0, chord / 2.0)))


class GeoIndex:
    """Spatial index over geographic points for nearest / k-nearest / radius /
    bounding-box queries.  Internally a KD-tree over unit-sphere ECEF
    coordinates, so Euclidean nearest is exactly great-circle nearest."""

    def __init__(self, points: Union[Mapping[NodeId, GeoPoint],
                                     Sequence[GeoPoint]],
                 radius: float = EARTH_RADIUS_M) -> None:
        self.radius = radius
        self._points: Dict[NodeId, GeoPoint] = _normalise_points(points)
        items = [(p.ecef_unit, nid) for nid, p in self._points.items()]
        self._tree = _KDTree(items)

    def __len__(self) -> int:
        return len(self._points)

    @property
    def points(self) -> Dict[NodeId, GeoPoint]:
        return dict(self._points)

    def nearest(self, query: Union[GeoPoint, Sequence[float]]
                ) -> Optional[Tuple[NodeId, float]]:
        """Return (node_id, distance_m) of the closest point, or None if empty."""
        if not self._points:
            return None
        q = _as_point(query).ecef_unit
        nid, d2 = self._tree.nearest(q)
        return (nid, _chord_to_metres(math.sqrt(d2), self.radius))

    def k_nearest(self, query: Union[GeoPoint, Sequence[float]], k: int
                  ) -> List[Tuple[NodeId, float]]:
        """The ``k`` closest points as (node_id, distance_m), nearest first."""
        if k < 1:
            raise ValueError("k must be >= 1, got %r" % (k,))
        q = _as_point(query).ecef_unit
        return [(nid, _chord_to_metres(math.sqrt(d2), self.radius))
                for d2, nid in self._tree.k_nearest(q, k)]

    def within_radius(self, query: Union[GeoPoint, Sequence[float]],
                      radius_m: float) -> List[Tuple[NodeId, float]]:
        """All points within ``radius_m`` metres, as (node_id, distance_m),
        nearest first."""
        if radius_m < 0:
            raise ValueError("radius_m must be >= 0, got %r" % (radius_m,))
        q = _as_point(query).ecef_unit
        chord = _chord(radius_m / self.radius)
        out = [(nid, _chord_to_metres(math.sqrt(d2), self.radius))
               for d2, nid in self._tree.within_radius(q, chord)]
        out.sort(key=lambda t: t[1])
        return out

    def within_bbox(self, min_lat: float, min_lon: float,
                    max_lat: float, max_lon: float) -> List[NodeId]:
        """All node ids whose point lies in the lat/lon bounding box.

        Following the GeoJSON convention, ``min_lon > max_lon`` denotes a box
        that CROSSES THE ANTIMERIDIAN (e.g. min_lon=170, max_lon=-170 selects the
        20-degree band around +/-180), rather than an empty box."""
        if min_lon <= max_lon:
            def lon_ok(lon: float) -> bool:
                return min_lon <= lon <= max_lon
        else:                                   # antimeridian-crossing box
            def lon_ok(lon: float) -> bool:
                return lon >= min_lon or lon <= max_lon
        return [nid for nid, p in self._points.items()
                if min_lat <= p.lat <= max_lat and lon_ok(p.lon)]


def _normalise_points(points: Union[Mapping[NodeId, GeoPoint],
                                    Sequence[GeoPoint]]
                      ) -> Dict[NodeId, GeoPoint]:
    if isinstance(points, Mapping):
        return {nid: _as_point(p) for nid, p in points.items()}
    return {i: _as_point(p) for i, p in enumerate(points)}


# ─────────────────────────────────────────────────────────────────────────────
# GeoGraph
# ─────────────────────────────────────────────────────────────────────────────

class GeoGraph:
    """A graph whose nodes carry geographic coordinates and whose edges are
    weighted (by default with the geodesic distance between endpoints)."""

    def __init__(self, *, directed: bool = False,
                 distance: str = "haversine") -> None:
        self.directed = directed
        self.distance = distance
        self._coords: Dict[NodeId, GeoPoint] = {}
        self._adj: Dict[NodeId, Dict[NodeId, float]] = {}

    # -- construction ---------------------------------------------------------

    def add_node(self, node: NodeId, point: Union[GeoPoint, Sequence[float]]
                 ) -> None:
        """Add (or move) a node.  Re-adding an existing id updates its
        coordinate but leaves any incident edge weights unchanged - recompute
        them with :meth:`add_edge` if you moved a connected node."""
        self._coords[node] = _as_point(point)
        self._adj.setdefault(node, {})

    def add_edge(self, u: NodeId, v: NodeId, weight: Optional[float] = None
                 ) -> None:
        if u not in self._coords or v not in self._coords:
            raise KeyError("both endpoints must be added as nodes first")
        if weight is None:
            weight = geodesic(self._coords[u], self._coords[v], self.distance)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("edge weight must be finite and >= 0, got %r"
                             % (weight,))
        self._adj[u][v] = weight
        if not self.directed:
            self._adj[v][u] = weight

    @property
    def nodes(self) -> List[NodeId]:
        return list(self._coords)

    @property
    def num_nodes(self) -> int:
        return len(self._coords)

    @property
    def num_edges(self) -> int:
        m = sum(len(nbrs) for nbrs in self._adj.values())
        return m if self.directed else m // 2

    def coord(self, node: NodeId) -> GeoPoint:
        return self._coords[node]

    def neighbors(self, node: NodeId) -> Dict[NodeId, float]:
        return dict(self._adj.get(node, {}))

    def weight(self, u: NodeId, v: NodeId) -> Optional[float]:
        return self._adj.get(u, {}).get(v)

    def index(self) -> GeoIndex:
        """A :class:`GeoIndex` over this graph's nodes."""
        return GeoIndex(self._coords)

    # -- builders -------------------------------------------------------------

    @classmethod
    def knn_graph(cls, points: Union[Mapping[NodeId, GeoPoint],
                                     Sequence[GeoPoint]], k: int, *,
                  directed: bool = False, mutual: bool = False,
                  distance: str = "haversine") -> "GeoGraph":
        """Connect each node to its ``k`` nearest neighbours.  ``mutual=True``
        keeps an (undirected) edge only when both endpoints choose each other."""
        if k < 1:
            raise ValueError("k must be >= 1, got %r" % (k,))
        pts = _normalise_points(points)
        g = cls(directed=directed, distance=distance)
        for nid, p in pts.items():
            g.add_node(nid, p)
        idx = GeoIndex(pts)
        chosen: Dict[NodeId, Set[NodeId]] = {}
        for nid, p in pts.items():
            nbrs = [o for o, _ in idx.k_nearest(p, min(k + 1, len(pts)))
                    if o != nid][:k]
            chosen[nid] = set(nbrs)
        for nid, nbrs in chosen.items():
            for o in nbrs:
                if mutual and nid not in chosen.get(o, ()):
                    continue
                g.add_edge(nid, o)
        return g

    @classmethod
    def radius_graph(cls, points: Union[Mapping[NodeId, GeoPoint],
                                        Sequence[GeoPoint]], radius_m: float, *,
                     distance: str = "haversine") -> "GeoGraph":
        """Connect every pair of nodes within ``radius_m`` metres (unit-disk
        graph)."""
        if radius_m < 0:
            raise ValueError("radius_m must be >= 0, got %r" % (radius_m,))
        pts = _normalise_points(points)
        g = cls(directed=False, distance=distance)
        for nid, p in pts.items():
            g.add_node(nid, p)
        idx = GeoIndex(pts)
        for nid, p in pts.items():
            for o, d in idx.within_radius(p, radius_m):
                if o != nid and g.weight(nid, o) is None:
                    g.add_edge(nid, o)
        return g

    @classmethod
    def complete_graph(cls, points: Union[Mapping[NodeId, GeoPoint],
                                          Sequence[GeoPoint]], *,
                       distance: str = "haversine") -> "GeoGraph":
        pts = _normalise_points(points)
        g = cls(directed=False, distance=distance)
        for nid, p in pts.items():
            g.add_node(nid, p)
        ids = list(pts)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                g.add_edge(ids[i], ids[j])
        return g

    @classmethod
    def gabriel_graph(cls, points: Union[Mapping[NodeId, GeoPoint],
                                         Sequence[GeoPoint]], *,
                      distance: str = "haversine") -> "GeoGraph":
        """Gabriel graph: edge (p, q) iff no other node lies inside the circle
        having pq as diameter (a planar proximity graph, subgraph of the
        Delaunay triangulation).  O(n^3)."""
        pts = _normalise_points(points)
        g = cls(directed=False, distance=distance)
        for nid, p in pts.items():
            g.add_node(nid, p)
        ids = list(pts)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = pts[ids[i]], pts[ids[j]]
                d_ab = geodesic(a, b, distance)
                mid = midpoint(a, b)
                rad = d_ab / 2
                ok = True
                for kk in range(len(ids)):
                    if kk == i or kk == j:
                        continue
                    if geodesic(mid, pts[ids[kk]], distance) < rad - _EPS:
                        ok = False
                        break
                if ok:
                    g.add_edge(ids[i], ids[j], d_ab)
        return g

    @classmethod
    def relative_neighborhood_graph(
            cls, points: Union[Mapping[NodeId, GeoPoint], Sequence[GeoPoint]],
            *, distance: str = "haversine") -> "GeoGraph":
        """Relative neighbourhood graph: edge (p, q) iff no node r is closer to
        both p and q than they are to each other.  O(n^3)."""
        pts = _normalise_points(points)
        g = cls(directed=False, distance=distance)
        for nid, p in pts.items():
            g.add_node(nid, p)
        ids = list(pts)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = pts[ids[i]], pts[ids[j]]
                d_ab = geodesic(a, b, distance)
                ok = True
                for kk in range(len(ids)):
                    if kk == i or kk == j:
                        continue
                    r = pts[ids[kk]]
                    if (geodesic(a, r, distance) < d_ab - _EPS
                            and geodesic(b, r, distance) < d_ab - _EPS):
                        ok = False
                        break
                if ok:
                    g.add_edge(ids[i], ids[j], d_ab)
        return g

    # -- routing --------------------------------------------------------------

    def dijkstra(self, source: NodeId
                 ) -> Tuple[Dict[NodeId, float], Dict[NodeId, NodeId]]:
        """Single-source shortest distances + predecessor tree."""
        if source not in self._coords:
            raise KeyError("unknown source %r" % (source,))
        dist: Dict[NodeId, float] = {source: 0.0}
        prev: Dict[NodeId, NodeId] = {}
        pq: List[Tuple[float, int, NodeId]] = [(0.0, 0, source)]
        seen: Set[NodeId] = set()
        c = 0
        while pq:
            d, _, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            for v, w in self._adj.get(u, {}).items():
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = u
                    c += 1
                    heapq.heappush(pq, (nd, c, v))
        return dist, prev

    def shortest_path(self, source: NodeId, target: NodeId
                      ) -> Optional[List[NodeId]]:
        """Minimum-distance path (Dijkstra) as a node list, or None."""
        dist, prev = self.dijkstra(source)
        if target not in dist:
            return None
        return _reconstruct(prev, source, target)

    def shortest_path_length(self, source: NodeId, target: NodeId
                             ) -> Optional[float]:
        dist, _ = self.dijkstra(source)
        return dist.get(target)

    def astar(self, source: NodeId, target: NodeId, *,
              heuristic: Optional[Any] = None
              ) -> Optional[List[NodeId]]:
        """A* shortest path.  The default heuristic is the straight-line
        distance to the target *under this graph's own distance metric*, which
        is admissible and consistent whenever edge weights are geodesic
        distances, so A* returns the same optimum as Dijkstra while exploring
        far fewer nodes.  (Using a different metric than the edges - e.g.
        haversine edges scored against vincenty - can over-estimate and make
        A* return a sub-optimal path, so the metric is matched here.)"""
        if source not in self._coords or target not in self._coords:
            raise KeyError("unknown source/target")
        if heuristic is None:
            tcoord = self._coords[target]
            method = self.distance

            def heuristic(n):                       # noqa: E306
                return geodesic(self._coords[n], tcoord, method)
        g_score: Dict[NodeId, float] = {source: 0.0}
        prev: Dict[NodeId, NodeId] = {}
        pq: List[Tuple[float, int, NodeId]] = [(heuristic(source), 0, source)]
        seen: Set[NodeId] = set()
        c = 0
        while pq:
            _, _, u = heapq.heappop(pq)
            if u == target:
                return _reconstruct(prev, source, target)
            if u in seen:
                continue
            seen.add(u)
            for v, w in self._adj.get(u, {}).items():
                ng = g_score[u] + w
                if ng < g_score.get(v, math.inf):
                    g_score[v] = ng
                    prev[v] = u
                    c += 1
                    heapq.heappush(pq, (ng + heuristic(v), c, v))
        return None

    def reachable_within(self, source: NodeId, budget_m: float
                         ) -> Dict[NodeId, float]:
        """Isochrone: all nodes whose shortest-path distance from ``source`` is
        <= ``budget_m`` metres, mapped to that distance."""
        dist, _ = self.dijkstra(source)
        return {n: d for n, d in dist.items() if d <= budget_m}

    def path_length(self, path: Sequence[NodeId]) -> float:
        """Total weight of a node path (uses edge weights; raises if a hop is
        missing)."""
        total = 0.0
        for u, v in zip(path, path[1:]):
            w = self.weight(u, v)
            if w is None:
                raise ValueError("no edge %r -> %r in path" % (u, v))
            total += w
        return total

    # -- tour (TSP heuristics) ------------------------------------------------

    def nearest_neighbor_tour(self, start: Optional[NodeId] = None
                              ) -> List[NodeId]:
        """A greedy nearest-neighbour tour visiting every node once and
        returning to the start (uses straight-line geodesic distance, so it
        works whether or not the graph is fully connected)."""
        ids = list(self._coords)
        if not ids:
            return []
        start = ids[0] if start is None else start
        unvisited = set(ids)
        unvisited.discard(start)
        tour = [start]
        cur = start
        while unvisited:
            nxt = min(unvisited,
                      key=lambda o: geodesic(self._coords[cur],
                                             self._coords[o], self.distance))
            tour.append(nxt)
            unvisited.discard(nxt)
            cur = nxt
        return tour

    def tour_length(self, tour: Sequence[NodeId], *, closed: bool = True
                    ) -> float:
        """Geodesic length of a tour (closed = return to start)."""
        if len(tour) < 2:
            return 0.0
        seq = list(tour) + ([tour[0]] if closed else [])
        return sum(geodesic(self._coords[a], self._coords[b], self.distance)
                   for a, b in zip(seq, seq[1:]))

    def two_opt(self, tour: Sequence[NodeId], *, max_passes: int = 50
                ) -> List[NodeId]:
        """Improve a closed tour with 2-opt edge swaps until no swap shortens
        it (or ``max_passes`` is reached)."""
        route = list(tour)
        n = len(route)
        if n < 4:
            return route

        def d(a, b):
            return geodesic(self._coords[a], self._coords[b], self.distance)

        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue          # adjacent in the closed tour
                    a, b = route[i], route[i + 1]
                    c2, d2 = route[j], route[(j + 1) % n]
                    if d(a, b) + d(c2, d2) > d(a, c2) + d(b, d2) + _EPS:
                        route[i + 1:j + 1] = reversed(route[i + 1:j + 1])
                        improved = True
        return route


def _reconstruct(prev: Mapping[NodeId, NodeId], source: NodeId,
                 target: NodeId) -> List[NodeId]:
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

def bounding_box(points: Iterable[Union[GeoPoint, Sequence[float]]]
                 ) -> Tuple[float, float, float, float]:
    """Axis-aligned bounding box (min_lat, min_lon, max_lat, max_lon)."""
    pts = [_as_point(p) for p in points]
    if not pts:
        raise ValueError("no points")
    lats = [p.lat for p in pts]
    lons = [p.lon for p in pts]
    return (min(lats), min(lons), max(lats), max(lons))


def centroid(points: Iterable[Union[GeoPoint, Sequence[float]]]) -> GeoPoint:
    """Spherical centroid (mean of ECEF unit vectors projected back)."""
    pts = [_as_point(p) for p in points]
    if not pts:
        raise ValueError("no points")
    x = y = z = 0.0
    for p in pts:
        ex, ey, ez = p.ecef_unit
        x += ex
        y += ey
        z += ez
    n = len(pts)
    x, y, z = x / n, y / n, z / n
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < _EPS:
        return GeoPoint(0.0, 0.0)               # antipodal cancellation
    lat = math.degrees(math.asin(z / norm))
    lon = math.degrees(math.atan2(y, x))
    return GeoPoint(lat, lon)


def convex_hull(points: Sequence[Union[GeoPoint, Sequence[float]]]
                ) -> List[GeoPoint]:
    """Convex hull (Andrew's monotone chain) on the (lon, lat) plane, returned
    counter-clockwise without the closing repeat.  Suitable for regions small
    relative to the globe."""
    pts = sorted({(p.lon, p.lat) for p in (_as_point(q) for q in points)})
    if len(pts) <= 2:
        return [GeoPoint(lat, lon) for lon, lat in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return [GeoPoint(lat, lon) for lon, lat in hull]


def point_in_polygon(point: Union[GeoPoint, Sequence[float]],
                     polygon: Sequence[Union[GeoPoint, Sequence[float]]]
                     ) -> bool:
    """Ray-casting point-in-polygon test on the (lon, lat) plane (even-odd
    rule).  The polygon is a ring of vertices (auto-closed); points exactly on
    an edge may return either result."""
    p = _as_point(point)
    ring = [_as_point(q) for q in polygon]
    if len(ring) < 3:
        return False
    x, y = p.lon, p.lat
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i].lon, ring[i].lat
        xj, yj = ring[j].lon, ring[j].lat
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def spherical_polygon_area(polygon: Sequence[Union[GeoPoint, Sequence[float]]],
                           radius: float = EARTH_RADIUS_M, *,
                           max_segment_deg: float = 0.5) -> float:
    """Area (square metres) of a simple polygon on the sphere, via the
    longitude-integral ``A = R^2/2 |sum (lon_{i+1}-lon_i)(sin lat_i +
    sin lat_{i+1})|``.  Vertex order does not matter (absolute value returned).

    That integral is only exact in the small-edge limit (it assumes ``sin lat``
    varies linearly with longitude along each edge), so each edge is first
    densified with great-circle waypoints no more than ``max_segment_deg`` apart.
    This keeps the result accurate (<0.01%) even for coarse polygons with edges
    spanning large latitude ranges, where the un-densified formula can be off by
    tens of percent."""
    ring = [_as_point(q) for q in polygon]
    if len(ring) < 3:
        return 0.0
    step = math.radians(max(1e-6, max_segment_deg))
    # Densify: walk each edge in great-circle steps, collecting (sin lat, lon).
    dense: List[Tuple[float, float]] = []
    n = len(ring)
    for i in range(n):
        a = ring[i]
        b = ring[(i + 1) % n]
        ax, ay, az = a.ecef_unit
        bx, by, bz = b.ecef_unit
        omega = math.acos(max(-1.0, min(1.0, ax * bx + ay * by + az * bz)))
        nseg = max(1, int(math.ceil(omega / step)))
        for s in range(nseg):
            lat, lon = _slerp_latlon(a, b, s / nseg)
            dense.append((math.sin(math.radians(lat)), lon))
    total = 0.0
    m = len(dense)
    for i in range(m):
        sin1, lon1 = dense[i]
        sin2, lon2 = dense[(i + 1) % m]
        # Normalise the longitude step into [-180, 180] so an edge crossing the
        # antimeridian contributes its true (small) span rather than a ~360 jump.
        dlon = (lon2 - lon1 + 180.0) % 360.0 - 180.0
        total += math.radians(dlon) * (2 + sin1 + sin2)
    return abs(total * radius * radius / 2.0)


def _slerp_latlon(a: GeoPoint, b: GeoPoint, f: float) -> Tuple[float, float]:
    """Great-circle interpolation: the point a fraction ``f`` of the way from
    ``a`` to ``b`` along the shorter arc, as (lat, lon) degrees."""
    ax, ay, az = a.ecef_unit
    bx, by, bz = b.ecef_unit
    omega = math.acos(max(-1.0, min(1.0, ax * bx + ay * by + az * bz)))
    if omega < 1e-12:
        return a.lat, a.lon
    so = math.sin(omega)
    s1 = math.sin((1.0 - f) * omega) / so
    s2 = math.sin(f * omega) / so
    x, y, z = ax * s1 + bx * s2, ay * s1 + by * s2, az * s1 + bz * s2
    nr = math.sqrt(x * x + y * y + z * z)
    return math.degrees(math.asin(z / nr)), math.degrees(math.atan2(y, x))


def polygon_perimeter(polygon: Sequence[Union[GeoPoint, Sequence[float]]],
                      *, closed: bool = True, distance: str = "haversine"
                      ) -> float:
    """Geodesic perimeter (metres) of a polygon / polyline."""
    ring = [_as_point(q) for q in polygon]
    if len(ring) < 2:
        return 0.0
    seq = ring + ([ring[0]] if closed else [])
    return sum(geodesic(a, b, distance) for a, b in zip(seq, seq[1:]))


def douglas_peucker(polyline: Sequence[Union[GeoPoint, Sequence[float]]],
                    tolerance_m: float) -> List[GeoPoint]:
    """Ramer-Douglas-Peucker simplification: drop vertices whose great-circle
    perpendicular (cross-track) distance from the retained segments is below
    ``tolerance_m`` metres."""
    pts = [_as_point(q) for q in polyline]
    if len(pts) < 3:
        return list(pts)
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        max_d = -1.0
        idx = lo
        for i in range(lo + 1, hi):
            d = abs(cross_track_distance(pts[i], pts[lo], pts[hi]))
            if d > max_d:
                max_d, idx = d, i
        if max_d > tolerance_m:
            keep[idx] = True
            stack.append((lo, idx))
            stack.append((idx, hi))
    return [p for p, k in zip(pts, keep) if k]


# ─────────────────────────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────────────────────────

def dbscan(points: Union[Mapping[NodeId, GeoPoint], Sequence[GeoPoint]],
           eps_m: float, min_samples: int = 4) -> Dict[NodeId, int]:
    """Density-based clustering (DBSCAN) with neighbourhood radius ``eps_m``
    metres (great-circle).  Returns ``node_id -> cluster_label``; label ``-1``
    marks noise.  Uses a :class:`GeoIndex` for efficient region queries."""
    if eps_m < 0:
        raise ValueError("eps_m must be >= 0, got %r" % (eps_m,))
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1, got %r" % (min_samples,))
    pts = _normalise_points(points)
    idx = GeoIndex(pts)
    labels: Dict[NodeId, int] = {nid: -2 for nid in pts}   # -2 = unvisited
    cluster = -1
    for nid in pts:
        if labels[nid] != -2:
            continue
        neighbors = [o for o, _ in idx.within_radius(pts[nid], eps_m)]
        if len(neighbors) < min_samples:
            labels[nid] = -1                       # noise (may be reclaimed)
            continue
        cluster += 1
        labels[nid] = cluster
        seeds = [o for o in neighbors if o != nid]
        si = 0
        while si < len(seeds):
            o = seeds[si]
            si += 1
            if labels[o] == -1:
                labels[o] = cluster                # border point
            if labels[o] != -2:
                continue
            labels[o] = cluster
            o_neighbors = [q for q, _ in idx.within_radius(pts[o], eps_m)]
            if len(o_neighbors) >= min_samples:
                seeds.extend(o_neighbors)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# I/O & payloads
# ─────────────────────────────────────────────────────────────────────────────

def _json_num(x: Any) -> Any:
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return None
    return x


def to_geojson(graph_or_points: Union["GeoGraph",
                                      Mapping[NodeId, GeoPoint],
                                      Sequence[GeoPoint]]) -> Dict[str, Any]:
    """Render a GeoGraph (nodes as Point features, edges as LineStrings) or a
    set of points (Point features) as a GeoJSON FeatureCollection.  GeoJSON
    coordinate order is [lon, lat]."""
    features: List[Dict[str, Any]] = []
    if isinstance(graph_or_points, GeoGraph):
        g = graph_or_points
        for nid in g.nodes:
            p = g.coord(nid)
            features.append({
                "type": "Feature",
                "properties": {"id": str(nid)},
                "geometry": {"type": "Point", "coordinates": [p.lon, p.lat]},
            })
        seen: Set[Tuple[NodeId, NodeId]] = set()
        for u in g.nodes:
            for v, w in g.neighbors(u).items():
                key = (u, v) if g.directed else tuple(sorted((u, v), key=repr))
                if key in seen:
                    continue
                seen.add(key)
                pu, pv = g.coord(u), g.coord(v)
                features.append({
                    "type": "Feature",
                    "properties": {"weight": _json_num(w)},
                    "geometry": {"type": "LineString",
                                 "coordinates": [[pu.lon, pu.lat],
                                                 [pv.lon, pv.lat]]},
                })
    else:
        for nid, p in _normalise_points(graph_or_points).items():
            features.append({
                "type": "Feature",
                "properties": {"id": str(nid)},
                "geometry": {"type": "Point", "coordinates": [p.lon, p.lat]},
            })
    return {"type": "FeatureCollection", "features": features}


def build_geo_payload(graph: "GeoGraph") -> Dict[str, Any]:
    """Frontend-ready summary of a GeoGraph."""
    nodes = [{"id": str(n), "lat": _json_num(graph.coord(n).lat),
              "lon": _json_num(graph.coord(n).lon)} for n in graph.nodes]
    edges = []
    seen: Set[Tuple[NodeId, NodeId]] = set()
    for u in graph.nodes:
        for v, w in graph.neighbors(u).items():
            key = (u, v) if graph.directed else tuple(sorted((u, v), key=repr))
            if key in seen:
                continue
            seen.add(key)
            edges.append({"source": str(u), "target": str(v),
                          "weight": _json_num(round(w, 3))})
    bbox = bounding_box([graph.coord(n) for n in graph.nodes]) \
        if graph.nodes else (0.0, 0.0, 0.0, 0.0)
    return {
        "numNodes": graph.num_nodes,
        "numEdges": graph.num_edges,
        "directed": graph.directed,
        "bbox": {"minLat": bbox[0], "minLon": bbox[1],
                 "maxLat": bbox[2], "maxLon": bbox[3]},
        "nodes": nodes,
        "edges": edges,
    }


def build_a2ui_payload(graph: "GeoGraph") -> Dict[str, Any]:
    """Map-visualisation-ready dict: positioned markers + edges + stat cards."""
    markers = [{"id": str(n), "lat": _json_num(graph.coord(n).lat),
                "lon": _json_num(graph.coord(n).lon),
                "degree": len(graph.neighbors(n))} for n in graph.nodes]
    edges = []
    seen: Set[Tuple[NodeId, NodeId]] = set()
    for u in graph.nodes:
        for v, w in graph.neighbors(u).items():
            key = (u, v) if graph.directed else tuple(sorted((u, v), key=repr))
            if key in seen:
                continue
            seen.add(key)
            edges.append({"from": [_json_num(graph.coord(u).lat),
                                   _json_num(graph.coord(u).lon)],
                          "to": [_json_num(graph.coord(v).lat),
                                 _json_num(graph.coord(v).lon)],
                          "weight": _json_num(round(w, 1))})
    total_km = sum(e["weight"] for e in edges) / 1000.0 if edges else 0.0
    return {
        "markers": markers,
        "edges": edges,
        "statCards": [
            {"label": "Nodes", "value": graph.num_nodes},
            {"label": "Edges", "value": graph.num_edges},
            {"label": "Total km", "value": round(total_km, 1)},
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

_CITIES = {
    "London": GeoPoint(51.5074, -0.1278),
    "Paris": GeoPoint(48.8566, 2.3522),
    "Berlin": GeoPoint(52.5200, 13.4050),
    "Madrid": GeoPoint(40.4168, -3.7038),
    "Rome": GeoPoint(41.9028, 12.4964),
    "Amsterdam": GeoPoint(52.3676, 4.9041),
    "Vienna": GeoPoint(48.2082, 16.3738),
    "Prague": GeoPoint(50.0755, 14.4378),
    "Brussels": GeoPoint(50.8503, 4.3517),
    "Zurich": GeoPoint(47.3769, 8.5417),
}


def _demo() -> None:
    print("=" * 70)
    print("GEOSPATIAL GRAPHS TEMPLATE - demo")
    print("=" * 70)

    print("\n[1] Distances London <-> Paris (4 methods)")
    lo, pa = _CITIES["London"], _CITIES["Paris"]
    for m in ("haversine", "cosines", "equirectangular", "vincenty"):
        print("    %-16s %.1f km" % (m, geodesic(lo, pa, m) / 1000.0))

    print("\n[2] Navigation: bearing & destination")
    print("    initial bearing London->Paris: %.1f deg" % bearing(lo, pa))
    d = destination(lo, 90.0, 100_000)
    print("    100 km due east of London: (%.4f, %.4f)" % (d.lat, d.lon))

    print("\n[3] Spatial index: nearest city to (50.0, 5.0)")
    idx = GeoIndex(_CITIES)
    nid, dist = idx.nearest((50.0, 5.0))
    print("    nearest: %s (%.1f km)" % (nid, dist / 1000.0))
    print("    3 nearest:", [(n, round(d / 1000.0))
                             for n, d in idx.k_nearest((50.0, 5.0), 3)])
    print("    within 500 km:", sorted(n for n, _ in
                                       idx.within_radius((50.0, 5.0), 500_000)))

    print("\n[4] Geohash of Paris (precision 7)")
    gh = geohash_encode(pa.lat, pa.lon, 7)
    back = geohash_decode(gh)
    print("    geohash=%s -> decoded (%.4f, %.4f)" % (gh, back.lat, back.lon))
    print("    east neighbor:", geohash_neighbors(gh)["e"])

    print("\n[5] k-NN graph (k=3) + routing")
    g = GeoGraph.knn_graph(_CITIES, k=3)
    print("    nodes=%d edges=%d" % (g.num_nodes, g.num_edges))
    path = g.astar("Madrid", "Berlin")
    if path:
        print("    A* Madrid->Berlin:", " -> ".join(map(str, path)))
        print("    length: %.0f km" % (g.path_length(path) / 1000.0))
        dij = g.shortest_path("Madrid", "Berlin")
        print("    matches Dijkstra:", path == dij or
              abs(g.path_length(path) - g.path_length(dij)) < 1e-6)

    print("\n[6] Isochrone: cities within 1500 km of Brussels (k-NN graph)")
    reach = g.reachable_within("Brussels", 1_500_000)
    print("    ", sorted((n, round(d / 1000.0)) for n, d in reach.items()))

    print("\n[7] TSP tour (nearest-neighbour + 2-opt)")
    tour = g.nearest_neighbor_tour("London")
    nn_len = g.tour_length(tour)
    opt = g.two_opt(tour)
    print("    NN tour:  %.0f km" % (nn_len / 1000.0))
    print("    2-opt:    %.0f km" % (g.tour_length(opt) / 1000.0))

    print("\n[8] Geometry: convex hull + area + point-in-polygon")
    hull = convex_hull(list(_CITIES.values()))
    print("    hull has %d vertices" % len(hull))
    area = spherical_polygon_area(hull)
    print("    hull area: %.0f thousand km^2" % (area / 1e9))
    print("    is Zurich inside the hull:",
          point_in_polygon(_CITIES["Zurich"], hull))

    print("\n[9] DBSCAN clustering (eps=600 km, min_samples=2)")
    labels = dbscan(_CITIES, 600_000, min_samples=2)
    clusters: Dict[int, List[str]] = {}
    for nid, lab in labels.items():
        clusters.setdefault(lab, []).append(str(nid))
    for lab in sorted(clusters):
        tag = "noise" if lab == -1 else "cluster %d" % lab
        print("    %-10s %s" % (tag, sorted(clusters[lab])))

    print("\n[10] Douglas-Peucker simplification + GeoJSON")
    line = [_CITIES["London"], _CITIES["Brussels"], _CITIES["Amsterdam"],
            _CITIES["Berlin"], _CITIES["Prague"], _CITIES["Vienna"]]
    simp = douglas_peucker(line, 50_000)
    print("    polyline %d -> %d vertices (tol 50 km)" % (len(line), len(simp)))
    import json
    gj = to_geojson(g)
    print("    GeoJSON features:", len(gj["features"]),
          "| bytes:", len(json.dumps(gj)))

    print("\n" + "=" * 70)
    print("demo OK")
    print("=" * 70)


if __name__ == "__main__":
    _demo()

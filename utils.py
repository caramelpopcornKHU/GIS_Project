import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from shapely.geometry import Polygon, box, LineString
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPoint
from scipy.spatial import Voronoi
from tqdm import tqdm  # tqdm 추가
import os


import time
import hashlib
import scipy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, Birch
from sklearn.datasets import make_blobs


def construct_voronoi_polygons(X_geo, extend_distance=None):
    vor = Voronoi(X_geo)

    center = vor.points.mean(axis=0)
    if extend_distance is None:
        extend_distance = np.ptp(vor.points, axis=0).max() * 2 # 데이터 범위의 2배로 연장

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    polygons = []

    for p_idx, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if -1 not in vertices:
            # 유한한 영역 -> 바로 Polygon 생성
            poly_points = [vor.vertices[v] for v in vertices]
            polygons.append(Polygon(poly_points))
        else:
            # 무한한 영역 -> 보완이 필요
            ridge_segments = []
            ridges = all_ridges[p_idx]
            for p2, v1, v2 in ridges:
                v1, v2 = np.asarray(v1), np.asarray(v2)
                if v1 >= 0 and v2 >= 0:
                    # 유한한 에지
                    ridge_segments.append(vor.vertices[[v1, v2]])
                else:
                    # 무한한 에지
                    if v1 == -1:
                        v = v2
                    else:
                        v = v1
                    tangent = vor.points[p2] - vor.points[p_idx]
                    tangent = tangent / np.linalg.norm(tangent)
                    normal = np.array([-tangent[1], tangent[0]])

                    if np.dot(normal, vor.vertices[v] - center) < 0:
                        normal = -normal

                    far_point = vor.vertices[v] + normal * extend_distance
                    ridge_segments.append(np.vstack([vor.vertices[v], far_point]))

            # 이 모든 선분들의 꼭짓점을 연결
            poly_points = []
            for seg in ridge_segments:
                poly_points.append(seg[0])
                poly_points.append(seg[1])
            poly = Polygon(poly_points)
            polygons.append(poly.convex_hull)  # Convex Hull로 정리해서 Polygon 생성

    return polygons, vor


# ====== Geo-SOM Class ======
class GeoSOM:
    def __init__(self, grid_shape, geo_tolerance, input_dim, learning_rate=0.01, radius=2, fixed_geo=False):
        self.grid_shape = grid_shape
        self.geo_tolerance = geo_tolerance
        self.input_dim = input_dim  # X_ngf.shape[1] 대신 직접 받도록 수정
        self.learning_rate = learning_rate
        self.radius = radius
        self.fixed_geo = fixed_geo
        self.init_map()

    def init_map(self):
        rows, cols = self.grid_shape
        self.map_geo = np.array([[(i, j) for j in range(cols)] for i in range(rows)], dtype=np.float64).reshape(-1, 2)
        self.map_ngf = np.random.rand(rows * cols, self.input_dim)

    def train(self, X_geo, X_ngf, epochs=100):
        for epoch in tqdm(range(epochs), desc="Training GeoSOM"):  # tqdm 적용
            alpha = self.learning_rate * (1 - epoch / epochs)
            radius = self.radius * (1 - epoch / epochs)
            for i in range(len(X_geo)):
                x_geo = X_geo[i]
                x_ngf = X_ngf[i]
                dists_geo = np.linalg.norm(self.map_geo - x_geo, axis=1)
                geo_winner_idx = np.argmin(dists_geo)
                geo_winner_pos = self.map_geo[geo_winner_idx]
                neighborhood_idx = [j for j, unit_pos in enumerate(self.map_geo)
                                    if np.linalg.norm(geo_winner_pos - unit_pos) <= self.geo_tolerance]

                bmu_idx = min(neighborhood_idx,
                              key=lambda j: np.linalg.norm(x_ngf - self.map_ngf[j]))

                for j in neighborhood_idx:
                    dist_output = np.linalg.norm(self.map_geo[bmu_idx] - self.map_geo[j])
                    if dist_output <= radius:
                        h = np.exp(-(dist_output ** 2) / (2 * (radius ** 2)))
                        if self.fixed_geo:
                            self.map_ngf[j] += alpha * h * (x_ngf - self.map_ngf[j])
                        else:
                            combined_input = np.concatenate([x_geo, x_ngf])
                            combined_weight = np.concatenate([self.map_geo[j], self.map_ngf[j]])
                            delta = alpha * h * (combined_input - combined_weight)
                            self.map_geo[j] += delta[:2]
                            self.map_ngf[j] += delta[2:]


    def get_cluster_assignments(self, X_geo, X_ngf):
        assignments = []
        for i in range(len(X_geo)):
            x_geo = X_geo[i]
            x_ngf = X_ngf[i]
            dists_geo = np.linalg.norm(self.map_geo - x_geo, axis=1)
            geo_winner_idx = np.argmin(dists_geo)
            geo_winner_pos = self.map_geo[geo_winner_idx]
            neighborhood_idx = [j for j, unit_pos in enumerate(self.map_geo)
                                if np.linalg.norm(geo_winner_pos - unit_pos) <= self.geo_tolerance]
            bmu_idx = min(neighborhood_idx, key=lambda j: np.linalg.norm(x_ngf - self.map_ngf[j]))
            assignments.append(bmu_idx)
        return np.array(assignments)


# https://www.kaggle.com/code/mallikarjunaj/gap-statistics

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def optimalK(data, maxClusters):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie

    Params:
        data: ndarray of shape (n_samples, n_features)
        maxClusters: Maximum number of clusters to test for

    Returns: (optimal_k, resultsdf)
    """
    nrefs = 3
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame(columns=['clusterCount', 'gap'])

    for gap_index, k in enumerate(range(1, maxClusters)):
        # 기준선 분산값 저장
        refDisps = np.zeros(nrefs)

        # nrefs개 기준선 데이터에 대해 KMeans
        for i in range(nrefs):
            randomReference = np.random.random_sample(size=data.shape)
            km = KMeans(n_clusters=k, n_init=10)
            km.fit(randomReference)
            refDisps[i] = km.inertia_

        # 실제 데이터에 대해 KMeans
        km = KMeans(n_clusters=k, n_init=10)
        km.fit(data)
        origDisp = km.inertia_

        # Gap 통계량 계산
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        gaps[gap_index] = gap

        # 결과 누적
        new_row = pd.DataFrame([{'clusterCount': k, 'gap': gap}])
        resultsdf = pd.concat([resultsdf, new_row], ignore_index=True)

    optimal_k = gaps.argmax() + 1  # index 0 → k=1
    return optimal_k, resultsdf



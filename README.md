# GIS_Project
2025 GIS Project repo


GIS_Project_GEO_SOM.ipynb 파일은 서울시 건물정보 데이터를 활용하여, GEO-SOM알고리즘을 학습시킨다.   
건물데이터의 다차원 특성이 반영되어 할당된 cluster들은 보르노이(Vornoi) 폴리곤으로 면적을 할당한다.

GEO_SOM_Seoul_K_Means.ipynb 파일은 할당된 클러스터들을 유클리디언 거리기반의 k-means알고리즘을 통하여   
서울시의 건물정보기반의 군집을 형성한다. 이때, 최적의 군집의 수를 확인하기 위하여, Gap-statistics를 사용하여   
선정된 군집의 수로 서울을 나눈다.   

코드의 독해를 위하여, GEO-SOM알고리즘, Vornoi폴리곤 자르기, Gap-statistics는 utils.py 파일에   
저장 후, 함수만을 꺼내서 사용하였다.

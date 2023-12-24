from osgeo import ogr
import json

# KML 파일을 읽습니다
kml_file = 'datafiles/Korea International Circuit.kml'
kml_ds = ogr.Open(kml_file)

# 첫 번째 레이어를 가져옵니다
layer = kml_ds.GetLayer()

# GeoJSON으로 변환
geojson = layer.ExportToJson()

# GeoJSON 파일로 저장
with open('output.geojson', 'w') as f:
    f.write(geojson)

# 이제 GeoJSON 파일을 geopandas로 읽을 수 있습니다.

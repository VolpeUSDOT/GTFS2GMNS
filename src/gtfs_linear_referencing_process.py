import os
import time
import arcpy
from collections import defaultdict

# INPUTS
gtfs_folder = r'C:\Tasks\RDR\gtfs_HR\helper_tool\GTFS_HamptonRoads'
output_dir = r'C:\Tasks\RDR\gtfs_HR\helper_tool\output'
# TODO: write code to "create" this XLSX file from the current outputs of gtfs2gmns.py
# TODO: what is missing - (1) geometry_id field (where does it come from, only filled for service links, set to 0 for boarding links/transfer links)
xls_with_stop_data = 'C:\Tasks\RDR\gtfs_HR\helper_tool\other_inputs\HRTExample.xlsx'

# SETUP
if not os.path.exists(output_dir):
    print('making directory {}'.format(output_dir))
    os.mkdir(output_dir)

gdb_name = "gtfs_gis.gdb"
fp_to_gdb = os.path.join(output_dir, gdb_name)

if arcpy.Exists(fp_to_gdb):
    print('deleting existing gdb ...')
    arcpy.Delete_management(fp_to_gdb)
    time.sleep(.5)

print('creating gdb ...')
arcpy.CreateFileGDB_management(output_dir, gdb_name)
arcpy.env.workspace = fp_to_gdb

# MAIN
# Converts shapes to GIS format
print('Creating GIS format of shapes.txt ...')
arcpy.conversion.GTFSShapesToFeatures(os.path.join(gtfs_folder, "shapes.txt"), "gtfs_shapes")

# Converts stops to GIS format
print('Creating GIS format of stops.txt ...')
arcpy.conversion.GTFSStopsToFeatures(os.path.join(gtfs_folder, "stops.txt"), "gtfs_stops")

# Create routes
print('Linear reference the shapes ...')
arcpy.lr.CreateRoutes("gtfs_shapes", "shape_id", "gtfs_shapes_routes", "LENGTH")

# Locate stops
print('Locates stops along these linear referenced shapes ...')
arcpy.lr.LocateFeaturesAlongRoutes("gtfs_stops", "gtfs_shapes_routes", "shape_id", "100 meters", "stops_along_shapes", "shape_id POINT mp", "ALL")

# Build up the from and to_ linear events using the above output and the list of stop pairs that we have.
print('build dictionary of stop mileposts ...')
stop_mp_dict = defaultdict(dict)
for row in arcpy.da.SearchCursor("stops_along_shapes", ["shape_id", "stop_id", "mp"]):
    shape_id = row[0]
    stop_id = row[1]
    mp = row[2]
    stop_mp_dict[shape_id][stop_id] = mp

# Create table to store route events
print('Creating route event table ...')
arcpy.management.CreateTable(fp_to_gdb, "route_event_table")
arcpy.AddField_management("route_event_table", "link_id", "TEXT")
arcpy.AddField_management("route_event_table", "shape_id", "TEXT")
arcpy.AddField_management("route_event_table", "from_stop_id", "TEXT")
arcpy.AddField_management("route_event_table", "from_mp", "DOUBLE")
arcpy.AddField_management("route_event_table", "to_stop_id", "TEXT")
arcpy.AddField_management("route_event_table", "to_mp", "DOUBLE")

print('Process the table of stop-stop pairs ...')
# First get it into an ArcGIS format
arcpy.conversion.ExcelToTable(xls_with_stop_data, "stop_data", "link")

icursor = arcpy.da.InsertCursor("route_event_table", ['link_id', 'shape_id', 'from_stop_id', 'from_mp', 'to_stop_id', 'to_mp'])

# Generate the event layer by iterating through the table of shape/stop pairs and the located stops
# geometry_id is same as shape_id
# TODO need stop ids rather than node ids
for row in arcpy.da.SearchCursor("stop_data", ["link_id", "from_node_id", "to_node_id", "geometry_id"]):
    link_id = row[0]
    from_stop_id = row[1]
    to_stop_id = row[2]
    geometry_id = row[3]
    from_mp = stop_mp_dict[geometry_id][from_stop_id]
    to_mp = stop_mp_dict[geometry_id][to_stop_id]

    icursor.insertRow([link_id, geometry_id, from_stop_id, from_mp, to_stop_id, to_mp])

# Make route event layer
print('Making GIS dataset of route events ...')
arcpy.lr.MakeRouteEventLayer("gtfs_shapes_routes", "shape_id", "route_event_table", "shape_id LINE fr_mp to_mp", "route_event_lyr")

# Save to disk
arcpy.CopyFeatures_management("route_event_lyr", "link_id_routes")
                             
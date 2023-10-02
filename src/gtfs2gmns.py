# -*- coding:utf-8 -*-
##############################################################
# Created Date: Wednesday, November 16th 2022
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
# Modified: Monday, April 10th 2023 by Volpe
##############################################################

import os
import sys
import math
import datetime
import numpy as np
import pandas as pd
import time
import argparse
from pathlib import Path

base_dir = Path(__file__).parent
sys.path.append(str(base_dir))

from utility_lib import (func_running_time,
                         get_txt_files_from_folder,
                         check_required_files_exist,
                         path2linux,
                         validate_filename)

from func_lib import (reading_text,
                      hhmm_to_minutes,
                      determine_terminal_flag,
                      stop_sequence_label,
                      convert_route_type_to_node_type_p,
                      convert_route_type_to_node_type_s,
                      convert_route_type_to_link_type,
                      calculate_distance_from_geometry,
                      allowed_use_function,
                      transferring_penalty,
                      allowed_use_transferring)

class GTFS2GMNS:

    def __init__(self, gtfs_dir: str, gtfs_result_dir:str = "", time_period: str = '0700_0800'):

        # TDD development
        if not os.path.isdir(gtfs_dir):
            raise ValueError('The input folder does not exist.')

        if not os.path.isdir(gtfs_result_dir):
            raise ValueError('The output folder does not exist.')

        self.gtfs_dir = gtfs_dir
        self.time_period = time_period
        self.gtfs_result_dir = gtfs_result_dir

        self.period_start_time, self.period_end_time = hhmm_to_minutes(self.time_period)
        self.required_files = ['agency.txt', 'stops.txt', 'routes.txt', 'trips.txt', 'stop_times.txt']

    @func_running_time
    def read_gtfs_data(self, gtfs_dir: dict) -> list:
        """Function to read GTFS data

        Files to read:
            - agency.txt
            - routes.txt
            - trips.txt
            - stops.txt
            - stop_times.txt

        Returns a list of DataFrames:
            - stop_df, route_df, trip_df, trip_route_df, stop_time_df, directed_trip_route_stop_time_df
        """

        # required_files = ['agency.txt', 'stops.txt', 'routes.txt', 'trips.txt', 'stop_times.txt']
        required_files_dict = {file: path2linux(os.path.join(gtfs_dir, file)) for file in self.required_files}

        # Step 1: check if required files exist in the folder
        print(f"Info: Checking if required files exist in the folder: {gtfs_dir} \n")

        txt_files_from_folder_abspath = get_txt_files_from_folder(gtfs_dir)
        if not check_required_files_exist(list(required_files_dict.values()), txt_files_from_folder_abspath):
            raise Exception("Error: Required files not exist in the folder!")

        # Step 2: read GTFS data
        print('Info: start reading GTFS: agency file...')
        agency_df = pd.read_csv(required_files_dict.get("agency.txt"), encoding='utf-8-sig')
        agency_name = agency_df['agency_name'][0]
        print(f"Info: agent_name: {agency_name} \n")

        print('Info: start reading GTFS: stops file...')
        stop_df = pd.read_csv(required_files_dict.get("stops.txt"), encoding='utf-8-sig')
        stop_df = stop_df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]
        print(f"Info: number of stops = {len(stop_df)} \n")

        print('Info: start reading GTFS: routes file...')
        route_df = pd.read_csv(required_files_dict.get("routes.txt"), encoding='utf-8-sig', dtype=str)
        route_df = route_df[['route_id', 'route_short_name', 'route_long_name', 'route_type']]
        print(f"Info: number of routes = {len(route_df)} \n")

        print('Info: start reading GTFS: trips file...')
        trip_df = pd.read_csv(required_files_dict.get("trips.txt"), encoding='utf-8-sig', dtype=str)
        trip_df["trip_id"] = trip_df["trip_id"].astype(str)

        # direction_id is mandatory field name here
        if 'direction_id' not in trip_df.columns.tolist():
            trip_df['direction_id'] = "0"
        trip_df['direction_id'] = trip_df.apply(lambda x: str(2 - int(x['direction_id'])), axis=1)

        # add a new column "directed_route_id"
        #  If trips on a route service opposite directions,distinguish directions using values 0 and 1.
        # revise the direction_id from 0,1 to 2,1
        # add a new field directed_route_id
        # deal with special issues of Agency 12 Fairfax CUE # Alicia, Nov 10:
        # route file has route id with quotes, e.g., '"green2"' while trip file does not have it, e.g.,'green2'
        directed_route_id = trip_df['route_id'].astype(str).str.cat(trip_df['direction_id'].astype(str), sep='.')
        trip_df['directed_route_id'] = directed_route_id

        if (route_df['route_id'][0][0] == '"') != (trip_df['route_id'][0][0] == '"'):
            if route_df['route_id'][0][0] == '"':
                route_df['route_id'] = route_df.apply(lambda x: x['route_id'].strip('"'), axis=1)
            else:
                trip_df['route_id'] = trip_df.apply(lambda x: x['route_id'].strip('"'), axis=1)

        # Left merge, as route is higher level planning than trips, len(trip_route_df)=len(trip_df)
        trip_route_df = pd.merge(trip_df, route_df, on='route_id')
        trip_route_df["trip_id"] = trip_route_df["trip_id"].astype(str)
        print(f"Info: number of trips = {len(trip_route_df)} ... { len(trip_df)} \n")

        print('Info: start reading GTFS: stop_times file...')
        stop_time_df = pd.read_csv(required_files_dict.get("stop_times.txt"), encoding='utf-8-sig')
        print(f"Info: number of stop_time records = {len(stop_time_df)}")

        # drop the stations without accurate arrival and departure time.
        # drop nan
        stop_time_df = stop_time_df.dropna(subset=['arrival_time'], how='any')
        # drop ''
        stop_time_df = stop_time_df[stop_time_df.arrival_time != '']
        stop_time_df = stop_time_df[stop_time_df.departure_time != '']

        # drop ' '
        stop_time_df = stop_time_df[stop_time_df.arrival_time != ' ']
        stop_time_df = stop_time_df[stop_time_df.departure_time != ' ']
        print(f"    number of stop_time records after dropping empty arrival and departure time = {len(stop_time_df)} \n")

        # convert timestamp to minute
        # as some agencies might have trips overlapping two days, should use _to_timedelta to convert the data
        print("Info: start converting the time stamps...")
        tt = datetime.datetime(2021, 1, 1, 0, 0, 0, 0)
        stop_time_df['arrival_time'] = pd.to_timedelta(stop_time_df['arrival_time']) + tt
        stop_time_df['departure_time'] = pd.to_timedelta(stop_time_df['departure_time']) + tt
        stop_time_df['arrival_time'] = stop_time_df['arrival_time'].apply(lambda x: x.hour * 60 + x.minute + 1440 * (x.day - 1))
        stop_time_df['departure_time'] = stop_time_df['departure_time'].apply(lambda x: x.hour * 60 + x.minute + 1440 * (x.day - 1))

        print("Info: start marking terminal flags for stops...")
        iteration_group = stop_time_df.groupby(['trip_id'])
        # mark terminal flag for each stop. The terminals can only be determined at the level of trips

        input_list = []
        time_start = time.time()
        for trip_id, trip_stop_time_df in iteration_group:
            trip_stop_time_df = trip_stop_time_df.sort_values(by=['stop_sequence'])
            trip_stop_time_df = trip_stop_time_df.reset_index()

            # select only the trips within the provided time window
            mask1 = trip_stop_time_df.arrival_time.min() <= self.period_end_time
            mask2 = trip_stop_time_df.arrival_time.min() >= self.period_start_time
            if mask1 and mask2:
                input_list.append(trip_stop_time_df)

        intermediate_output_list = list(map(determine_terminal_flag, input_list))
        output_list = list(map(stop_sequence_label, intermediate_output_list))
        print(f'Info: add terminal_flag for trips using CPU time:{time.time() - time_start} s \n')

        time_start = time.time()
        stop_time_df_with_terminal = pd.concat(output_list, axis=0)
        stop_time_df_with_terminal["trip_id"] = stop_time_df_with_terminal["trip_id"].astype(str)
        print(f'Info: concatenate different trips using CPU time: {time.time() - time_start} s')
        print(f"    have updated {len(stop_time_df_with_terminal)} stop_time records \n")

        print("Info: merge the route information with trip information...")
        directed_trip_route_stop_time_df = pd.merge(trip_route_df, stop_time_df_with_terminal, on='trip_id')
        print(f"    number of final merged records = {len(directed_trip_route_stop_time_df)}")
        print("Info: Data reading done.. \n")

        #  as trip is higher level planning than stop time scheduling, len(stop_time_df)>=len(trip_df)
        #  Each record of directed_trip_route_stop_time_df represents a space-time state of a vehicle
        # trip_id (different vehicles, e.g., train lines)
        # stop_id (spatial location of the vehicle)
        # arrival_time,departure_time (time index of the vehicle)

        directed_route_stop_id = directed_trip_route_stop_time_df['directed_route_id'].astype(
            str).str.cat(directed_trip_route_stop_time_df['stop_id'].astype(str), sep='.')

        # directed_route_stop_id is a unique id to identify the route, direction, and stop of a vehicle at a time point
        directed_trip_route_stop_time_df['directed_route_stop_id'] = directed_route_stop_id

        directed_trip_route_stop_time_df['stop_sequence'] = directed_trip_route_stop_time_df['stop_sequence'].astype('int32')

        # two important concepts :
        # 1 directed_service_stop_id (directed_route_stop_id + stop sequence)
        directed_trip_route_stop_time_df['directed_service_stop_id'] = \
            directed_trip_route_stop_time_df.directed_route_stop_id.astype(str) + ':' + \
            directed_trip_route_stop_time_df.stop_sequence_label

        # 2. directed service id (directed_route_id + stop sequence) same directed route id might have different sequences
        directed_trip_route_stop_time_df['directed_service_id'] = \
            directed_trip_route_stop_time_df.directed_route_id.astype(str) + ':' + \
            directed_trip_route_stop_time_df.stop_sequence_label

        # attach stop name and geometry for stops
        directed_trip_route_stop_time_df = pd.merge(directed_trip_route_stop_time_df, stop_df, on='stop_id')
        directed_trip_route_stop_time_df['agency_name'] = agency_name

        return [stop_df, route_df, trip_df, trip_route_df, stop_time_df, directed_trip_route_stop_time_df]

    @func_running_time
    def create_nodes(self, directed_trip_route_stop_time_df: pd.DataFrame) -> pd.DataFrame:

        print("Info: start creating physical nodes...")

        """create physical (station) node... \n"""
        physical_node_df = pd.DataFrame()
        temp_df = directed_trip_route_stop_time_df.drop_duplicates(subset=['stop_id'])
        physical_node_df['name'] = temp_df['stop_id']
        physical_node_df = physical_node_df.sort_values(by=['name'])
        physical_node_df['node_id'] = \
            np.linspace(start=1, stop=len(physical_node_df), num=len(physical_node_df)).astype('int32')
        physical_node_df['node_id'] += int('1000000')
        physical_node_df['physical_node_id'] = physical_node_df['node_id']
        physical_node_df['x_coord'] = temp_df['stop_lon'].astype(float)
        physical_node_df['y_coord'] = temp_df['stop_lat'].astype(float)
        physical_node_df['route_type'] = temp_df['route_type']
        physical_node_df['route_id'] = temp_df['route_id']
        physical_node_df['node_type'] = \
            physical_node_df.apply(lambda x: convert_route_type_to_node_type_p(x.route_type), axis=1)
        physical_node_df['directed_route_id'] = ""
        physical_node_df['directed_service_id'] = ""
        physical_node_df['zone_id'] = ""
        physical_node_df['agency_name'] = temp_df['agency_name']
        physical_node_df['geometry'] = 'POINT (' + physical_node_df['x_coord'].astype(str) + \
                                    ' ' + physical_node_df['y_coord'].astype(str) + ')'
        stop_name_id_dict = dict(zip(physical_node_df['name'], physical_node_df['node_id']))
        physical_node_df['terminal_flag'] = temp_df['terminal_flag']
        physical_node_df['ctrl_type'] = ""
        physical_node_df['agent_type'] = ""

        print("Info: start creating service nodes... \n")
        """ create service node..."""
        service_node_df = pd.DataFrame()
        temp_df = directed_trip_route_stop_time_df.drop_duplicates(subset=['directed_service_stop_id'])
        # 2.2.2 route stop node
        service_node_df['name'] = temp_df['directed_service_stop_id']
        service_node_df = service_node_df.sort_values(by=['name'])
        service_node_df['node_id'] = \
            np.linspace(start=1, stop=len(service_node_df), num=len(service_node_df)).astype('int32')
        service_node_df['physical_node_id'] = temp_df.apply(lambda x: stop_name_id_dict[x.stop_id], axis=1)
        service_node_df['node_id'] += int('1500000')

        service_node_df['x_coord'] = temp_df['stop_lon'].astype(float) - 0.000100
        service_node_df['y_coord'] = temp_df['stop_lat'].astype(float) - 0.000100
        service_node_df['route_type'] = temp_df['route_type']
        service_node_df['route_id'] = temp_df['route_id']
        service_node_df['node_type'] = \
            service_node_df.apply(lambda x: convert_route_type_to_node_type_s(x.route_type), axis=1)
        # node_csv['terminal_flag'] = ' '
        service_node_df['directed_route_id'] = temp_df['directed_route_id'].astype(str)
        service_node_df['directed_service_id'] = temp_df['directed_service_id'].astype(str)
        service_node_df['zone_id'] = ""
        service_node_df['agency_name'] = temp_df['agency_name']
        service_node_df['geometry'] = \
            'POINT (' + service_node_df['x_coord'].astype(str) + ' ' + service_node_df['y_coord'].astype(str) + ')'

        service_node_df['terminal_flag'] = temp_df['terminal_flag']
        service_node_df['ctrl_type'] = ""
        service_node_df['agent_type'] = ""

        print("Info: finished creating nodes...")
        return pd.concat([physical_node_df, service_node_df])

    @func_running_time
    def create_service_boarding_links(self, directed_trip_route_stop_time_df: pd.DataFrame, node_df, one_agency_link_list: list) -> list:

        # initialize dictionaries
        node_id_dict = dict(zip(node_df['name'], node_df['node_id']))
        directed_service_dict = dict(zip(node_df['node_id'], node_df['name']))
        node_lon_dict = dict(zip(node_df['node_id'], node_df['x_coord']))
        node_lat_dict = dict(zip(node_df['node_id'], node_df['y_coord']))
        frequency_dict = {}

        print("Info: 1. start creating route links...")
        # generate service links
        number_of_route_links = 0
        iteration_group = directed_trip_route_stop_time_df.groupby('directed_service_id')
        labeled_directed_service_list = []

        time_start = time.time()
        for directed_service_id, route_df in iteration_group:
            if directed_service_id in labeled_directed_service_list:
                continue
            else:
                labeled_directed_service_list.append(directed_service_id)
                number_of_trips = len(route_df.trip_id.unique())
                frequency_dict[directed_service_id] = number_of_trips  # note the frequency of routes
                one_line_df = route_df[route_df.trip_id == route_df.trip_id.unique()[0]]
                one_line_df = one_line_df.sort_values(by=['stop_sequence'])
                number_of_records = len(one_line_df)
                one_line_df = one_line_df.reset_index()

                for k in range(number_of_records - 1):
                    link_id = 1000000 + number_of_route_links + 1
                    from_node_id = node_id_dict[one_line_df.iloc[k].directed_service_stop_id]
                    to_node_id = node_id_dict[one_line_df.iloc[k + 1].directed_service_stop_id]
                    facility_type = convert_route_type_to_link_type(one_line_df.iloc[k].route_type)
                    dir_flag = 1
                    directed_route_id = one_line_df.iloc[k].directed_route_id
                    link_type = 1
                    link_type_name = 'service_links'
                    from_node_lon = float(one_line_df.iloc[k].stop_lon)
                    from_node_lat = float(one_line_df.iloc[k].stop_lat)
                    to_node_lon = float(one_line_df.iloc[k + 1].stop_lon)
                    to_node_lat = float(one_line_df.iloc[k + 1].stop_lat)
                    length = calculate_distance_from_geometry(from_node_lon, from_node_lat, to_node_lon, to_node_lat, 'mile')
                    lanes = number_of_trips
                    capacity = 999999
                    VDF_fftt1 = one_line_df.iloc[k + 1].arrival_time - one_line_df.iloc[k].arrival_time
                    # minutes
                    VDF_cap1 = lanes * capacity
                    free_speed = (length / (VDF_fftt1 + 0.001)) * 60
                    # (miles/minutes)*60 = miles/hour
                    # free_speed = ((length / 1000) / (VDF_fftt1 + 0.001)) * 60
                    # (kilometers/minutes)*60 = kilometer/hour
                    VDF_alpha1 = 0.15
                    VDF_beta1 = 4
                    VDF_penalty1 = 0
                    cost = 0
                    geometry = 'LINESTRING (' + str(from_node_lon) + ' ' + str(from_node_lat) + ', ' + \
                            str(to_node_lon) + ' ' + str(to_node_lat) + ')'
                    geometry_id = (one_line_df.iloc[k].shape_id if pd.notnull(one_line_df.iloc[k].shape_id) else '0')
                    agency_name = one_line_df.agency_name[0]
                    allowed_use = allowed_use_function(one_line_df.iloc[k].route_type)
                    stop_sequence = one_line_df.iloc[k].stop_sequence
                    directed_service_id = one_line_df.iloc[k].directed_service_id
                    link_list = [link_id, from_node_id, to_node_id, facility_type, dir_flag, directed_route_id,
                                link_type, link_type_name, length, lanes, capacity, free_speed, cost,
                                VDF_fftt1, VDF_cap1, VDF_alpha1, VDF_beta1, VDF_penalty1, geometry, allowed_use,
                                agency_name,
                                stop_sequence, directed_service_id, geometry_id]
                    one_agency_link_list.append(link_list)
                    number_of_route_links += 1
                    if number_of_route_links % 50 == 0:
                        time_end = time.time()
                        print('convert ', number_of_route_links,
                            'service links successfully...', 'using time', time_end - time_start, 's')

        print("2. start creating boarding links from stations to their passing routes...")
        """boarding_links"""
        service_node_df = node_df[node_df.node_id != node_df.physical_node_id]
        #  select service node from node_df
        service_node_df = service_node_df.reset_index()
        number_of_sta2route_links = 0
        for iter, row in service_node_df.iterrows():
            link_id = 1000000 + number_of_route_links + number_of_sta2route_links
            from_node_id = row.physical_node_id
            to_node_id = row.node_id
            facility_type = convert_route_type_to_link_type(row.route_type)
            dir_flag = 1
            directed_route_id = row.directed_route_id
            link_type = 2
            link_type_name = 'boarding_links'
            to_node_lon = row.x_coord
            to_node_lat = row.y_coord
            from_node_lon = node_lon_dict[row.physical_node_id]
            from_node_lat = node_lat_dict[row.physical_node_id]
            length = calculate_distance_from_geometry(from_node_lon, from_node_lat, to_node_lon, to_node_lat, 'mile')
            free_speed = 2
            lanes = 1
            capacity = 999999
            VDF_cap1 = lanes * capacity
            VDF_alpha1 = 0.15
            VDF_beta1 = 4
            VDF_penalty1 = 0
            cost = 0
            stop_sequence = -1
            directed_service_id = directed_service_dict[to_node_id]
            geometry = 'LINESTRING (' + str(from_node_lon) + ' ' + str(from_node_lat) + ', ' + \
                    str(to_node_lon) + ' ' + str(to_node_lat) + ')'
            geometry_id = 0          
            agency_name = row.agency_name
            allowed_use = allowed_use_function(row.route_type)

            # inbound links (boarding)

            VDF_fftt1 = \
                0.5 * ((self.period_end_time - self.period_start_time) / frequency_dict[row.directed_service_id])
            VDF_fftt1 = min(VDF_fftt1, 10)
            # waiting time at a station is 10 minutes at most
            geometry = 'LINESTRING (' + str(to_node_lon) + ' ' + str(to_node_lat) + ', ' + \
                    str(from_node_lon) + ' ' + str(from_node_lat) + ')'
            # inbound link is average waiting time derived from frequency
            link_list_inbound = [link_id, from_node_id, to_node_id, facility_type, dir_flag, directed_route_id,
                                link_type, link_type_name, length, lanes, capacity, free_speed, cost,
                                VDF_fftt1, VDF_cap1, VDF_alpha1, VDF_beta1, VDF_penalty1, geometry, allowed_use,
                                agency_name,
                                stop_sequence, directed_service_id, geometry_id]
            number_of_sta2route_links += 1

            # outbound links (boarding)
            link_id = 1000000 + number_of_route_links + number_of_sta2route_links
            VDF_fftt1 = 1  # (length / free_speed) * 60
            #  the time of outbound time
            link_list_outbound = [link_id, to_node_id, from_node_id, facility_type, dir_flag, directed_route_id,
                                link_type, link_type_name, length, lanes, capacity, free_speed, cost,
                                VDF_fftt1, VDF_cap1, VDF_alpha1, VDF_beta1, VDF_penalty1, geometry, allowed_use,
                                agency_name,
                                stop_sequence, directed_service_id, geometry_id]
            one_agency_link_list.append(link_list_inbound)
            one_agency_link_list.append(link_list_outbound)
            number_of_sta2route_links += 1
            #  one inbound link and one outbound link
            if number_of_sta2route_links % 50 == 0:
                time_end = time.time()
                print('convert ', number_of_sta2route_links,
                    'boarding links successfully...', 'using time', time_end - time_start, 's')

        return one_agency_link_list

    @func_running_time
    def create_transferring_links(self, all_node_df: pd.DataFrame, all_link_list: list) -> list:

        physical_node_df = all_node_df[all_node_df.node_id == all_node_df.physical_node_id]
        physical_node_df = physical_node_df.reset_index()
        number_of_transferring_links = 0
        time_start = time.time()

        for i in range(len(physical_node_df)):
            ref_x = physical_node_df.iloc[i].x_coord
            ref_y = physical_node_df.iloc[i].y_coord

            mask1 = physical_node_df.x_coord >= (ref_x - 0.003)
            mask2 = physical_node_df.x_coord <= (ref_x + 0.003)
            mask3 = physical_node_df.y_coord >= (ref_y - 0.003)
            mask4 = physical_node_df.y_coord <= (ref_y + 0.003)

            neighboring_node_df = physical_node_df[mask1 & mask2 & mask3 & mask4]

            # neighboring_node_df = physical_node_df[(physical_node_df.x_coord >= (ref_x - 0.003)) &
            #                                        (physical_node_df.x_coord <= (ref_x + 0.003))]
            # neighboring_node_df = neighboring_node_df[(neighboring_node_df.y_coord >= (ref_y - 0.003)) &
            #                                           (neighboring_node_df.y_coord <= (ref_y + 0.003))]

            labeled_list = []
            count = 0
            for j in range(len(neighboring_node_df)):
                if count >= 10:
                    break
                if (physical_node_df.iloc[i].route_id, physical_node_df.iloc[i].agency_name) == \
                        (neighboring_node_df.iloc[j].route_id, neighboring_node_df.iloc[j].agency_name):
                    continue
                from_node_lon = float(physical_node_df.iloc[i].x_coord)
                from_node_lat = float(physical_node_df.iloc[i].y_coord)
                to_node_lon = float(neighboring_node_df.iloc[j].x_coord)
                to_node_lat = float(neighboring_node_df.iloc[j].y_coord)
                # calculate length in meters for threshold
                length = calculate_distance_from_geometry(from_node_lon, from_node_lat, to_node_lon, to_node_lat, 'meter')
                if (length > 321.869) | (length < 1):
                    continue
                if (neighboring_node_df.iloc[j].route_id, neighboring_node_df.iloc[j].agency_name) in labeled_list:
                    continue
                count += 1
                labeled_list.append((neighboring_node_df.iloc[j].route_id, neighboring_node_df.iloc[j].agency_name))
                # consider only one stops of another route
                # transferring 1
                # calculate length in miles for output
                length = calculate_distance_from_geometry(from_node_lon, from_node_lat, to_node_lon, to_node_lat, 'mile')
                #  print('transferring link length =', length)
                link_id = number_of_transferring_links + 1
                from_node_id = physical_node_df.iloc[i].node_id
                to_node_id = neighboring_node_df.iloc[j].node_id
                facility_type = 'sta2sta'
                dir_flag = 1
                directed_route_id = -1
                link_type = 3
                link_type_name = 'transferring_links'
                lanes = 1
                capacity = 999999
                VDF_fftt1 = (length / 1000) / 1
                VDF_cap1 = lanes * capacity
                free_speed = 2  # Volpe edited from 1 to 2
                # 1 kilo/hour
                VDF_alpha1 = 0.15
                VDF_beta1 = 4
                VDF_penalty1 = transferring_penalty(physical_node_df.iloc[i].node_type, neighboring_node_df.iloc[j].node_type)
                # penalty of transferring
                cost = 60
                geometry = 'LINESTRING (' + str(from_node_lon) + ' ' + str(from_node_lat) + ', ' + \
                        str(to_node_lon) + ' ' + str(to_node_lat) + ')'
                geometry_id = 0
                agency_name = ""
                allowed_use = allowed_use_transferring(physical_node_df.iloc[i].node_type, neighboring_node_df.iloc[j].node_type)
                stop_sequence = ""
                directed_service_id = ""
                link_list = [link_id, from_node_id, to_node_id, facility_type, dir_flag, directed_route_id,
                            link_type, link_type_name, length, lanes, capacity, free_speed, cost,
                            VDF_fftt1, VDF_cap1, VDF_alpha1, VDF_beta1, VDF_penalty1, geometry, allowed_use, agency_name,
                            stop_sequence, directed_service_id, geometry_id]
                all_link_list.append(link_list)
                # transferring 2
                number_of_transferring_links += 1
                geometry = 'LINESTRING (' + str(to_node_lon) + ' ' + str(to_node_lat) + ', ' + \
                        str(from_node_lon) + ' ' + str(from_node_lat) + ')'
                link_id = number_of_transferring_links + 1
                link_list = [link_id, to_node_id, from_node_id, facility_type, dir_flag, directed_route_id,
                            link_type, link_type_name, length, lanes, capacity, free_speed, cost,
                            VDF_fftt1, VDF_cap1, VDF_alpha1, VDF_beta1, VDF_penalty1, geometry, allowed_use, agency_name,
                            stop_sequence, directed_service_id, geometry_id]
                all_link_list.append(link_list)
                number_of_transferring_links += 1
                if number_of_transferring_links % 50 == 0:
                    time_end = time.time()
                    print('convert ', number_of_transferring_links,
                        'transferring links successfully...', 'using time', time_end - time_start, 's')

        return all_link_list

    def main(self, isSaveToCSV: bool = True):
        #  step 1. reading data
        stop_df, route_df, trip_df, trip_route_df, stop_time_df, directed_trip_route_stop_time_df = self.read_gtfs_data(self.gtfs_dir)

        #  directed_trip_route_stop_time_df.to_csv(gtfs_folder_list[i] + '/timetable.csv', index=False)
        #  directed_trip_route_stop_time_df = pd.read_csv(gtfs_folder_list[i] + '/timetable.csv')

        # step 2. create nodes
        node_df = self.create_nodes(directed_trip_route_stop_time_df)
        node_df.reset_index(inplace=True)
        node_df = node_df.drop(['index'], axis=1)

        # step 3. create links
        all_link_list = []
        all_link_list = self.create_service_boarding_links(directed_trip_route_stop_time_df, node_df, all_link_list)

        # transferring links
        all_link_list = self.create_transferring_links(node_df, all_link_list)

        all_link_df = pd.DataFrame(all_link_list)
        all_link_df.rename(columns={0: 'link_id',
                                    1: 'from_node_id',
                                    2: 'to_node_id',
                                    3: 'facility_type',
                                    4: 'dir_flag',
                                    5: 'directed_route_id',
                                    6: 'link_type',
                                    7: 'link_type_name',
                                    8: 'length',
                                    9: 'lanes',
                                    10: 'capacity',
                                    11: 'free_speed',
                                    12: 'cost',
                                    13: 'VDF_fftt1',
                                    14: 'VDF_cap1',
                                    15: 'VDF_alpha1',
                                    16: 'VDF_beta1',
                                    17: 'VDF_penalty1',
                                    18: 'geometry',
                                    19: 'VDF_allowed_uses1',
                                    20: 'agency_name',
                                    21: 'stop_sequence',
                                    22: 'directed_service_id',
                                    23: 'geometry_id'}, inplace=True)
        all_link_df = all_link_df.drop_duplicates(
                    subset=['from_node_id', 'to_node_id'],
                    keep='last').reset_index(drop=True)

        # step 4. save node and link data
        # create node and link result path
        if isSaveToCSV:
            node_result_file = path2linux(os.path.join(self.gtfs_result_dir, "node.csv"))
            link_result_file = path2linux(os.path.join(self.gtfs_result_dir, "link.csv"))

            # validate result file path exist or not, if exist, create new file wit _1 suffix
            node_result_file = validate_filename(node_result_file)
            link_result_file = validate_filename(link_result_file)

            #  zone_df = pd.read_csv('zone.csv')
            #  source_node_df = pd.read_csv('source_node.csv')
            #  node_df = pd.concat([zone_df, all_node_df])
            node_df.to_csv(node_result_file, index=False)
            all_link_df.to_csv(link_result_file, index=False)
            print(f"Info: successfully converted gtfs data to node and link data:\n  {node_result_file}, {link_result_file}")
        else:
            print("Info: successfully converted gtfs data to node and link and return node and link dataframe")

        return node_df, all_link_df


if __name__ == '__main__':

    # PARSE ARGS
    # ----------------------------------------------------------------------------------------------

    program_description = 'GTFS2GMNS Helper Tool'

    help_text = """
    The command-line input expected for this script is as follows:

    TheFilePathOfThisScript GTFSFilePath
    """

    parser = argparse.ArgumentParser(description=program_description, usage=help_text)

    parser.add_argument("gtfs_folder", help="The full path to the GTFS input folder", type=str)

    if len(sys.argv) == 2:
        args = parser.parse_args()
    else:
        parser.print_help()
        sys.exit()

    gtfs_dir = args.gtfs_folder
    output_gmns_path = args.gtfs_folder
    time_period = '0700_0800'

    node_df, link_df = GTFS2GMNS(gtfs_dir, output_gmns_path, time_period).main()

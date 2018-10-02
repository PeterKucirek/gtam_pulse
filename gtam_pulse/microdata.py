from typing import List, Union, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import numexpr as ne
import attr

from balsa import LinkedDataFrame, read_mdf, peek_mdf
from balsa.models.analyses import distance_matrix

from .constants import TimeFormat


class PulseData:

    # region Attributes

    _households: LinkedDataFrame
    _persons: LinkedDataFrame
    _trips: LinkedDataFrame
    _trip_modes: LinkedDataFrame
    _station_trips: LinkedDataFrame
    _passenger_trips: LinkedDataFrame
    _zone_attributes: pd.DataFrame
    _impedances: pd.DataFrame

    _household_spec = {'HouseholdID': 'i4', 'Zone': 'i4', 'ExpansionFactor': 'f4', 'DwellingType': 'category',
                       'NumberOfPersons': 'i2', 'NumberOfVehicles': 'i2'}

    _persons_spec = {
        'household_id': 'i4', 'person_id': 'i1', 'age': 'i1', 'sex': 'category', 'license': bool, 'transit_pass': bool,
        'employment_status': 'category', 'occupation': 'category', 'free_parking': bool, 'student_status': 'category',
        'work_zone': 'i2', 'school_zone': 'i2', 'weight': 'f4'
    }

    _trips_spec = {
        'household_id': 'i4', 'person_id': 'i1', 'trip_id': 'i1', 'o_zone': 'i2', 'o_act': 'category', 'd_zone': 'i2',
        'd_act': 'category', 'weight': 'f4'
    }

    _tmodes_spec = {
        'household_id': 'i4', 'person_id': 'i1', 'trip_id': 'i1', 'mode': 'category', 'o_depart': np.object_,
        'd_arrive': np.object_, 'weight': 'i2'
    }

    _tstation_spec = {
        'household_id': 'i4', 'person_id': 'i1', 'trip_id': 'i1', 'station': np.object_, 'direction': 'category',
        'weight': 'i2'
    }

    _passenger_spec = {
        'household_id': 'i4', 'passenger_id': 'i1', 'driver_id': 'i1', 'driver_trip_idstation': 'i4', 'weight': 'i2'
    }

    # endregion
    # region Properties

    @property
    def households_loaded(self): return self._households is not None

    @property
    def persons_loaded(self):
        return self._persons is not None

    @property
    def trips_loaded(self):
        return self._trips is not None

    @property
    def trip_modes_loaded(self):
        return self._trip_modes is not None

    @property
    def station_trips_loaded(self):
        return self._station_trips is not None

    @property
    def passenger_trips_loaded(self):
        return self._passenger_trips is not None

    @property
    def zones_loaded(self): return self._zone_attributes is not None

    @property
    def households(self) -> LinkedDataFrame: return self._households

    @property
    def persons(self) -> LinkedDataFrame: return self._persons

    @property
    def trips(self) -> LinkedDataFrame: return self._trips

    @property
    def trip_modes(self) -> LinkedDataFrame: return self._trip_modes

    @property
    def station_trips(self) -> LinkedDataFrame: return self._station_trips

    @property
    def passenger_trips(self) -> LinkedDataFrame: return self._passenger_trips

    @property
    def impedances(self) -> pd.DataFrame: return self._impedances

    @property
    def zone_attributes(self) -> pd.DataFrame: return self._zone_attributes

    # endregion
    # region IO

    @staticmethod
    def load_from_run(run_folder: Path, *, hh_file: Path = None, time_format: TimeFormat = TimeFormat.MINUTE_DELTA,
                      zones_file: Path=None, coord_unit: float=0.001) -> 'PulseData':
        if not isinstance(run_folder, Path): run_folder = Path(run_folder)
        assert run_folder.exists()
        assert run_folder.is_dir()

        if hh_file is None:
            hh_file = run_folder / "Households.csv"
        elif not isinstance(hh_file, Path): hh_file = Path(hh_file)
        assert hh_file.exists()

        def _prep_file(name):
            uncompressed = run_folder / name
            compressed = run_folder / (name + '.gz')

            if uncompressed.exists(): return uncompressed
            if compressed.exists(): return compressed

            compressed, uncompressed = str(compressed.name) + '.gz', str(uncompressed.name)

            raise FileExistsError(f"Could not find '{uncompressed}' or '{uncompressed}' in folder '{str(run_folder)}'")

        persons_fp = _prep_file('persons.csv')
        trips_fp = _prep_file('trips.csv')
        tmodes_fp = _prep_file('trip_modes.csv')
        tstations_fp = _prep_file('trip_stations.csv')

        try:
            tpass_fp = _prep_file('facilitate_passenger.csv')
        except FileExistsError:
            # Some older versions have this
            tpass_fp = None

        data = PulseData()
        data._load_tables(hh_fp=hh_file, person_fp=persons_fp, trips_fp=trips_fp, tmodes_fp=tmodes_fp,
                          tstations_fp=tstations_fp, tpass_fp=tpass_fp, zones_fp=zones_file)

        if data.trip_modes_loaded and time_format is not None:
            data._classify_times(time_format=time_format)

        if data.zones_loaded:
            data._init_imped(coord_unit=coord_unit)

        data._verify_integrity()
        data._link_all()

        return data

    def _load_tables(self, *, hh_fp: Path=None, person_fp: Path=None, trips_fp: Path=None, tmodes_fp: Path=None,
                     tstations_fp: Path=None, tpass_fp: Path=None, zones_fp: Path=None):

        if hh_fp is not None:
            table = LinkedDataFrame(pd.read_csv(hh_fp, dtype=self._household_spec))
            print(f"Loaded {len(table)} households")
            self._households = table

        if person_fp is not None:
            table = LinkedDataFrame(pd.read_csv(person_fp, dtype=self._persons_spec))
            print(f"Loaded {len(table)} persons")
            self._persons = table

        if trips_fp is not None:
            table = LinkedDataFrame(pd.read_csv(trips_fp, dtype=self._trips_spec))
            print(f"Loaded {len(table)} trips")
            self._trips = table

        if tmodes_fp is not None:
            table = LinkedDataFrame(pd.read_csv(tmodes_fp, dtype=self._tmodes_spec))
            print(f"Loaded {len(table)} trip-modes")
            self._trip_modes = table

        if tstations_fp is not None:
            table = LinkedDataFrame(pd.read_csv(tstations_fp, dtype=self._tmodes_spec))
            print(f"Loaded {len(table)} station trips")
            self._station_trips = table

        if tpass_fp is not None:
            table = LinkedDataFrame(pd.read_csv(tpass_fp, dtype=self._passenger_spec))
            print(f"Loaded {len(table)} facilitate passenger trips")
            self._passenger_trips = table

        if zones_fp is not None:
            table = pd.read_csv(zones_fp, index_col=0)
            print(f"Loaded attributes for {len(table)} zones")
            self._zone_attributes = table

    def add_skim(self, skim_path: Path, name: str=None):
        assert self.zones_loaded, "Cannot add a skim unless zone coordinates are loaded"
        assert skim_path.exists(), f"Could not find file '{str(skim_path)}'"
        assert self._zone_attributes.index.equals(peek_mdf(skim_path)), "Zone systems not compatible"

        matrix_series = read_mdf(skim_path, raw=False, tall=True)
        assert len(matrix_series) == len(self.impedances)

        if name is None: name = skim_path.stem

        self.impedances[name] = matrix_series

    # endregion
    # region Derived attributes

    def _classify_times(self, time_format: TimeFormat):
        assert self.trip_modes_loaded

        print("Parsing time formats")

        table = self.trip_modes

        table['o_depart_hr'] = self._convert_time_to_hours(table.o_depart, time_format)
        print("Parsed o_depart")

        table['d_arrive_hr'] = self._convert_time_to_hours(table.d_arrive, time_format)
        print("Parsed d_arrive")

        table['time_period'] = self._classify_time_period(table.o_depart_hr)
        print("Classified time periods")

    @staticmethod
    def _convert_time_to_hours(column: pd.Series, time_format: TimeFormat) -> pd.Series:
        if time_format == time_format.MINUTE_DELTA:
            return PulseData._floordiv_minutes(column)
        elif time_format == time_format.COLON_SEP:
            return PulseData._convert_text_to_datetime(column)
        else:
            raise NotImplementedError(time_format)

    @staticmethod
    def _convert_text_to_datetime(s: pd.Series) -> pd.Series:
        colon_count = s.str.count(':')
        filtr = colon_count == 1

        new_time: pd.Series = s.copy()
        new_time.loc[filtr] += ':00'

        filtr = new_time.str.contains('-')
        if filtr.sum() > 0:
            new_time.loc[filtr] = "0:00:00"
            print(f"Found {filtr.sum()} cells with negative time. These have been corrected to 0:00:00")

        time_table = new_time.str.split(':', expand=True).astype(np.int8)
        hours = time_table.iloc[:, 0]

        return hours

    @staticmethod
    def _floordiv_minutes(column: pd.Series) -> pd.Series:
        converted = column.astype(np.float64)
        return (converted // 60).astype(np.int32)

    @staticmethod
    def _classify_time_period(start_hour: pd.Series) -> pd.Series:
        new_col = pd.Series('ON', index=start_hour.index)

        new_col.loc[start_hour.isin({6, 7, 8})] = 'AM'
        new_col.loc[start_hour.isin({9, 10, 11, 12, 13, 14})] = 'MD'
        new_col.loc[start_hour.isin({15, 16, 17, 18})] = 'PM'
        new_col.loc[start_hour.isin({19, 20, 21, 22, 23})] = 'EV'

        return new_col.astype('category')

    def _init_imped(self, coord_unit: float):
        assert self.zones_loaded

        print("Initializing impedance matrices from zone coordinates")

        methods = ['manhattan', 'euclidean']

        imped = {
            key: distance_matrix(self._zone_attributes.X, self._zone_attributes.Y, method=key, coord_unit=coord_unit,
                                 tall=True)
            for key in methods
        }
        self._impedances = pd.DataFrame(imped)
        self._impedances.index.names = ['o', 'd']

    def _verify_integrity(self):
        if self.households_loaded and self.persons_loaded:
            hh_sizes = self.persons.household_id.value_counts(dropna=False)
            isin = hh_sizes.index.isin(self.households.HouseholdID)
            n_homeless = hh_sizes.loc[~isin].sum()
            if n_homeless > 0:
                raise RuntimeError("Found %s persons with invalid or missing household IDs" % n_homeless)

    def _link_all(self):

        if self.households_loaded and self.persons_loaded:
            self.persons.link_to(self.households, 'household', on_self='household_id', on_other='HouseholdID')
            self.households.link_to(self.persons, 'persons', on_self='HouseholdID', on_other='household_id')

            self.persons['home_zone'] = self.persons.household.Zone

        if self.trips_loaded and self.persons_loaded:
            self.trips.link_to(self.persons, 'person', on=['household_id', 'person_id'])

        if self.trip_modes_loaded and self.trips_loaded:
            self.trip_modes.link_to(self.trips, 'trip', on=['household_id', 'person_id', 'trip_id'])
            self.trips.link_to(self.trip_modes, 'modes', on=['household_id', 'person_id', 'trip_id'])

            if self.persons_loaded:
                self.trip_modes.link_to(self.persons, 'person', on=['household_id', 'person_id'])
            if self.households_loaded:
                self.trip_modes.link_to(self.households, 'household', on_self='household_id', on_other='HouseholdID')

        if self.station_trips_loaded and self.trips_loaded and self.trip_modes_loaded:
            self.station_trips.link_to(self.trips, 'trip', on=['household_id', 'person_id', 'trip_id'])
            self.station_trips.link_to(self.trip_modes, 'mode_data', on=['household_id', 'person_id', 'trip_id'])

        if self.zones_loaded:
            if self.trips_loaded:
                self.trips.link_to(self.impedances, 'imped', on_self=['o_zone', 'd_zone'])
            if self.persons_loaded:
                self.persons.link_to(self.impedances, 'work_imped', on_self=['home_zone', 'work_zone'])
                self.persons.link_to(self.impedances, 'school_imped', on_self=['home_zone', 'school_zone'])

    def _check_zones(self):
        if self.households_loaded:
            self._check_column_zones(self.households.Zone, 'household')

        if self.persons_loaded:
            work_zones = self.persons.loc[self.persons.employment_status != 'O', 'work_zone']
            self._check_column_zones(work_zones, 'worker')

            school_zones = self.persons.loc[self.persons.student_status != 'O', 'school_zone']
            self._check_column_zones(school_zones, 'student')

        if self.trips_loaded:
            self._check_column_zones(self.trips.o_zone, 'trip')
            self._check_column_zones(self.trips.d_zone, 'trip')

    def _check_column_zones(self, col: pd.Series, item_name: str):
        zone_counts = col.value_counts()
        extra_zones = zone_counts.index.difference(self._zone_attributes.index)
        extra_zones = zone_counts.loc[extra_zones]
        if len(extra_zones) > 0:
            raise AssertionError(f"Found {len(extra_zones)} extra zones, for {extra_zones.sum()} {item_name}s")

    def add_ensemble(self, name: str, definition: pd.Series, missing_value: int=0):
        assert np.all(definition.index.isin(self._zone_attributes.index))

        if self.households_loaded:
            assert name not in self.households
            self.households[name] = definition.reindex(self.households.Zone, fill_value=missing_value).values
            self.households[name] = self.households[name].astype('category')

        if self.persons_loaded:
            assert name not in self.persons
            self.persons[f"work_{name}"] = definition.reindex(self.persons.work_zone, fill_value=missing_value).values
            self.persons[f"work_{name}"] = self.persons[f"work_{name}"].astype('category')

            self.persons[f"school_{name}"] = definition.reindex(
                self.persons.school_zone, fill_value=missing_value).values
            self.persons[f"school_{name}"] = self.persons[f"school_{name}"].astype('category')

            self.persons[f"home_{name}"] = definition.reindex(self.persons.home_zone, fill_value=missing_value).values

        if self.trips_loaded:
            assert name not in self.trips
            self.trips[f"o_{name}"] = definition.reindex(self.trips.o_zone, fill_value=missing_value).values
            self.trips[f"o_{name}"] = self.trips[f"o_{name}"].astype('category')

            self.trips[f"d_{name}"] = definition.reindex(self.trips.d_zone, fill_value=missing_value).values
            self.trips[f"d_{name}"] = self.trips[f"d_{name}"].astype('category')

    def classify_trips(self):
        assert self.trips_loaded

        print("Classifying trip directions")
        orig_is_home = self.trips.o_act == 'Home'
        dest_is_home = self.trips.d_act == 'Home'
        self.trips['direction'] = 'NHB'
        self.trips.loc[orig_is_home & dest_is_home, 'direction'] = 'H2H'
        self.trips.loc[orig_is_home & ~dest_is_home, 'direction'] = 'OUT'
        self.trips.loc[~orig_is_home & dest_is_home, 'direction'] = 'IN'

        self.trips['direction'] = self.trips.direction.astype('category')

    def reweight_trips(self):
        assert self.trips_loaded
        trips = self.trips

        if self.trip_modes_loaded:
            trip_modes = self.trip_modes
            trips['repetitions'] = trips.modes.sum('weight')
            trip_modes['full_weight'] = trip_modes.weight / trip_modes.trip.repetitions * trip_modes.trip.weight

        # TODO: Reweight other tables (like stations and facilitate passengers)

    # endregion

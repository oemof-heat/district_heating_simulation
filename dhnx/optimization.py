# -*- coding: utf-8

"""
This module is designed to hold optimization model implementations. The
implementation makes use of oemof-solph.

This file is part of project dhnx (). It's copyrighted
by the contributors recorded in the version control history of the file,
available from its original location:

SPDX-License-Identifier: MIT
"""

import os
import logging
import pandas as pd
import oemof.solph as solph
from oemof.solph import helpers
from oemof.tools import logger

from .optimization_modules.dhs_nodes import add_nodes_dhs, add_nodes_houses
from .model import OperationOptimizationModel, InvestOptimizationModel


class OemofOperationOptimizationModel(OperationOptimizationModel):
    r"""
    Implementation of an operation optimization model using oemof-solph.
    """
    def __init__(self, thermal_network):
        super().__init__(thermal_network)
        self.results = {}

    def setup(self):
        pass

    def solve(self):
        pass

    def get_results(self):
        return self.results


class OemofInvestOptimizationModel(InvestOptimizationModel):
    """
    Implementation of an invest optimization model using oemof-solph.

    ...

    Attributes
    ----------
    settings : dict
        Dictionary holding the optimisation settings. See .
    invest_options : dict
        Dictionary holding the investment options for the district heating system.
    nodes : list
        Empty list for collecting all oemof.solph nodes.
    buses : dict
        Empty dictionary for collecting all oemof.solph.Buses of the energy system.
    es : oemof.solph.EnergySystem
        Empty oemof.solph.EnergySystem.
    om : oemof.solph.Model
        Attribute, which will be the oemof.solph.Model for optimisation.
    oemof_flow_attr : set
        Possible flow attributes, which can be used additionally:
        {'nominal_value', 'min', 'max', 'variable_costs', 'fix'}
    results : dict
        Empty dictionary for the results.

    Methods
    -------
    check_input():
        Performs checks on the input data.
    complete_exist_data():
        Sets the investment status for the results dataframe of the pipes.
    get_pipe_data():
        Adds heat loss and investment costs to pipes dataframe.
    setup_oemof_es():
        The energy system *es* is build.
    setup():
        Calls *check_input()*, *complete_exist_data()*, *get_pipe_data()*, and *setup_oemof_es()*.

    """
    def __init__(self, thermal_network, settings, investment_options):

        self.settings = settings
        self.invest_options = investment_options
        self.nodes = []  # list of all nodes
        self.buses = {}  # dict of all buses
        self.es = solph.EnergySystem()
        self.om = None

        # list of possible oemof flow attributes, e.g. for producers source
        self.oemof_flow_attr = {'nominal_value', 'min', 'max',
                                'variable_costs', 'fix'}

        super().__init__(thermal_network)
        self.results = {}

    def check_input(self):
        """Check 1:

        Check and make sure, that the dtypes of the columns of the sequences
        and the indices (=ids) of the forks, pipes, producers and consumers
        are of type 'str'. (They need to be the same dtye.)

        Check 2:

        Firstly, it is checked, if there are any not-allowed connection in the *pipe* data.
        The following connections are not allowed:

          * consumer -> consumer
          * producer -> producer
          * producer -> consumer
          * consumer -> fork

        Secondly, it is checked, if a pipes goes to a consumer, which does not exist.

        An error is raised if one of these connection occurs.
        """

        # Check 1
        # make sure that all ids are of type str
        # sequences
        sequ_items = self.network.sequences.keys()
        for it in sequ_items:
            for v in self.network.sequences[it].values():
                v.columns.astype('str')

        # components
        for comp in ['pipes', 'consumers', 'producers', 'forks']:
            self.network.components[comp].index = \
                self.network.components[comp].index.astype('str')

        # Check 2

        ids_consumers = self.network.components['consumers'].index

        for p, q in self.network.components['pipes'].iterrows():

            if (q['from_node'].split('-')[0] == "consumers") and (
                    q['to_node'].split('-')[0] == "consumers"):

                raise ValueError(
                    ""
                    "Pipe id {} goes from consumer to consumer. This is not "
                    "allowed!".format(p))

            if (q['from_node'].split('-')[0] == "producers") and (
                    q['to_node'].split('-')[0] == "producers"):

                raise ValueError(
                    ""
                    "Pipe id {} goes from producers to producers. "
                    "This is not allowed!".format(p))

            if ((q['from_node'].split('-')[0] == "producers") and (
                    q['to_node'].split('-')[0] == "consumers")) or ((
                        q['from_node'].split('-')[0] == "consumers") and (
                            q['to_node'].split('-')[0] == "producers")):

                raise ValueError(
                    ""
                    "Pipe id {} goes from producers directly "
                    "to consumers, or vice versa. This is not allowed!"
                    "".format(p))

            if (q['from_node'].split('-')[0] == "forks") and (
                    q['to_node'].split('-')[0] == "consumers"):

                cons_id = q['to_node'].split('-')[1]

                if cons_id not in ids_consumers:
                    raise ValueError(
                        ""
                        "The consumer of pipe id {} does not exist!".format(p))

        pipe_to_cons_ids = list(self.network.components['pipes']['to_node'].values)
        pipe_to_cons_ids = [x.split('-')[1] for x in pipe_to_cons_ids
                            if x.split('-')[0] == 'consumers']

        for id in list(self.network.components['consumers'].index):
            if id not in pipe_to_cons_ids:
                raise ValueError(
                    "The consumer id {} has no connection the the grid!".format(id))

    def remove_inactive(self):
        """
        If the attribute active is present in any of the components
        columns, or in any the investment options tables,
        all rows with active == 0 are deleted, and the column active
        is deleted.
        """
        def clean_df(df):
            if 'active' in df.columns:
                v_new = df[df['active'] == 1].copy()
                v_new.drop('active', axis=1, inplace=True)
                df = v_new
            return df

        for k, v in self.network.components.items():
            self.network.components[k] = clean_df(v)

        pipes = self.invest_options['network']['pipes']
        self.invest_options['network']['pipes'] = clean_df(pipes)

        for node_typ in ['consumers', 'producers']:
            for k, v in self.invest_options[node_typ].items():
                self.invest_options[node_typ][k] = clean_df(v)

    def prepare_heat_demand(self):
        """
        This method performs the pre-processing of the heat demand data, depending on
        the given optimisation settings.

        - If attribute 'P_heat_max' not given at the consumers, the maximum heat demand
          is calculated from the timeseries and added the consumers table.
        - If the optimisation setting 'heat_demand' == scalar, the number of time steps
          of the optimisation is set to 1, and the 'P_heat_max' values are copied to the
          consumers heat flow sequences (which is always the input for the optimisation model).
        - The consumers heat flow sequences are multiplied by the simultaneity factor.
        - Finally, a sufficient length of the heat demand timeseries is checked.

        Returns
        -------
        Updated `.network.components['consumers']` and
        `.network.sequences['consumers']['heat_flow']`
        """
        def check_len_timeseries():
            """
            Check, if given number of timesteps of optimization exceeds the length
            of the given heat demand timeseries.
            """
            if self.settings['num_ts'] > \
                    len(self.network.sequences['consumers']['heat_flow'].index):
                raise ValueError(
                    'The length of the heat demand timeseries is not sufficient '
                    'for the given number of {} timesteps.'.format(
                        self.settings['num_ts']))

        # prepare heat data, whether global simultanity or timeseries
        if 'P_heat_max' not in list(
                self.network.components['consumers'].columns):
            df_max = self.network.sequences['consumers']['heat_flow'].max(). \
                to_frame(name='P_heat_max')

            self.network.components['consumers'] = \
                pd.concat([self.network.components['consumers'], df_max],
                          axis=1, join='outer', sort=False)

        # check, which optimization type should be performed
        if self.settings['heat_demand'] == 'scalar':
            # just single timestep optimization, overwrite previous!
            self.settings['num_ts'] = 1

            # new approach
            p_max = self.network.components['consumers']['P_heat_max']
            df_ts = pd.DataFrame(data=[p_max.values],
                                 columns=list(p_max.index),
                                 index=pd.Index([0], name='timestep'))

            # heat load is maximum heat load
            self.network.sequences['consumers']['heat_flow'] = df_ts

        # apply global simultaneity for demand series
        self.network.sequences['consumers']['heat_flow'] = \
            self.network.sequences['consumers']['heat_flow'] * \
            self.settings['simultaneity']

        check_len_timeseries()

    def check_existing(self):
        """
        Checks if the attributes `existing` and `hp_type` are given in the `pipes` table.
        If not, the attribute is added, and set to `None` / 0.

        Checks for all existing pipes, if the heatpipe type is given in the pipe type table
        `.invest_options['network']['pipes']`, and if the capacity is greater than zero.
        """

        # check whether there the 'existing' attribute is present at the pipes
        if 'existing' not in self.network.components['pipes'].columns:
            self.network.components['pipes']['existing'] = 0

        # create pipes attribute hp_type, if not in the table so far
        if 'hp_type' not in list(self.network.components['pipes'].columns):
            self.network.components['pipes']['hp_type'] = None

        edges = self.network.components['pipes']
        pipe_types = self.invest_options['network']['pipes']

        hp_list = list({x for x in edges['hp_type'].tolist()
                        if isinstance(x, str)})

        for hp in hp_list:
            if hp not in list(pipe_types['label_3']):
                raise ValueError(
                    "Existing heatpipe type {} is not in the list of "
                    "ACTIVE heatpipe investment options!".format(hp)
                )

        for r, c in edges[edges['existing'] == 1].iterrows():
            if c['capacity'] <= 0:
                raise ValueError(
                    "The `capacity` of the existing pipe with id {} must be greater than 0!"
                    "".format(r)
                )

    def setup_oemof_es(self):
        """The oemof solph energy system is initialised based on the settings,
         and filled with oemof-solph object:

         The oemof-solph objects of the *consumers* and *producers* are defined at the consumers
         and producers investment options.

         For the heating infrastructure, there is a *oemof.solph.Bus* added for every fork,
         and a pipe component for every pipe as defined in */network/pipes.csv*.
         """

        date_time_index = pd.date_range(self.settings['start_date'],
                                        periods=self.settings['num_ts'],
                                        freq=self.settings['frequence'])

        logger.define_logging(screen_level=logging.INFO)

        logging.info('Initialize the energy system')

        self.es = solph.EnergySystem(timeindex=date_time_index)

        logging.info('Create oemof objects')

        # add houses and generation
        for typ in ['consumers', 'producers']:
            self.nodes, self.buses = add_nodes_houses(
                self, self.settings, self.nodes, self.buses, typ)

        logging.info('Producers, Consumers Nodes appended.')

        # add heating infrastructure
        self.nodes, self.buses = add_nodes_dhs(self, self.settings, self.nodes,
                                               self.buses)
        logging.info('DHS Nodes appended.')

        # add nodes and flows to energy system
        self.es.add(*self.nodes)

        logging.info('Energysystem has been created')

        if self.settings['print_logging_info']:
            print("*********************************************************")
            print("The following objects have been created:")
            for n in self.es.nodes:
                oobj = \
                    str(type(n)).replace("<class 'oemof.solph.", "").replace("'>",
                                                                             "")
                print(oobj + ':', n.label)
            print("*********************************************************")

    def setup(self):
        """
        Calls *remove_inactive()* *check_input()*, *prepare_heat_demand()*,
        *complete_exist_data()*, and *setup_oemof_es()*.
        """

        # removes all rows with attribute active == 0 - if 'active given
        self.remove_inactive()

        # initial check of pipes connections
        self.check_input()

        # pre-processes the heat demand data depending on optimisation settings
        self.prepare_heat_demand()

        # check if existing pipes are given
        self.check_existing()

        # set up oemof energy system
        self.setup_oemof_es()

    def solve(self):
        """Builds the oemof.solph.Model of the energysystem *es*."""

        logging.info('Build the operational model')
        self.om = solph.Model(self.es)

        logging.info('Solve the optimization problem')
        self.om.solve(solver=self.settings['solver'],
                      solve_kwargs=self.settings['solve_kw'])

        if self.settings['write_lp_file']:
            filename = os.path.join(
                helpers.extend_basic_path('lp_files'), 'DHNx.lp')
            logging.info('Store lp-file in %s', filename)
            self.om.write(filename, io_options={'symbolic_solver_labels': True})

        self.es.results['main'] = solph.processing.results(self.om)
        self.es.results['meta'] = solph.processing.meta_results(self.om)

    def get_results_edges(self):
        """Postprocessing of the investment results of the pipes."""

        def get_invest_val(lab):

            res = self.es.results['main']

            outflow = [x for x in res.keys()
                       if x[1] is not None
                       if lab == str(x[0].label)]

            if len(outflow) > 1:
                print('Multiple IDs!')

            try:
                invest = res[outflow[0]]['scalars']['invest']
            except (KeyError, IndexError):
                try:
                    # that's in case of a one timestep optimisation due to
                    # an oemof bug in outputlib
                    invest = res[outflow[0]]['sequences']['invest'][0]
                except (KeyError, IndexError):
                    # this is in case there is no bi-directional heatpipe, e.g. at
                    # forks-consumers, producers-forks
                    invest = 0

            # the rounding is performed due to numerical issues
            return round(invest, 6)

        def get_invest_status(lab):

            res = self.es.results['main']

            outflow = [x for x in res.keys()
                       if x[1] is not None
                       if lab == str(x[0].label)]

            try:
                invest_status = res[outflow[0]]['scalars']['invest_status']
            except (KeyError, IndexError):
                try:
                    # that's in case of a one timestep optimisation due to
                    # an oemof bug in outputlib
                    invest_status = res[outflow[0]]['sequences']['invest_status'][0]
                except (KeyError, IndexError):
                    # this is in case there is no bi-directional heatpipe, e.g. at
                    # forks-consumers, producers-forks
                    invest_status = 0

            return invest_status

        def get_hp_results(p):

            hp_lab = p['label_3']
            label_base = 'infrastructure_' + 'heat_' + hp_lab + '_'

            # maybe slow approach with lambda function
            df[hp_lab + '.' + 'dir-1'] = df['from_node'] + '-' + df['to_node']
            df[hp_lab + '.' + 'size-1'] = df[hp_lab + '.' + 'dir-1'].apply(
                lambda x: get_invest_val(label_base + x))
            df[hp_lab + '.' + 'dir-2'] = df['to_node'] + '-' + df['from_node']
            df[hp_lab + '.' + 'size-2'] = df[hp_lab + '.' + 'dir-2'].apply(
                lambda x: get_invest_val(label_base + x))

            df[hp_lab + '.' + 'size'] = \
                df[[hp_lab + '.' + 'size-1', hp_lab + '.' + 'size-2']].max(axis=1)

            # get direction of pipes
            for r, c in df.iterrows():
                if c[hp_lab + '.' + 'size-1'] > c[hp_lab + '.' + 'size-2']:
                    df.at[r, hp_lab + '.direction'] = 1
                elif c[hp_lab + '.' + 'size-1'] < c[hp_lab + '.' + 'size-2']:
                    df.at[r, hp_lab + '.direction'] = -1
                else:
                    df.at[r, hp_lab + '.direction'] = 0

            if p['nonconvex']:
                df[hp_lab + '.' + 'status-1'] = df[hp_lab + '.' + 'dir-1'].apply(
                    lambda x: get_invest_status(label_base + x))
                df[hp_lab + '.' + 'status-2'] = df[hp_lab + '.' + 'dir-2'].apply(
                    lambda x: get_invest_status(label_base + x))
                df[hp_lab + '.' + 'status'] = \
                    df[[hp_lab + '.' + 'status-1', hp_lab + '.' + 'status-2']].max(axis=1)

                for r, c in df.iterrows():
                    if df.at[r, hp_lab + '.' + 'status-1'] + \
                            df.at[r, hp_lab + '.' + 'status-2'] > 1:
                        print(
                            "Investment status of pipe id {} is 1 for both dircetions!"
                            " This is not allowed!".format(r)
                        )
                    if (df.at[r, hp_lab + '.' + 'status-1'] == 1 and df.at[
                        r, hp_lab + '.' + 'size-1'] == 0) or\
                            (df.at[r, hp_lab + '.' + 'status-2'] == 1 and df.at[
                                r, hp_lab + '.' + 'size-2'] == 0):
                        print(
                            "Investment status of pipe id {} is 1, and capacity is 0!"
                            "What happend?!".format(r)
                        )

            return df

        def check_multi_dir_invest(hp_lab):

            df_double_invest = \
                df[(df[hp_lab + '.' + 'size-1'] > 0) & (df[hp_lab + '.' + 'size-2'] > 0)]

            if self.settings['print_logging_info']:
                print('***')
                if df_double_invest.empty:
                    print('There is NO investment in both directions at the'
                          'following pipes for "', hp_lab, '"')
                else:
                    print('There is an investment in both directions at the'
                          'following pipes for "', hp_lab, '":')
                    print('----------')
                    print(' id | from_node | to_node | size-1 | size-2 ')
                    print('============================================')
                    for r, c in df_double_invest.iterrows():
                        print(r, ' | ', c['from_node'], ' | ', c['to_node'],
                              ' | ', c[hp_lab + '.' + 'size-1'], ' | ',
                              c[hp_lab + '.' + 'size-2'], ' | ')
                    print('----------')

        def catch_up_results():

            def check_invest_label(hp_type, edge_id):
                """
                If there is already a heatpipe type, an error is raised,
                because only one investment type for each edges makes sense.
                """
                if isinstance(hp_type, str):
                    raise ValueError(
                        "Pipe id {} already has an investment > 0!".format(edge_id))

            df['hp_type'] = None
            df['capacity'] = float(0)
            df['direction'] = 0

            for ahp in active_hp:
                # p = df_hp[df_hp['label_3'] == ahp].squeeze()   # series of heatpipe
                for r, c in df.iterrows():
                    if c[ahp + '.size'] > 0:
                        check_invest_label(c['hp_type'], id)
                        df.at[r, 'hp_type'] = ahp
                        df.at[r, 'capacity'] = c[ahp + '.size']
                        df.at[r, 'direction'] = c[ahp + '.direction']

        def recalc_costs_losses():

            df['costs'] = float(0)
            df['losses'] = float(0)

            for r, c in df.iterrows():
                if c['capacity'] > 0:
                    hp_lab = c['hp_type']
                    # select row from heatpipe type table
                    hp_p = df_hp[df_hp['label_3'] == hp_lab].squeeze()
                    if hp_p['nonconvex'] == 1:
                        df.at[r, 'costs'] = c['length[m]'] * (
                            c['capacity'] * hp_p['capex_pipes'] + hp_p['fix_costs']
                        )
                        df.at[r, 'losses'] = c['length[m]'] * (
                            c['capacity'] * hp_p['l_factor'] + hp_p['l_factor_fix']
                        )
                    elif hp_p['nonconvex'] == 0:
                        df.at[r, 'costs'] = c['length[m]'] * c['capacity'] * hp_p['capex_pipes']
                        # Note, that a constant loss is possible also for convex
                        df.at[r, 'losses'] = c['length[m]'] * (
                            c['capacity'] * hp_p['l_factor'] + hp_p['l_factor_fix']
                        )

        # use pipes dataframe as base and add results as new columns to it
        df = self.network.components['pipes']

        # only select not existing pipes
        df = df[df['existing'] == 0].copy()

        # remove input data
        df = df[['from_node', 'to_node', 'length[m]']].copy()

        # putting the results of the investments in heatpipes to the pipes:
        df_hp = self.invest_options['network']['pipes']

        # list of active heat pipes
        active_hp = list(df_hp['label_3'].values)

        for hp in active_hp:
            hp_param = df_hp[df_hp['label_3'] == hp].squeeze()
            get_hp_results(hp_param)
            check_multi_dir_invest(hp)

        catch_up_results()

        recalc_costs_losses()

        return df[['from_node', 'to_node', 'length[m]', 'hp_type', 'capacity', 'direction',
                   'costs', 'losses']]


def optimize_operation(thermal_network):
    r"""
    Takes a thermal network and returns the result of
    the operational optimization.
    """
    model = OemofOperationOptimizationModel(thermal_network)

    model.solve()

    results = model.get_results()

    return results


def setup_optimise_investment(thermal_network, invest_options, settings=None):
    """
    Function for setting up the oemof solph operational Model.

    Parameters
    ----------
    thermal_network : ThermalNetwork
        See the ThermalNetwork class.
    settings : dict
        Dictionary holding the optimisation settings.
    invest_options : dict
        Dictionary holding the investment options for the district heating system.

    Returns
    -------
    model : oemof.solph.Model
        The oemof.solph.Model is build.

    """

    setting_default = {
        'heat_demand': 'scalar',
        'num_ts': 1,
        'time_res': 1,
        'start_date': '1/1/2018',
        'frequence': 'H',
        'solver': 'cbc',
        'solve_kw': {'tee': True},
        'simultaneity': 1,
        'bidirectional_pipes': False,
        'dump_path': None,
        'dump_name': 'dump.oemof',
        'print_logging_info': False,
        'write_lp_file': False,
    }

    if settings is not None:
        given_keys = [x for x in settings.keys()
                      if x in setting_default.keys()]

        for key in given_keys:
            setting_default[key] = settings[key]

    model = OemofInvestOptimizationModel(thermal_network, setting_default,
                                         invest_options)

    return model


def solve_optimisation_investment(model):
    """

    Parameters
    ----------
    model : oemof.solph.Model
        The oemof model, which is optimized.

    Returns
    -------
    dict : Results of optimisation. Contains:
        - 'oemof' : Complete "oemof" results of the energy system optimisation (.results['main']).
        - 'oemof_meta' : Meta results of oemof solph optimisation.
        - 'components' : 'pipes' : Investment results of pipes.
    """

    model.solve()

    if model.settings['dump_path'] is not None:
        my_es = model.es
        my_es.dump(dpath=model.settings['dump_path'], filename=model.settings['dump_name'])
        print('oemof Energysystem stored in "{}"'.format(model.settings['dump_path']))

    edges_results = model.get_results_edges()

    results = {'oemof': model.es.results['main'],
               'oemof_meta': model.es.results['meta'],
               'components': {'pipes': edges_results}}

    return results

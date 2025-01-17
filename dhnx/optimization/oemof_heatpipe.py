# -*- coding: utf-8

"""
This module is designed to hold optimization model implementations. The
implementation makes use of oemof-solph.

This file is part of project dhnx (). It's copyrighted
by the contributors recorded in the version control history of the file,
available from its original location:

SPDX-License-Identifier: MIT
"""
# pylint: skip-file
import warnings
from collections import namedtuple

from oemof.network import Transformer
from oemof.solph import Investment
from oemof.solph._plumbing import sequence
from pyomo.core.base.block import ScalarBlock
from pyomo.environ import Constraint
from pyomo.environ import NonNegativeReals
from pyomo.environ import Set
from pyomo.environ import Var


class Label(namedtuple('solph_label', ['tag1', 'tag2', 'tag3', 'tag4'])):
    __slots__ = ()

    def __str__(self):
        """The string is used within solph as an ID, so it hast to be unique"""
        return '_'.join(map(str, self._asdict().values()))


class HeatPipeline(Transformer):
    r"""A HeatPipeline represent a Pipeline in a district heating system.

    This is done by a Transformer with a constant energy loss independent of
    actual power, but dependent on the nominal power.
    The HeatPipeline is a single-input-single-output transformer.

    Parameters
    ----------
    length : float
        Length of HeatPipeline.
    heat_loss_factor : float
        Heat loss per length unit as fraction of the nominal power. Can also be
        defined by a series.


    Note
    ----
    This component is experimental. Use it with care. See also
    :py:class:`~oemof.solph.network.Transformer`.


    The following sets, variables, constraints and objective parts are created
     * :py:class:`~dhnx.optimization.oemof_heatpipe.HeatPipelineBlock` (if no
       Investment object present)
     * :py:class:`~dhnx.optimization.oemof_heatpipe.HeatPipelineInvestBlock`
      (ifInvestment object present)

    Examples
    --------
    # TODO : example

    """

    def __init__(
        self,
        inputs,
        outputs,
        label=None,
        heat_loss_factor=0,
        heat_loss_factor_fix=0,
    ):

        self.heat_loss_factor = sequence(heat_loss_factor)
        self.heat_loss_factor_fix = sequence(heat_loss_factor_fix)

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            label=label,
        )

        self._invest_group = False
        self._nonconvex_group = False
        self._demand_group = False

        if len(self.inputs) > 1 or len(self.outputs) > 2:
            if len(self.outputs) == 2:
                self._demand_group = True
            else:
                raise ValueError("Heatpipe must not have more than"
                                 " one input and two outputs!")

        for f in self.inputs.values():
            if f.nonconvex is not None:
                raise ValueError(
                    "Inputflow must not be of type NonConvexFlow!")

        for f in self.outputs.values():
            if f.nonconvex is not None:
                self._nonconvex_group = True

        self._check_flows_invest()

        if (self._nonconvex_group is True) and (self._invest_group is True):
            raise ValueError(
                "Either an investment OR a switchable heatloss can be set"
                " (NonConvexFlow)."
                " Remove the NonConvexFlow or drop "
                "the Investment attribute.")

        if self._invest_group is True:
            self._set_flows()
            o = list(self.outputs.keys())[0]
            if (self.heat_loss_factor_fix[0] > 0) \
                    and (self.outputs[o].investment.nonconvex is False):
                warnings.warn(
                    "Be careful! In case of a convex Investment "
                    "(Investment.nonconvex is False), the "
                    "'heat_loss_factor_fix' is considered, even though the "
                    "investment might be zero! => A simple sink could be "
                    "the results. Hopefully, you know what you are doing.")
        else:
            self._set_nominal_value()

    def _check_flows_invest(self):
        # for flow in self.inputs.values():
        #     if isinstance(flow.investment, Investment):
        #         raise ValueError(
        #             "The investment must be defined at the Outputflow!")

        for flow in self.outputs.values():
            if isinstance(flow.investment, Investment):
                self._invest_group = True

    def _set_flows(self):
        # sets the input flow to investment
        for flow in self.inputs.values():
            flow.investment = Investment()

    # set nominal values of in- and outflow equal in case of
    # invest_group = False
    def _set_nominal_value(self):
        i = list(self.inputs.keys())[0]
        o = list(self.outputs.keys())[0]
        if self.outputs[o].nominal_value is not None:
            self.inputs[i].nominal_value = \
                self.outputs[o].nominal_value
        elif self.inputs[i].nominal_value is not None:
            self.outputs[o].nominal_value = \
                self.inputs[i].nominal_value

    def constraint_group(self):
        if self._invest_group is True:
            return HeatPipelineInvestBlock
        return HeatPipelineBlock


class HeatPipelineBlock(ScalarBlock):  # pylint: disable=too-many-ancestors
    r"""Block representing a pipeline of a district heating system.
    :class:`~dhnx.optimization.oemof_heatpipe.HeatPipeline`

    **The following constraints are created:**

    # TODO : Check and fix equations!

    .. _HeatPipelineBlock-equations:

    .. math:: \dot{Q}_{out}(t) = \dot{Q}_{in}(t) \cdot
        \frac{\eta_{out}}{\eta_{in}} - \dot{Q}_{loss}(t)

    .. math:: \dot{Q}_{loss}(t) = f_{loss}(t) \cdot l \cdot \dot{Q}_{nominal}

    The symbols used are defined as follows
    (with Variables (V) and Parameters (P)):

    .. csv-table::
        :header: "symbol", "attribute", "type", "explanation"
        :widths: 1, 1, 1, 1

        ":math:`\dot{Q}_{out}(t)`", ":py:obj:`flow[n, o, t]`", "V", "Heat
        output"
        ":math:`\dot{Q}_{in}(t)`", ":py:obj:`flow[i, n, t]`", "V", "Heat input"
        ":math:`\dot{Q}_{loss}(t)`", ":py:obj:`heat_loss[n, t]`", "P", "Heat
        loss of heat pipeline"
        ":math:`\dot{Q}_{nominal}`", ":py:obj:`flows[n, o].nominal_value`", "
        P", "Nominal capacity of heating pipeline"
        ":math:`f_{loss}(t)`", ":py:obj:`heat_loss_factor`", "P", "Specific
        heat loss factor for pipeline"
        ":math:`l`", ":py:obj:`length`", "P", "Length of heating pipeline"


    """

    CONSTRAINT_GROUP = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create(self, group=None):
        """ Creates the linear constraint for the class:`Heatpipe`
        block.

        Parameters
        ----------
        group : list

        """
        if group is None:
            return None

        m = self.parent_block()

        self.HEATPIPES = Set(initialize=list(group))
        self.CONVEX_HEATPIPES = Set(
            initialize=[n for n in group
                        if n.outputs[list(n.outputs.keys())[0]].nonconvex
                        is None]
        )
        self.NONCONVEX_HEATPIPES = Set(
            initialize=[n for n in group
                        if n.outputs[list(n.outputs.keys())[0]].nonconvex
                        is not None])

        # Defining Variables
        self.heat_loss = Var(self.HEATPIPES, m.TIMESTEPS,
                             within=NonNegativeReals)

        def _heat_loss_rule_fix(block, n, t):
            """Rule definition for the heat loss depending on the nominal
            capacity for fix fix heat loss.
            """
            o = list(n.outputs.keys())[0]

            expr = 0
            expr += - block.heat_loss[n, t]
            expr +=\
                n.heat_loss_factor[t] * m.flows[n, o].nominal_value
            expr += n.heat_loss_factor_fix[t]
            return expr == 0

        self.heat_loss_equation_fix = Constraint(
            self.CONVEX_HEATPIPES, m.TIMESTEPS, rule=_heat_loss_rule_fix)

        def _heat_loss_rule_on_off(block, n, t):
            """Rule definition for the heat loss depending on the nominal
            capacity. Here, the losses can be "switched off".
            """
            o = list(n.outputs.keys())[0]

            expr = 0
            expr += - block.heat_loss[n, t]
            expr += (
                n.heat_loss_factor[t] * m.flows[n, o].nominal_value
                + n.heat_loss_factor_fix[t]
            ) * m.NonConvexFlowBlock.status[n, o, t]
            return expr == 0

        self.heat_loss_equation_on_off = Constraint(
            self.NONCONVEX_HEATPIPES, m.TIMESTEPS, rule=_heat_loss_rule_on_off)

        def _relation_rule(block, n, t):
            """Link input and output flow and subtract heat loss."""
            i = list(n.inputs.keys())[0]
            o = list(n.outputs.keys())[0]

            expr = 0
            try:  # oemof.solph<=0.5.0
                expr += - m.flow[n, o, t]
                expr += m.flow[i, n, t]
            except KeyError:  # oemof.solph>=0.5.1
                period = 0  # Periods are not (yet) supported in DHNx
                expr += - m.flow[n, o, period, t]
                expr += m.flow[i, n, period, t]

            expr += - block.heat_loss[n, t]
            return expr == 0

        self.relation = Constraint(self.HEATPIPES, m.TIMESTEPS,
                                   rule=_relation_rule)


class HeatPipelineInvestBlock(ScalarBlock):  # pylint: disable=too-many-ancestors
    r"""Block representing a pipeline of a district heating system.
    :class:`~oemof.solph.custom.HeatPipeline`

    **The following constraints are created:**

    # TODO : Check and fix equations!

    .. _HeatPipelineInvestBlock-equations:

    .. math:: \dot{Q}_{out}(t) = \dot{Q}_{in}(t) \cdot
        \frac{\eta_{out}}{\eta_{in}} - \dot{Q}_{loss}(t)

    .. math:: \dot{Q}_{loss}(t) = f_{loss}(t) \cdot l \cdot \dot{Q}_{nominal}

    The symbols used are defined as follows
    (with Variables (V) and Parameters (P)):

    .. csv-table::
        :header: "symbol", "attribute", "type", "explanation"
        :widths: 1, 1, 1, 1

        ":math:`\dot{Q}_{out}(t)`", ":py:obj:`flow[n, o, t]`", "V", "Heat
        output"
        ":math:`\dot{Q}_{in}(t)`", ":py:obj:`flow[i, n, t]`", "V", "Heat input"
        ":math:`\dot{Q}_{loss}(t)`", ":py:obj:`heat_loss[n, t]`", "V", "Heat
        loss of heat pipeline"
        ":math:`\dot{Q}_{nominal}`", ":py:obj:`flows[n, o].nominal_value`", "
        V", "Nominal capacity of heating pipeline"
        ":math:`f_{loss}(t)`", ":py:obj:`heat_loss_factor`", "P", "Specific
        heat loss factor for pipeline"
        ":math:`l`", ":py:obj:`length`", "P", "Length of heating pipeline"

    """

    CONSTRAINT_GROUP = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create(self, group=None):
        """ Creates the linear constraint for the class:`HeatPipeline`
        block.

        Parameters
        ----------
        group : list

        """
        if group is None:
            return None

        m = self.parent_block()

        # Defining Sets
        self.INVESTHEATPIPES = Set(initialize=list(group))
        self.CONVEX_INVESTHEATPIPES = Set(
            initialize=[n for n in group
                        if n.outputs[
                            list(n.outputs.keys())[0]
                        ].investment.nonconvex is False]
        )
        self.NONCONVEX_INVESTHEATPIPES = Set(
            initialize=[n for n in group
                        if n.outputs[
                            list(n.outputs.keys())[0]
                        ].investment.nonconvex is True]
        )

        self.INVESTHEATPIPES_NO_DEMAND = Set(
            initialize=[n for n in group if len(n.outputs.keys()) == 1])
        self.INVESTHEATPIPES_WITH_DEMAND = Set(
            initialize=[n for n in group if len(n.outputs.keys()) == 2])

        # Defining Variables
        self.heat_loss = Var(self.INVESTHEATPIPES, m.TIMESTEPS,
                             within=NonNegativeReals)

        def _heat_loss_rule_convex(block, n, t):
            """Rule definition for constraint to connect the installed capacity
            and the heat loss
            """
            expr = 0
            expr += - block.heat_loss[n, t]
            try:  # oemof.solph<=0.5.0
                expr += n.heat_loss_factor[t] * m.InvestmentFlowBlock.invest[
                    n, list(n.outputs.keys())[0],
                ]
            except KeyError:  # oemof.solph>=0.5.1
                period = 0  # Periods are not (yet) supported in DHNx
                expr += n.heat_loss_factor[t] * m.InvestmentFlowBlock.invest[
                    n, list(n.outputs.keys())[0], period
                ]
            expr += n.heat_loss_factor_fix[t]
            return expr == 0
        self.heat_loss_equation_convex = Constraint(
            self.CONVEX_INVESTHEATPIPES, m.TIMESTEPS,
            rule=_heat_loss_rule_convex
        )

        def _heat_loss_rule_nonconvex(block, n, t):
            """Rule definition for constraint to connect the installed capacity
            and the heat loss
            """
            expr = 0
            expr += - block.heat_loss[n, t]
            try:  # oemof.solph<=0.5.0
                expr += n.heat_loss_factor[t] * m.InvestmentFlowBlock.invest[
                    n, list(n.outputs.keys())[0]]
                expr += n.heat_loss_factor_fix[t] * \
                    m.InvestmentFlowBlock.invest_status[
                        n, list(n.outputs.keys())[0]]
            except KeyError:  # oemof.solph>=0.5.1
                period = 0  # Periods are not (yet) supported in DHNx
                expr += n.heat_loss_factor[t] * m.InvestmentFlowBlock.invest[
                    n, list(n.outputs.keys())[0], period]
                expr += n.heat_loss_factor_fix[t] * \
                    m.InvestmentFlowBlock.invest_status[
                        n, list(n.outputs.keys())[0], period]
            return expr == 0

        self.heat_loss_equation_nonconvex = Constraint(
            self.NONCONVEX_INVESTHEATPIPES, m.TIMESTEPS,
            rule=_heat_loss_rule_nonconvex
        )

        def _relation_rule_no_demand(block, n, t):
            """Link input and output flow and subtract heat loss."""
            i = list(n.inputs.keys())[0]
            o = list(n.outputs.keys())[0]

            expr = 0
            try:  # oemof.solph<=0.5.0
                expr += - m.flow[n, o, t]
                expr += m.flow[i, n, t]
            except KeyError:  # oemof.solph>=0.5.1
                period = 0  # Periods are not (yet) supported in DHNx
                expr += - m.flow[n, o, period, t]
                expr += m.flow[i, n, period, t]

            expr += - block.heat_loss[n, t]

            return expr == 0

        self.relation_no_demand = Constraint(
            self.INVESTHEATPIPES_NO_DEMAND, m.TIMESTEPS,
            rule=_relation_rule_no_demand
        )

        def _relation_rule_with_demand(block, n, t):
            """Link input and output flow and subtract heat loss."""
            i = list(n.inputs.keys())[0]
            o = list(n.outputs.keys())[0]
            d = list(n.outputs.keys())[1]

            expr = 0
            try:  # oemof.solph<=0.5.0
                expr += - m.flow[n, o, t]
                expr += m.flow[i, n, t]
                expr += - block.heat_loss[n, t]
                expr += - m.flow[n, d, t]
            except KeyError:  # oemof.solph>=0.5.1
                period = 0  # Periods are not (yet) supported in DHNx
                expr += - m.flow[n, o, period, t]
                expr += m.flow[i, n, period, t]
                expr += - block.heat_loss[n, t]
                expr += - m.flow[n, d, period, t]

            return expr == 0
        self.relation_with_demand = Constraint(
            self.INVESTHEATPIPES_WITH_DEMAND, m.TIMESTEPS,
            rule=_relation_rule_with_demand
        )

        def _inflow_outflow_invest_coupling_rule(block, n):  # pylint: disable=unused-argument
            """Rule definition of constraint connecting the inflow
            `InvestmentFlowBlock.invest of pipe with invested outflow `invest`
            by nominal_storage_capacity__inflow_ratio
            """
            i = list(n.inputs.keys())[0]
            o = list(n.outputs.keys())[0]

            try:  # oemof.solph<=0.5.0
                expr = (m.InvestmentFlowBlock.invest[i, n]
                        == m.InvestmentFlowBlock.invest[n, o])
            except KeyError:  # oemof.solph>=0.5.1
                period = 0  # Periods are not (yet) supported in DHNx
                expr = (m.InvestmentFlowBlock.invest[i, n, period]
                        == m.InvestmentFlowBlock.invest[n, o, period])
            return expr
        self.inflow_outflow_invest_coupling = Constraint(
            self.INVESTHEATPIPES, rule=_inflow_outflow_invest_coupling_rule
        )

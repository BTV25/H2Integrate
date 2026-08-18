import numpy as np
import openmdao.api as om
from attrs import field, define


try:
    from ard.api import set_up_ard_model
except ModuleNotFoundError:
    set_up_ard_model = None

try:
    from ard.farm_aero.surrogate import SurrogateFarmPower
    from ard.cost.surrogate_cost import SurrogateWindCost
except ModuleNotFoundError:
    SurrogateFarmPower = None
    SurrogateWindCost = None

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.model_baseclasses import (
    CostModelBaseClass,
    CostModelBaseConfig,
    PerformanceModelBaseClass,
)


class AEPBroadcastComponent(om.ExplicitComponent):
    """Converts scalar AEP (W*h) to a flat power time-series (kW).

    FLOWFarm produces a single scalar AEP from a wind-rose analysis, but
    H2Integrate expects a power time-series of length n_timesteps in kW.
    This component distributes the AEP evenly: P_kw = AEP_Wh / (8760 * 1000).
    """

    def initialize(self):
        self.options.declare("n_timesteps", types=int)

    def setup(self):
        n = self.options["n_timesteps"]
        self._n = n
        self._scale = 1.0 / (8760.0 * 1000.0)
        self.add_input("AEP_farm", val=0.0, units="W*h")
        self.add_output("ard_electricity_out", val=np.zeros(n), units="kW")

    def setup_partials(self):
        self.declare_partials(
            "ard_electricity_out", "AEP_farm",
            val=np.full((self._n, 1), self._scale),
        )

    def compute(self, inputs, outputs):
        outputs["ard_electricity_out"] = np.full(
            self._n, inputs["AEP_farm"].item() * self._scale
        )


class BatchPowerKWComponent(om.ExplicitComponent):
    """Converts FLOWFarmBatchPower output (W, n_timesteps) to kW time-series.

    FLOWFarmBatchPower returns per-state farm power in Watts in temporal order.
    H2Integrate expects a power time-series of length n_timesteps in kW.
    """

    def initialize(self):
        self.options.declare("n_timesteps", types=int)

    def setup(self):
        n = self.options["n_timesteps"]
        self.add_input("power_farm", val=np.zeros(n), units="W")
        self.add_output("ard_electricity_out", val=np.zeros(n), units="kW")

    def setup_partials(self):
        n = self.options["n_timesteps"]
        idx = np.arange(n)
        self.declare_partials("ard_electricity_out", "power_farm",
                              rows=idx, cols=idx, val=1e-3)

    def compute(self, inputs, outputs):
        outputs["ard_electricity_out"] = inputs["power_farm"] * 1e-3


@define
class WindPlantArdModelConfig(BaseConfig):
    """Configuration container for Ard wind plant model inputs.

    Attributes:
        ard_system (dict): Dictionary of Ard system / layout parameters (turbine specs,
            layout bounds, wake model settings, etc.) passed through to ``set_up_ard_model``.
        ard_data_path (str): Root path to Ard data resources (e.g., turbine libraries).
    """

    ard_system: dict = field()
    ard_data_path: str = field()


class WindArdPerformanceCompatibilityComponent(PerformanceModelBaseClass):
    """The class is needed to allow connecting the Ard cost_year easily in H2Integrate.

    This component takes some of the output of Ard and returns it in the format expected
    by H2Integrate. Some minor calculations are performed to get metrics required by
    H2Integrate.
    """

    def initialize(self):
        super().initialize()
        self.commodity = "electricity"
        self.commodity_rate_units = "kW"
        self.commodity_amount_units = "kW*h"
        if set_up_ard_model is None:
            msg = (
                "Please install `ard-nrel` or `h2integrate[ard]` to use the"
                " `WindArdPerformanceCompatibilityComponent` Ard-based model."
                " It is highly recommended to run `conda install wisdem` first. See H2I's"
                "installation instructions for further details."
            )
            raise ModuleNotFoundError(msg)

    def setup(self):
        self.config = WindPlantArdModelConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )

        super().setup()

        self.hours_per_year = 8760
        n_turbines_init = self.config.ard_system["modeling_options"]["layout"]["N_turbines"]
        turbine_specs = self.config.ard_system["modeling_options"]["windIO_plant"]["wind_farm"][
            "turbine"
        ]
        # windio rated power in W, convert to kW
        self.turbine_rating_kw = turbine_specs["performance"]["rated_power"] * 1e-3

        # N_turbines is an input (not a fixed setup-time constant) so that
        # rated_electricity_production/capacity_factor track the driven value in
        # surrogate mode, rather than staying pinned to the config's initial N.
        # In flowfarm/FLORIS mode this input is left unconnected -- N_turbines is
        # fixed by the x/y layout there anyway -- so it just keeps the previous,
        # config-derived constant behavior.
        self.add_input("N_turbines", val=n_turbines_init)

        self.add_input(
            "ard_electricity_out",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
        )

    def setup_partials(self):
        n = self.n_timesteps
        pl = self.plant_life
        idx = np.arange(n)
        self.declare_partials("electricity_out", "ard_electricity_out", rows=idx, cols=idx, val=1.0)
        self.declare_partials("total_electricity_produced", "ard_electricity_out",
                              val=np.ones((1, n)))
        scale = 1.0 / self.fraction_of_year_simulated
        self.declare_partials("annual_electricity_produced", "ard_electricity_out",
                              val=np.full((pl, n), scale))
        self.declare_partials("capacity_factor", "ard_electricity_out")
        self.declare_partials("capacity_factor", "N_turbines")
        self.declare_partials("rated_electricity_production", "N_turbines",
                              val=self.turbine_rating_kw)

    def compute(self, inputs, outputs):
        ard_electricity_series = inputs["ard_electricity_out"]
        n_turbines = inputs["N_turbines"].item()
        plant_rating_kw = n_turbines * self.turbine_rating_kw
        plant_capacity = plant_rating_kw * self.hours_per_year

        # ard has no concept of time and will simulate for all
        # wind conditions provided, including duplicates. Here we
        # convert for time step length and simulation length
        # to get an estimate of the annual energy production regardless
        # of the length of the simulation
        aep = ard_electricity_series.sum() / (self.fraction_of_year_simulated)

        outputs["electricity_out"] = ard_electricity_series
        outputs["total_electricity_produced"] = ard_electricity_series.sum()
        outputs["annual_electricity_produced"] = aep
        outputs["rated_electricity_production"] = plant_rating_kw
        outputs["capacity_factor"] = aep / plant_capacity

    def compute_partials(self, inputs, partials):
        n = self.n_timesteps
        pl = self.plant_life
        ard_electricity_series = inputs["ard_electricity_out"]
        n_turbines = inputs["N_turbines"].item()
        plant_rating_kw = n_turbines * self.turbine_rating_kw
        plant_capacity = plant_rating_kw * self.hours_per_year
        aep = ard_electricity_series.sum() / self.fraction_of_year_simulated
        scale = 1.0 / self.fraction_of_year_simulated

        partials["capacity_factor", "ard_electricity_out"] = np.full((pl, n), scale / plant_capacity)
        # d(capacity_factor)/d(N_turbines) = -aep / (N^2 * turbine_rating_kw * hours_per_year)
        #                                   = -capacity_factor / N_turbines
        partials["capacity_factor", "N_turbines"] = np.full(
            (pl, 1), -(aep / plant_capacity) / n_turbines
        )


class WindArdCostCompatibilityComponent(CostModelBaseClass):
    """The class is needed to allow connecting the Ard cost_year easily in H2Integrate.

    We could almost use the CostModelBaseClass directly, but its setup method
    requires a self.config attribute to be defined, so we create this minimal subclass.
    """

    def setup(self):
        self.config = CostModelBaseConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost")
        )

        super().setup()

        self.add_input("ard_CapEx", val=0, units="USD")
        self.add_input("ard_OpEx", val=0.0, units="USD/year")
        self.add_input("total_length_cables", val=0.0, units="m")

        # Literature default for onshore 34.5 kV underground collection cable, fully
        # installed (material + trench + labor); scaled down from NREL/CP-500-41135's
        # submarine collection-cable material costs ($152-731/m, 2006$) since that
        # report notes unburied/onshore installation costs less than submarine burial.
        # Override in tech_config.yaml wind.model_inputs.cable_cost_per_meter if a
        # site-specific quote is available. Kept outside cost_parameters since
        # CostModelBaseConfig strictly validates that dict's keys.
        self.cable_cost_per_meter = self.options["tech_config"]["model_inputs"].get(
            "cable_cost_per_meter", 120.0
        )

    def setup_partials(self):
        self.declare_partials("CapEx", "ard_CapEx", val=1.0)
        self.declare_partials("CapEx", "total_length_cables", val=self.cable_cost_per_meter)
        self.declare_partials("OpEx", "ard_OpEx", val=1.0)

    def compute(self, inputs, outputs):
        outputs["CapEx"] = inputs["ard_CapEx"] + self.cable_cost_per_meter * inputs[
            "total_length_cables"
        ]
        outputs["OpEx"] = inputs["ard_OpEx"]


class CachedArdSubmodelComp(om.SubmodelComp):
    """SubmodelComp wrapping the Ard/FLOWFarm sub-problem, with input-hash caching.

    Once turbine layout (x/y positions, substation positions) is held fixed -- e.g.
    during the sizing/control co-optimization phase, where layout is no longer a
    design variable -- repeated compute() calls receive identical inputs and would
    otherwise re-solve the FLOWFarm wake model for no reason (each solve costs several
    seconds). Skip the inner problem's run_model()/run_driver() when the inputs are
    unchanged from the last call and reuse the cached outputs instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_key = None
        self._cache_outputs = None

    def compute(self, inputs, outputs):
        key = inputs.asarray().tobytes()
        if key == self._cache_key and self._cache_outputs is not None:
            self._outputs.set_val(self._cache_outputs)
            return
        super().compute(inputs, outputs)
        self._cache_key = key
        self._cache_outputs = self._outputs.asarray().copy()


class ArdWindPlantModel(om.Group):
    """OpenMDAO Group integrating the Ard wind plant as a sub-problem.

    Subsystems:

        ard_sub_prob (SubmodelComp): Encapsulated Ard Problem exposing specified inputs/outputs.
        wind_ard_performance_compatibility (WindArdPerformanceCompatibilityComponent):
            Necessary for providing required performance metrics to H2Integrate.
        wind_ard_cost_compatibility (WindArdCostCompatibilityComponent):
            Necessary for providing cost_year to H2Integrate.

    Promoted Inputs:

        spacing_primary: Primary spacing parameter.
        spacing_secondary: Secondary spacing parameter.
        angle_orientation: Orientation angle.
        angle_skew: Skew angle.
        x_substations: X-coordinates of substations.
        y_substations: Y-coordinates of substations.

    Promoted Outputs:

        electricity_out (float): Annual energy production (AEP) in MWh (as provided by ARD/FLORIS).
        CapEx (float): Capital expenditure from ARD turbine & balance of plant cost model.
        OpEx (float): Operating expenditure from ARD.
        boundary_distances (array): Distances from turbines to boundary segments.
        turbine_spacing (array): Inter-turbine spacing metrics.
        cost_year: Cost year from cost component.
        VarOpEx: Variable operating expenditure (currently placeholder).
    """

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = WindPlantArdModelConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )

        # create ard model
        ard_input_dict = self.config.ard_system
        ard_data_path = self.config.ard_data_path

        modeling_options = ard_input_dict.get("modeling_options", {})
        use_surrogate = "surrogate" in modeling_options

        n_timesteps = self.options["plant_config"]["plant"]["simulation"]["n_timesteps"]

        if use_surrogate:
            # N_turbines-only surrogate mode (build_surrogate.py, hybridfarm repo
            # root): skips FLOWFarm/optiwindnet/LandBOSSE entirely -- no x/y layout,
            # no Julia sub-problem -- in favor of the (ws, wd, N_turbines) -> power
            # and N_turbines -> CapEx/OpEx fits. N_turbines becomes a continuous,
            # driveable design variable rather than a fixed layout-derived count.
            if SurrogateFarmPower is None or SurrogateWindCost is None:
                raise ModuleNotFoundError(
                    "modeling_options.surrogate was set but ard.farm_aero.surrogate/"
                    "ard.cost.surrogate_cost could not be imported."
                )
            surrogate_opts = modeling_options["surrogate"]
            n_turbines_init = float(
                modeling_options.get("layout", {}).get("N_turbines", 25)
            )

            self.add_subsystem(
                "n_turbines_ivc",
                om.IndepVarComp("N_turbines", n_turbines_init),
                promotes=["*"],
            )
            self.add_subsystem(
                "surrogate_power",
                SurrogateFarmPower(
                    surrogate_pkl_path=surrogate_opts["f_power_pkl"],
                    wind_resource_npz_path=surrogate_opts["wind_resource_npz"],
                    n_timesteps=n_timesteps,
                ),
                promotes_inputs=["N_turbines"],
                promotes_outputs=[("electricity_out", "ard_electricity_out")],
            )
            self.add_subsystem(
                "surrogate_cost",
                SurrogateWindCost(surrogate_pkl_path=surrogate_opts["g_cost_pkl"]),
                promotes_inputs=["N_turbines"],
                promotes_outputs=[("CapEx", "ard_CapEx"), ("OpEx", "ard_OpEx")],
            )
            # surrogate CapEx already includes cabling cost (direct + LandBOSSE BOS
            # effect, see SurrogateWindCost) -- zero this out so
            # WindArdCostCompatibilityComponent's cable_cost_per_meter add-on below
            # doesn't double-count it.
            self.add_subsystem(
                "cable_length_zero_ivc",
                om.IndepVarComp("total_length_cables", 0.0, units="m"),
                promotes=["*"],
            )
        else:
            self._setup_flowfarm_subprob(ard_input_dict, ard_data_path, modeling_options, n_timesteps)

        # add performance model to include inputs and
        # outputs as expected by H2Integrate
        self.add_subsystem(
            "wind_ard_performance_compatibility",
            WindArdPerformanceCompatibilityComponent(
                driver_config=self.options["driver_config"],
                plant_config=self.options["plant_config"],
                tech_config=self.options["tech_config"],
            ),
            promotes=["*"],
        )

        # add pass-through cost model to include cost_year as expected by H2Integrate.
        # Promote everything EXCEPT cost_year — cost_year is a discrete constant that
        # must not appear in the derivative chain from x_turbines (which flows through
        # ard_CapEx into this component). A separate IndepVarComp owns cost_year instead.
        self.add_subsystem(
            "wind_ard_cost_compatibility",
            WindArdCostCompatibilityComponent(
                driver_config=self.options["driver_config"],
                plant_config=self.options["plant_config"],
                tech_config=self.options["tech_config"],
            ),
            promotes_inputs=["*"],
            promotes_outputs=["CapEx", "OpEx", "VarOpEx"],
        )

        # Provide cost_year independently so it has no dependency on continuous DVs.
        cost_params = self.options["tech_config"]["model_inputs"].get("cost_parameters", {})
        cost_year = int(cost_params.get("cost_year", 2024))
        cost_year_ivc = om.IndepVarComp()
        cost_year_ivc.add_discrete_output("cost_year", val=cost_year)
        self.add_subsystem("cost_year_src", cost_year_ivc, promotes_outputs=["cost_year"])

    def _setup_flowfarm_subprob(self, ard_input_dict, ard_data_path, modeling_options, n_timesteps):
        """Full FLOWFarm/FLORIS + optiwindnet collection + LandBOSSE sub-problem,
        driven by an explicit x/y turbine layout (as opposed to the N_turbines-only
        `use_surrogate` path in setup())."""
        ard_prob = set_up_ard_model(input_dict=ard_input_dict, root_data_path=ard_data_path)

        # x_turbines_in/y_turbines_in are unconnected inputs in the inner problem;
        # OpenMDAO's _auto_ivc backs them automatically and SubmodelComp exposes
        # _auto_ivc-tagged outputs as driveable inputs to the outer problem.

        # detect FLOWFarm vs FLORIS and wind resource mode
        use_flowfarm = "flowfarm" in modeling_options
        wind_resource = (
            modeling_options
            .get("windIO_plant", {})
            .get("site", {})
            .get("energy_resource", {})
            .get("wind_resource", {})
        )
        # timeseries mode → FLOWFarmBatchPower returns per-state power_farm (W)
        # probability mode → FLOWFarmAEP returns scalar AEP_farm (W*h)
        use_batch_power = use_flowfarm and "time" in wind_resource

        if use_batch_power:
            subprob_outputs = [
                ("power_farm", "power_farm"),
                ("power_turbines", "power_turbines"),
                ("tcc.tcc", "ard_CapEx"),
                ("opex.opex", "ard_OpEx"),
                "boundary_distances",
                "turbine_spacing",
                "total_length_cables",
            ]
        elif use_flowfarm:
            subprob_outputs = [
                ("AEP_farm", "AEP_farm"),
                ("tcc.tcc", "ard_CapEx"),
                ("opex.opex", "ard_OpEx"),
                "boundary_distances",
                "turbine_spacing",
                "total_length_cables",
            ]
        else:
            subprob_outputs = [
                ("power_farm", "ard_electricity_out"),
                ("tcc.tcc", "ard_CapEx"),
                ("opex.opex", "ard_OpEx"),
                "boundary_distances",
                "turbine_spacing",
                "total_length_cables",
            ]

        # add ard to the h2i model as a sub-problem
        # SubmodelComp alias ("inner_name", "outer_promoted_name") maps the outer IVC's
        # x_turbines/y_turbines to the inner x_turbines_in/y_turbines_in.
        subprob_ard = CachedArdSubmodelComp(
            problem=ard_prob,
            inputs=[
                ("x_turbines_in", "x_turbines"),
                ("y_turbines_in", "y_turbines"),
                "x_substations",
                "y_substations",
            ],
            outputs=subprob_outputs,
        )

        # add the ard sub-problem to the parent group
        self.add_subsystem(
            "ard_sub_prob",
            subprob_ard,
            promotes=["*"],
        )

        if use_batch_power:
            # FLOWFarmBatchPower: per-state power (W, length n_timesteps) → kW time-series
            self.add_subsystem(
                "batch_power_kw",
                BatchPowerKWComponent(n_timesteps=n_timesteps),
                promotes=["*"],
            )
        elif use_flowfarm:
            # FLOWFarmAEP: scalar AEP (W*h) → flat kW time-series
            self.add_subsystem(
                "aep_broadcast",
                AEPBroadcastComponent(n_timesteps=n_timesteps),
                promotes=["*"],
            )


import numpy as np
from attrs import field, define

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.validators import gte_zero
from h2integrate.core.model_baseclasses import PerformanceModelBaseClass


@define(kw_only=True)
class SimpleGenericStorageConfig(BaseConfig):
    commodity: str = field()
    commodity_rate_units: str = field()  # TODO: update to commodity_rate_units
    max_charge_rate: float = field(validator=gte_zero)


class SimpleGenericStorage(PerformanceModelBaseClass):
    """
    Simple generic storage model that acts as a pass-through component.

    Note: this storage performance model is intended to be used with the
    `DemandOpenLoopStorageController` controller and has not been tested
    with other controllers.

    """

    def setup(self):
        self.config = SimpleGenericStorageConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            strict=False,
            additional_cls_name=self.__class__.__name__,
        )
        self.commodity = self.config.commodity
        self.commodity_rate_units = self.config.commodity_rate_units
        self.commodity_amount_units = f"({self.commodity_rate_units})*h"
        super().setup()
        self.add_input(
            f"{self.commodity}_set_point",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
        )
        self.add_input(
            "max_charge_rate",
            val=self.config.max_charge_rate,
            units=self.config.commodity_rate_units,
            desc="Storage charge/discharge rate",
        )

    def setup_partials(self):
        n = self.n_timesteps
        arange = np.arange(n)
        c = self.commodity
        # identity pass-throughs
        self.declare_partials(f"{c}_out", f"{c}_set_point", rows=arange, cols=arange, val=1.0)
        self.declare_partials(f"rated_{c}_production", "max_charge_rate", val=1.0)
        # total = sum(set_point): dense row of ones
        self.declare_partials(f"total_{c}_produced", f"{c}_set_point",
                              rows=np.zeros(n, dtype=int), cols=arange, val=1.0)
        # annual = total / fraction_of_year: same sparsity, scaled
        self.declare_partials(f"annual_{c}_produced", f"{c}_set_point",
                              rows=np.zeros(n, dtype=int), cols=arange,
                              val=1.0 / self.fraction_of_year_simulated)
        # capacity_factor depends on state; computed in compute_partials
        self.declare_partials("capacity_factor", f"{c}_set_point")
        self.declare_partials("capacity_factor", "max_charge_rate")

    def compute_partials(self, inputs, partials):
        c = self.commodity
        rated = float(inputs["max_charge_rate"][0])
        denom = rated * self.n_timesteps * (self.dt / 3600)
        total = float(np.sum(inputs[f"{c}_set_point"]))
        # capacity_factor is (plant_life,) — Jacobian is (plant_life, n_ts) dense
        partials["capacity_factor", f"{c}_set_point"] = np.full(
            (self.plant_life, self.n_timesteps), 1.0 / denom
        )
        partials["capacity_factor", "max_charge_rate"] = -total / (rated * denom)

    def compute(self, inputs, outputs):
        # Pass the commodity_out as the commodity_set_point
        outputs[f"{self.commodity}_out"] = inputs[f"{self.commodity}_set_point"]

        # Set the rated commodity production from the max_charge_rate input
        outputs[f"rated_{self.commodity}_production"] = inputs["max_charge_rate"]

        # Calculate the total and annual commodity produced
        outputs[f"total_{self.commodity}_produced"] = outputs[f"{self.commodity}_out"].sum()
        outputs[f"annual_{self.commodity}_produced"] = outputs[
            f"total_{self.commodity}_produced"
        ] * (1 / self.fraction_of_year_simulated)

        # Calculate the maximum theoretical commodity production over the simulation
        rated_production = (
            outputs[f"rated_{self.commodity}_production"] * self.n_timesteps * (self.dt / 3600)
        )

        # rated_production == 0 when max_charge_rate == 0 (battery/storage removed);
        # capacity_factor is undefined there, so define it as 0 rather than dividing.
        outputs["capacity_factor"] = np.where(
            rated_production > 0,
            outputs[f"total_{self.commodity}_produced"] / np.where(rated_production > 0, rated_production, 1.0),
            0.0,
        )

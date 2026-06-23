"""H2 revenue credit component for co-production LCOE calculations.

Credits hydrogen sales revenue against electricity LCOE by outputting a
negative VarOpEx equal to annual H2 production × H2 sale price.
"""

import numpy as np
import openmdao.api as om


class H2RevenueModel(om.ExplicitComponent):
    """Cost model that represents H2 sales revenue as a negative VarOpEx.

    When included in the electricity finance subgroup, the negative VarOpEx
    flows through ProFAST's coproduct mechanism and reduces LCOE.

    Config keys (under model_inputs.cost_parameters):
        h2_price (float): H2 sale price in $/kg.
        cost_year (int): Dollar year for costs.
    """

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        model_inputs = self.options["tech_config"]["model_inputs"]
        cost_params = {
            **model_inputs.get("shared_parameters", {}),
            **model_inputs.get("cost_parameters", {}),
        }
        self.h2_price = float(cost_params["h2_price"])
        self.cost_year_val = int(cost_params["cost_year"])
        self.plant_life = int(self.options["plant_config"]["plant"]["plant_life"])
        n_timesteps = self.options["plant_config"]["plant"]["simulation"]["n_timesteps"]

        self.add_input("hydrogen_in", val=0.0, shape=n_timesteps, units="kg/h")

        self.add_output("CapEx", val=0.0, units="USD")
        self.add_output("OpEx", val=0.0, units="USD/year")
        self.add_output(
            "VarOpEx", val=0.0, shape=self.plant_life, units="USD/year",
            desc="Negative = H2 revenue credit against LCOE",
        )
        self.add_output(
            "replacement_schedule", val=0.0, shape=self.plant_life, units="unitless"
        )

    def setup_partials(self):
        pl = self.plant_life
        n_ts = self.options["plant_config"]["plant"]["simulation"]["n_timesteps"]
        rows = np.repeat(np.arange(pl), n_ts)
        cols = np.tile(np.arange(n_ts), pl)
        self.declare_partials("VarOpEx", "hydrogen_in", rows=rows, cols=cols)

    def compute_partials(self, inputs, partials):
        n_ts = inputs["hydrogen_in"].size
        partials["VarOpEx", "hydrogen_in"] = np.full(self.plant_life * n_ts, -self.h2_price)

    def compute(self, inputs, outputs):
        annual_h2_kg = np.sum(inputs["hydrogen_in"])  # kg/yr
        annual_revenue = annual_h2_kg * self.h2_price  # USD/yr
        outputs["CapEx"] = 0.0
        outputs["OpEx"] = 0.0
        outputs["VarOpEx"] = np.full(self.plant_life, -annual_revenue)
        outputs["replacement_schedule"] = np.zeros(self.plant_life)

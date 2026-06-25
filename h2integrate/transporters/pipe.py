import numpy as np
import openmdao.api as om


class PipePerformanceModel(om.ExplicitComponent):
    """
    Pass-through pipe with no losses.
    """

    def initialize(self):
        self.options.declare(
            "transport_item",
            values=[
                "hydrogen",
                "co2",
                "methanol",
                "ammonia",
                "nitrogen",
                "natural_gas",
                "wellhead_gas",
                "water",
            ],
        )
        self.options.declare("plant_config", default=None)

    def setup(self):
        transport_item = self.options["transport_item"]
        self.input_name = transport_item + "_in"
        self.output_name = transport_item + "_out"

        if transport_item == "natural_gas":
            units = "MMBtu/h"
        elif transport_item == "water":
            units = "galUS"
        elif transport_item == "co2":
            units = "kg/h"
        else:
            units = "kg/s"

        self.add_input(
            self.input_name,
            val=-1.0,
            shape_by_conn=True,
            copy_shape=self.output_name,
            units=units,
        )
        self.add_output(
            self.output_name,
            val=-1.0,
            shape_by_conn=True,
            copy_shape=self.input_name,
            units=units,
        )

    def setup_partials(self):
        plant_config = self.options["plant_config"]
        if plant_config is not None:
            n = plant_config["plant"]["simulation"]["n_timesteps"]
            arange = np.arange(n)
            self.declare_partials(self.output_name, self.input_name,
                                  rows=arange, cols=arange, val=1.0)
        else:
            self.declare_partials(self.output_name, self.input_name, method="cs")

    def compute(self, inputs, outputs):
        outputs[self.output_name] = inputs[self.input_name]

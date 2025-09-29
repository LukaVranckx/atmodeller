import logging

import numpy as np
import optimistix as optx
from atmodeller import (
    InteriorAtmosphere,
    Planet,
    Species,
    SpeciesCollection,
    debug_logger,
    earth_oceans_to_hydrogen_mass,
    SolverParameters,
)
from atmodeller.eos import get_eos_models
from atmodeller.solubility import get_solubility_models
from atmodeller.thermodata import IronWustiteBuffer

logger = debug_logger()
logger.setLevel(logging.INFO)

# For more output use DEBUG
# logger.setLevel(logging.DEBUG)

species = SpeciesCollection(
    (
        Species.create_gas("Al"),
        Species.create_gas("AlO"),
        Species.create_gas("AlO2"),
        Species.create_gas("Al2"),
        Species.create_gas("Al2O"),
        Species.create_gas("Al2O2"),
        Species.create_gas("Al2O3"),
        Species.create_gas("Ca"),
        Species.create_gas("CaO"),
        Species.create_gas("Ca2"),
        Species.create_gas("Cr"),
        Species.create_gas("CrO"),
        Species.create_gas("CrO2"),
        Species.create_gas("CrO3"),
        Species.create_gas("Na"),
        Species.create_gas("NaO"),
        Species.create_gas("Na2"),
        Species.create_gas("Na2O"),
        Species.create_gas("Na2O2"),
        Species.create_gas("Mg"),
        Species.create_gas("MgO"),
        Species.create_gas("Fe"),
        Species.create_gas("FeO"),
        Species.create_gas("Si"),
        Species.create_gas("OSi"),
        Species.create_gas("O2Si"),
        Species.create_gas("Ti"),
        Species.create_gas("OTi"),
        Species.create_gas("O2Ti"),
    )
)
interior_atmosphere = InteriorAtmosphere(species)
fugacity_constraints = {
    # "Al":
}

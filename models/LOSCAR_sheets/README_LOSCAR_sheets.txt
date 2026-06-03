This workbook contains model configuration information used to initialize the LOSCAR model within ESBMTK. 

Specifically, it contains worksheets describing the following model information:
- reservoirs: defines ocean reservoirs, sources, and sinks.
- gas_reservoirs: Atmospheric gas reservoir definitions. 
- transport_matrix: defines transport fluxes between reservoirs.
- gas_exchange: Air-sea gas exchange definitions.

Notes:
- Column names must exactly match the expected names described below.
- Reservoir names used in the worksheets must match those used in the model.
- Any variables specified herein must also be defined in the python model file. 
- Numerical values may be entered directly or calculated using standard Excel formulas.

Column names: reservoirs
- name: unique reservoir, source, or sink name.
- type: Object type: reservoir, source, or sink
- z_top: Upper depth boundary of the reservoir (m).
- z_bottom: Lower depth boundary of the reservoir (m).
- area_percentage: Fraction (%) of total ocean surface area occupied by the reservoir.
- temperature: Initial reservoir temperature (°C)
- pressure: Reservoir pressure (atm).
- salinity: Reservoir salinity (psu)
- [chemical species]: Initial concentration of the specified species (in umol/lg). Column names must match species names defined in the model. 

Column names: transport_matrix
- source: Name of the source reservoir.
- sink: Name of the sink reservoir.
- flux_id: type of flux, “thermohaline” or “mix_up”. Note that mixing fluxes are symmetric, so a corresponding “mix_down” flux need not be specified and will be created automatically.
- sc: transport scaling factor. May be specified in Sverdrups (e.g. “10 Sverdrup”) or as a variable defined within the python file.

Column names: gas_reservoirs
- name: name of atmospheric reservoir
- species: gaseous species to which the atmospheric reservoir corresponds to. Must exist as a registered species within ESBMTK.
- species_ppm: initial concentration of the gas species. May be specified in ppm or percent (e.g.: 280 ppm; e.g.: 21 percent)

Column names: gas_exchange
- species: gaseous species being exchanged between the atmosphere and ocean
- basins: comma-separated list of ocean surface boxes participating in air-sea gas exchange
- piston_velocity: velocity of piston gas exchange (e.g.: “4m/d”)

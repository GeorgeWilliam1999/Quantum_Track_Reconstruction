# Run 3 (2024 W40-42, magnet down, nu = 7.6) minimum bias, Gauss Sim10 production settings,
# without the spill-over crossings (only the main crossing's VELO hits are used downstream).
import os
from Gaudi.Configuration import importOptions
from Configurables import LHCbApp, Gauss, GenInit
importOptions("$APPCONFIGOPTS/Gauss/Beam6800GeV-md100-2024.W40.42-nu7.6.py")
importOptions("$APPCONFIGOPTS/Gauss/Run3-detector.py")
importOptions("$APPCONFIGOPTS/Gauss/DataType-2024.py")
importOptions("$GAUSSOPTS/DBTags-2024.py")
importOptions("$APPCONFIGOPTS/Persistency/BasketSize-10.py")
importOptions("$DECFILESROOT/options/30000000.py")      # minimum bias
importOptions("$LBPYTHIA8ROOT/options/Pythia8.py")
importOptions("$APPCONFIGOPTS/Gauss/G4PL_FTFP_BERT_EmOpt2.py")
run = int(os.environ.get("RUNNUMBER", "5000"))
GenInit("GaussGen").FirstEventNumber = 1
GenInit("GaussGen").RunNumber = run
LHCbApp().EvtMax = int(os.environ.get("NEVT", "3"))
Gauss().OutputType = "SIM"
Gauss().DatasetName = f"run3_minbias_md_2024_run{run}"

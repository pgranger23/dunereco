cet_build_plugin(FlashMatchBDT art::EDProducer
  SOURCE
  FlashMatchBDT_module.cc
  FlashMatchBDTAlg.cxx
  BDTModel.cxx
  LIBRARIES PRIVATE
  larcore::Geometry_Geometry_service
  larcorealg::Geometry
  cetlib_except::cetlib_except
  lardataobj::AnalysisBase
  lardataobj::RecoBase
  art::Framework_Principal
  canvas::canvas
  cetlib::cetlib
  messagefacility::MF_MessageLogger
  fhiclcpp::fhiclcpp
)

install_headers()
install_fhicl()
install_fhicl(LIST bdt_augmented.bin bdt_baseline.bin bdt_vd_light.bin bdt_vd_vertex.bin SUBDIRNAME FlashMatch)
install_source()
install_scripts(LIST scripts/export_bin_bdt.py)

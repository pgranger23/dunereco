#ifndef DUNERECO_FLASHMATCH_FLASHMATCHBDTALG_H
#define DUNERECO_FLASHMATCH_FLASHMATCHBDTALG_H

/**
 * \file FlashMatchBDTAlg.h
 *
 * \brief Pure C++ algorithmic engine for BDT-based charge-light matching in DUNE.
 *
 * Evaluates candidate flashes against a reconstructed charge position using
 * the 13-feature XGBoost BDT model loaded from an external compact binary asset,
 * selecting the most compatible flash and determining interaction T0.
 *
 * \author Pierre Granger (pierre.granger@cern.ch)
 * \date 2026
 */

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstddef>

#include "fhiclcpp/ParameterSet.h"
#include "lardataobj/RecoBase/OpFlash.h"
#include "BDTModel.h"

namespace flashmatch {

/// One OpHit of a flash, at the centre of its photon detector (FeatureSet "VD")
struct FlashHit {
  double pe = 0.;
  double time = 0.;             ///< PeakTime [us]
  double x = 0., y = 0., z = 0.; ///< PD centre [cm]
  int channel = -1;
};

struct MatchResult {
  bool matched = false;
  int flash_idx = -1;       ///< Index in original flash collection
  double flash_time = 0.0;  ///< Reconstructed flash time [us]
  double bdt_score = 0.0;   ///< BDT classification score in [0, 1]
  double pe = 0.0;          ///< Flash total PE
  double dy = 0.0;          ///< Centroid delta Y [cm]
  double dz = 0.0;          ///< Centroid delta Z [cm]
  double chi2 = 0.0;        ///< Spatial chi2
  int n_candidates = 0;     ///< Number of candidate flashes passing PE threshold
};

class FlashMatchBDTAlg {
public:
  struct Config {
    std::string weight_file = "FlashMatch/bdt_augmented.bin"; ///< Path or filename of binary model weights
    double pe_threshold = 100.0;                              ///< Minimum flash PE to consider as candidate
    double spatial_sigma = 62.64;                             ///< Spatial resolution scale sigma [cm]
    double min_bdt_score = 0.30;                              ///< Minimum BDT score to declare a positive match
    bool fallback_to_charge = true;                           ///< Allow fallback to charge centroid if no vertex
    std::string feature_set = "HD";                           ///< "HD" (Match) or "VD" (MatchVD)
    double cathode_x = -320.;                                 ///< VD: PDs with x below this are on the cathode [cm]
    int min_channels = 3;                                     ///< VD: minimum distinct channels of a candidate
    int rank_min_channels = 20;                               ///< VD: second PE ranking among flashes with this many channels
  };

  FlashMatchBDTAlg();
  explicit FlashMatchBDTAlg(const fhicl::ParameterSet& pset);
  explicit FlashMatchBDTAlg(const Config& config);

  void Configure(const fhicl::ParameterSet& pset);
  void Configure(const Config& config);

  /**
   * @brief Match a collection of flashes to a reconstructed charge position.
   *
   * @param flashes Input collection of recob::OpFlash
   * @param charge_y Y coordinate of reconstructed vertex or centroid [cm]
   * @param charge_z Z coordinate of reconstructed vertex or centroid [cm]
   * @param has_charge Whether the charge position is valid
   * @return MatchResult Output matching summary and best flash index
   */
  MatchResult Match(const std::vector<recob::OpFlash>& flashes,
                    double charge_y,
                    double charge_z,
                    bool has_charge = true) const;

  /**
   * @brief FeatureSet "VD": select the neutrino flash from the light alone.
   *
   * Candidates are flashes with >= min_channels distinct channels and >= pe_threshold PE.
   * Each is described by 18 observables of its hits: PE, channel and hit counts, PE-weighted
   * centroid and widths, PE fractions on the cathode and on each membrane, PE fraction in the
   * 1 and 3 brightest channels, duration, its PE rank among all candidates and among those with
   * >= rank_min_channels channels, and the number of candidates. Neither the flash time nor a
   * charge position enters, so the result is the same for every PFParticle of the event.
   *
   * @param flashes Input collection of recob::OpFlash
   * @param hits hits[i] are the OpHits of flashes[i]
   * @return MatchResult with the highest-scoring candidate
   */
  MatchResult MatchVD(const std::vector<recob::OpFlash>& flashes,
                      const std::vector<std::vector<FlashHit>>& hits) const;

  const Config& GetConfig() const { return fConfig; }
  const BDTModel& GetModel() const { return fModel; }
  bool IsVD() const { return fConfig.feature_set == "VD"; }

private:
  static constexpr size_t kNumVDFeatures = 18;

  void LoadModel();

  Config fConfig;
  BDTModel fModel;
};

} // namespace flashmatch

#endif // DUNERECO_FLASHMATCH_FLASHMATCHBDTALG_H

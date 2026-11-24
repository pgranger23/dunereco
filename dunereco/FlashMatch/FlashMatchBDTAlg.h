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

/// One OpHit of a flash, with the centre of its photon detector (VD feature set)
struct FlashHit {
  double pe = 0.;
  double time = 0.;    ///< PeakTime [us]
  double x = 0., y = 0., z = 0.;
  int channel = -1;
};

/// A flash as the VD feature set sees it: its index in the OpFlash collection and its hits
struct FlashHits {
  int flash_idx = -1;
  std::vector<FlashHit> hits;
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
    // VD feature set (FeatureSet: "VD"); see MatchVD
    std::string feature_set = "HD";                           ///< "HD" (13 OpFlash features) or "VD"
    std::string light_weight_file = "";                       ///< VD: light-only model, used without a charge position
    double cathode_x = -320.;                                 ///< VD: PDs with x below this are on the cathode [cm]
    int min_channels = 3;                                     ///< VD: minimum distinct channels of a candidate
    int rank_min_channels = 20;                               ///< VD: second PE ranking among flashes with this many channels
  };

  /// Number of VD features: light-only model, and light + charge-position model
  static constexpr size_t kNumVDLight = 18;
  static constexpr size_t kNumVDVertex = 23;

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
   * @brief VD feature set: match flashes, described by their OpHits, to a charge position.
   *
   * Candidates are flashes with >= min_channels distinct channels and >= pe_threshold PE.
   * Each is described by light observables computed from its hits (PE, channel and hit counts,
   * PE-weighted centroid and widths, PE fractions on the cathode and on each membrane, PE fraction
   * in the 1 and 3 brightest channels, duration) and by its PE rank among all candidates and among
   * those with >= rank_min_channels channels. With a charge position, the (y, z) offsets are added
   * and the main model (WeightFile) is used; without one, the light-only model (LightWeightFile).
   * No flash time enters the features. The highest-scoring candidate is returned.
   *
   * @param flashes Flashes with their hits (flash_idx = index in the OpFlash collection)
   * @param charge_y Y of the reconstructed vertex or charge centroid [cm]
   * @param charge_z Z of the reconstructed vertex or charge centroid [cm]
   * @param has_charge Whether the charge position is valid
   * @return MatchResult (flash_time is left to the caller)
   */
  MatchResult MatchVD(const std::vector<FlashHits>& flashes,
                      double charge_y,
                      double charge_z,
                      bool has_charge) const;

  /// VD features of the candidates (in feature order of the models); cand receives their indices in flashes
  std::vector<std::vector<float>> ComputeVDFeatures(const std::vector<FlashHits>& flashes,
                                                    double charge_y,
                                                    double charge_z,
                                                    bool with_charge,
                                                    std::vector<int>& cand) const;

  const Config& GetConfig() const { return fConfig; }
  const BDTModel& GetModel() const { return fModel; }
  const BDTModel& GetLightModel() const { return fLightModel; }
  bool IsVD() const { return fConfig.feature_set == "VD"; }

private:
  void LoadModels();

  Config fConfig;
  BDTModel fModel;
  BDTModel fLightModel;
};

} // namespace flashmatch

#endif // DUNERECO_FLASHMATCH_FLASHMATCHBDTALG_H

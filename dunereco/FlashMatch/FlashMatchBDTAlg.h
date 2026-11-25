#ifndef DUNERECO_FLASHMATCH_FLASHMATCHBDTALG_H
#define DUNERECO_FLASHMATCH_FLASHMATCHBDTALG_H

/**
 * \file FlashMatchBDTAlg.h
 *
 * \brief Pure C++ algorithmic engine for BDT-based charge-light matching in DUNE.
 *
 * Scores candidate flashes with an XGBoost BDT loaded from an external compact
 * binary asset and selects the highest-scoring one. The model inputs are given by
 * name in the configuration (Features), in the order the model was trained with;
 * the available features are listed in FlashMatchBDTAlg.cxx. They come from:
 *  - the recob::OpFlash (PE, centre, widths),
 *  - the OpHits of the flash (channel counts, PE sharing, hit-based widths, duration),
 *  - the set of candidates of the event (PE ranks, number of candidates),
 *  - a charge position (offsets between flash centre and vertex or charge centroid).
 * The detector-specific choices are thus configuration only.
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

/// One OpHit of a flash, at the centre of its photon detector
struct FlashHit {
  double pe = 0.;
  double time = 0.;              ///< PeakTime [us]
  double x = 0., y = 0., z = 0.; ///< PD centre [cm]
  int channel = -1;
};

/// Observables of a candidate flash; hit-based ones are only filled when hits are given
struct Candidate {
  int flash_idx = -1;        ///< Index in the OpFlash collection
  double time = 0.;          ///< OpFlash time [us]
  double pe = 0., y = 0., z = 0., y_width = 0., z_width = 0.;   ///< from the OpFlash
  int n_channels = 0, n_hits = 0;
  double hit_y_width = 0., hit_z_width = 0., duration = 0.;
  double frac_cathode = 0., frac_membrane_neg_y = 0., frac_membrane_pos_y = 0.;
  double frac_top1_channel = 0., frac_top3_channels = 0.;
  double pe_rank = 999., log10_pe_ratio = -6.;             ///< among all candidates
  double pe_rank_wide = 999., log10_pe_ratio_wide = -6.;   ///< among candidates with >= wide_min_channels
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
  int n_candidates = 0;     ///< Number of candidate flashes
};

class FlashMatchBDTAlg {
public:
  struct Config {
    std::string weight_file = "FlashMatch/bdt_augmented.bin"; ///< Path or filename of binary model weights
    std::vector<std::string> features = {"log10_pe", "dy", "dz", "abs_dy", "abs_dz", "chi2", "chi",
                                         "y_width", "z_width", "width_sum", "pe_rank", "log10_pe_ratio",
                                         "n_candidates"};    ///< Model inputs, in model order
    double pe_threshold = 100.0;                              ///< Minimum flash PE to consider as candidate
    int min_channels = 0;                                     ///< Minimum distinct channels of a candidate (needs hits if > 0)
    double spatial_sigma = 62.64;                             ///< Spatial resolution scale sigma [cm] (chi2, chi)
    int wide_min_channels = 20;                               ///< Channels of the candidates ranked by pe_rank_wide
    double cathode_x = -320.;                                 ///< PDs with x below this are on the cathode [cm] (frac_*)
    double min_bdt_score = 0.30;                              ///< Minimum BDT score to declare a positive match
    bool fallback_to_charge = true;                           ///< Allow fallback to charge centroid if no vertex
  };

  FlashMatchBDTAlg();
  explicit FlashMatchBDTAlg(const fhicl::ParameterSet& pset);
  explicit FlashMatchBDTAlg(const Config& config);

  void Configure(const fhicl::ParameterSet& pset);
  void Configure(const Config& config);

  /// Whether the features (or the candidate selection) need the OpHits of the flashes
  bool UsesHits() const { return fUsesHits; }
  /// Whether the features need a charge position; if not, the match is the same for every PFParticle
  bool UsesCharge() const { return fUsesCharge; }

  /**
   * @brief Candidate flashes (>= pe_threshold PE and >= min_channels channels) and their observables.
   *
   * @param flashes Input collection of recob::OpFlash
   * @param hits hits[i] are the OpHits of flashes[i]; required if UsesHits()
   */
  std::vector<Candidate> SelectCandidates(const std::vector<recob::OpFlash>& flashes,
                                          const std::vector<std::vector<FlashHit>>* hits = nullptr) const;

  /**
   * @brief Select the highest-scoring candidate for a charge position.
   *
   * @param candidates Output of SelectCandidates
   * @param charge_y Y coordinate of reconstructed vertex or centroid [cm]
   * @param charge_z Z coordinate of reconstructed vertex or centroid [cm]
   * @param has_charge Whether the charge position is valid (required if UsesCharge())
   * @return MatchResult Output matching summary and best flash index
   */
  MatchResult Match(const std::vector<Candidate>& candidates,
                    double charge_y,
                    double charge_z,
                    bool has_charge = true) const;

  /// SelectCandidates + Match
  MatchResult Match(const std::vector<recob::OpFlash>& flashes,
                    double charge_y,
                    double charge_z,
                    bool has_charge = true,
                    const std::vector<std::vector<FlashHit>>* hits = nullptr) const;

  const Config& GetConfig() const { return fConfig; }
  const BDTModel& GetModel() const { return fModel; }

  /// Inputs of one feature evaluation
  struct FeatureInput {
    const Candidate& c;
    double dy, dz, sigma;
    int n_candidates;
  };
  using FeatureFn = double (*)(const FeatureInput&);

private:
  void Setup();

  Config fConfig;
  BDTModel fModel;
  std::vector<FeatureFn> fFeatures;   ///< resolved fConfig.features
  bool fUsesHits = false;
  bool fUsesCharge = false;
};

} // namespace flashmatch

#endif // DUNERECO_FLASHMATCH_FLASHMATCHBDTALG_H

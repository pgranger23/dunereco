/**
 * \file FlashMatchBDTAlg.cxx
 *
 * \brief Pure C++ algorithmic engine for BDT-based charge-light matching in DUNE.
 *
 * \author Pierre Granger (pierre.granger@cern.ch)
 * \date 2026
 */

#include "FlashMatchBDTAlg.h"
#include "cetlib_except/exception.h"
#include <functional>
#include <map>
#include <numeric>
#include <limits>
#include <cmath>

namespace flashmatch {

FlashMatchBDTAlg::FlashMatchBDTAlg()
  : fConfig{}
{}

FlashMatchBDTAlg::FlashMatchBDTAlg(const fhicl::ParameterSet& pset)
{
  Configure(pset);
}

FlashMatchBDTAlg::FlashMatchBDTAlg(const Config& config)
  : fConfig(config)
{
  Configure(config);
}

void FlashMatchBDTAlg::Configure(const fhicl::ParameterSet& pset)
{
  fConfig.weight_file = pset.get<std::string>("WeightFile", "FlashMatch/bdt_augmented.bin");
  fConfig.pe_threshold = pset.get<double>("PEThreshold", 100.0);
  fConfig.spatial_sigma = pset.get<double>("SpatialSigma", 62.64);
  fConfig.min_bdt_score = pset.get<double>("MinBDTScore", 0.30);
  fConfig.fallback_to_charge = pset.get<bool>("FallbackToCharge", true);
  fConfig.feature_set = pset.get<std::string>("FeatureSet", "HD");
  fConfig.cathode_x = pset.get<double>("CathodeX", -320.);
  fConfig.min_channels = pset.get<int>("MinChannels", 3);
  fConfig.rank_min_channels = pset.get<int>("RankMinChannels", 20);

  LoadModel();
}

void FlashMatchBDTAlg::Configure(const Config& config)
{
  fConfig = config;
  LoadModel();
}

void FlashMatchBDTAlg::LoadModel()
{
  if (fConfig.feature_set != "HD" && fConfig.feature_set != "VD") {
    throw cet::exception("FlashMatchBDTAlg") << "FeatureSet must be \"HD\" or \"VD\", got \""
                                             << fConfig.feature_set << "\"\n";
  }
  fModel.Load(fConfig.weight_file);
  if (IsVD() && fModel.IsLoaded() && fModel.NumFeatures() != kNumVDFeatures) {
    throw cet::exception("FlashMatchBDTAlg") << fConfig.weight_file << " has " << fModel.NumFeatures()
                                             << " features, FeatureSet \"VD\" needs " << kNumVDFeatures << "\n";
  }
}

namespace {

  /// VD light observables of one flash, from its hits
  struct VDFlash {
    int idx = -1;   ///< index in the OpFlash collection
    double pe = 0., y = 0., z = 0., yw = 0., zw = 0.;
    double f_cath = 0., f_memL = 0., f_memR = 0., f_top1 = 0., f_top3 = 0., dur = 0.;
    int nch = 0, nhit = 0;
  };

  VDFlash Summarise(std::vector<FlashHit> hits, double cathode_x)
  {
    VDFlash s;
    s.nhit = static_cast<int>(hits.size());
    if (hits.empty()) return s;
    // time order, so that the sums are done in the same order as in the training code
    std::stable_sort(hits.begin(), hits.end(), [](FlashHit const& a, FlashHit const& b) { return a.time < b.time; });
    double sy = 0., sz = 0., cath = 0., memL = 0., memR = 0.;
    std::map<int, double> chpe;
    for (auto const& h : hits) {
      s.pe += h.pe;
      sy += h.pe * h.y;
      sz += h.pe * h.z;
      if (h.x < cathode_x) cath += h.pe;
      else if (h.y < 0.) memL += h.pe;
      else memR += h.pe;
      chpe[h.channel] += h.pe;
    }
    s.y = sy / s.pe;
    s.z = sz / s.pe;
    double vy = 0., vz = 0.;
    for (auto const& h : hits) {
      vy += h.pe * (h.y - s.y) * (h.y - s.y);
      vz += h.pe * (h.z - s.z) * (h.z - s.z);
    }
    s.yw = std::sqrt(vy / s.pe);
    s.zw = std::sqrt(vz / s.pe);
    s.f_cath = cath / s.pe;
    s.f_memL = memL / s.pe;
    s.f_memR = memR / s.pe;
    s.nch = static_cast<int>(chpe.size());
    std::vector<double> top;
    for (auto const& c : chpe) top.push_back(c.second);
    std::sort(top.begin(), top.end(), std::greater<>());
    s.f_top1 = top[0] / s.pe;
    s.f_top3 = (top[0] + (top.size() > 1 ? top[1] : 0.) + (top.size() > 2 ? top[2] : 0.)) / s.pe;
    s.dur = hits.back().time - hits.front().time;
    return s;
  }

  /// PE rank (0 = brightest) and log10(PE / brightest) among the selected candidates; 999 / -6 for the others
  void Ranks(std::vector<VDFlash> const& c, std::vector<bool> const& sel, std::vector<float>& rank, std::vector<float>& lrel)
  {
    rank.assign(c.size(), 999.f);
    lrel.assign(c.size(), -6.f);
    std::vector<size_t> order;
    for (size_t i = 0; i < c.size(); ++i)
      if (sel[i]) order.push_back(i);
    if (order.empty()) return;
    std::stable_sort(order.begin(), order.end(), [&c](size_t a, size_t b) { return c[a].pe > c[b].pe; });
    const double mx = std::max(c[order[0]].pe, 1.0);
    for (size_t r = 0; r < order.size(); ++r) {
      rank[order[r]] = static_cast<float>(r);
      lrel[order[r]] = static_cast<float>(std::log10(std::max(c[order[r]].pe / mx, 1e-6)));
    }
  }

} // namespace

MatchResult FlashMatchBDTAlg::MatchVD(const std::vector<recob::OpFlash>& flashes,
                                      const std::vector<std::vector<FlashHit>>& hits) const
{
  MatchResult result;
  if (!fModel.IsLoaded()) return result;

  std::vector<VDFlash> cand;
  for (size_t i = 0; i < hits.size(); ++i) {
    VDFlash s = Summarise(hits[i], fConfig.cathode_x);
    if (s.nch < fConfig.min_channels || s.pe < fConfig.pe_threshold) continue;
    s.idx = static_cast<int>(i);
    cand.push_back(s);
  }
  const size_t n = cand.size();
  result.n_candidates = static_cast<int>(n);
  if (n == 0) return result;

  std::vector<bool> all(n, true), wide(n);
  for (size_t c = 0; c < n; ++c) wide[c] = cand[c].nch >= fConfig.rank_min_channels;
  std::vector<float> rank, lrel, rank20, lrel20;
  Ranks(cand, all, rank, lrel);
  Ranks(cand, wide, rank20, lrel20);

  float best_score = -1.f;
  size_t best = 0;
  for (size_t c = 0; c < n; ++c) {
    auto const& s = cand[c];
    const float x[kNumVDFeatures] = {
      static_cast<float>(std::log10(s.pe)), static_cast<float>(s.nch), static_cast<float>(s.nhit),
      static_cast<float>(s.y), static_cast<float>(s.z), static_cast<float>(s.yw), static_cast<float>(s.zw),
      static_cast<float>(s.f_cath), static_cast<float>(s.f_memL), static_cast<float>(s.f_memR),
      static_cast<float>(s.f_top1), static_cast<float>(s.f_top3), static_cast<float>(s.dur),
      rank[c], lrel[c], rank20[c], lrel20[c], static_cast<float>(n)};
    const float score = fModel.PredictProba(x);
    if (score > best_score) {
      best_score = score;
      best = c;
    }
  }

  result.flash_idx = cand[best].idx;
  result.flash_time = flashes.at(cand[best].idx).Time();
  result.bdt_score = best_score;
  result.pe = cand[best].pe;
  result.matched = best_score >= fConfig.min_bdt_score;
  return result;
}

MatchResult FlashMatchBDTAlg::Match(const std::vector<recob::OpFlash>& flashes,
                                    double charge_y,
                                    double charge_z,
                                    bool has_charge) const
{
  MatchResult result;

  if (!fModel.IsLoaded() || !has_charge || !std::isfinite(charge_y) || !std::isfinite(charge_z)) {
    return result;
  }

  // Identify candidates passing PE threshold
  struct Candidate {
    int orig_idx;
    double pe;
    float rank;
  };

  std::vector<Candidate> candidates;
  candidates.reserve(flashes.size());

  double pe_max = 0.0;
  for (size_t i = 0; i < flashes.size(); ++i) {
    const auto& f = flashes[i];
    if (f.TotalPE() >= fConfig.pe_threshold) {
      candidates.push_back({static_cast<int>(i), f.TotalPE(), 0.0f});
      if (f.TotalPE() > pe_max) {
        pe_max = f.TotalPE();
      }
    }
  }

  const int n_cand = static_cast<int>(candidates.size());
  result.n_candidates = n_cand;

  if (n_cand == 0) {
    return result;
  }

  // Calculate PE ranks (0 = highest PE, 1 = 2nd highest, ...)
  std::vector<size_t> order(n_cand);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&candidates](size_t a, size_t b) {
    return candidates[a].pe > candidates[b].pe;
  });

  for (size_t r = 0; r < order.size(); ++r) {
    candidates[order[r]].rank = static_cast<float>(r);
  }

  float best_score = -1.0f;
  int best_cand_idx = -1;
  double best_dy = 0.0;
  double best_dz = 0.0;
  double best_chi2 = 0.0;

  static constexpr size_t NUM_FEATURES = 13;
  float x[NUM_FEATURES];

  for (int c = 0; c < n_cand; ++c) {
    const auto& cand = candidates[c];
    const auto& f = flashes[cand.orig_idx];

    const double pe = cand.pe;
    const double dy = f.YCenter() - charge_y;
    const double dz = f.ZCenter() - charge_z;
    const double abs_dy = std::abs(dy);
    const double abs_dz = std::abs(dz);
    const double dy_norm = dy / fConfig.spatial_sigma;
    const double dz_norm = dz / fConfig.spatial_sigma;
    const double chi2 = dy_norm * dy_norm + dz_norm * dz_norm;
    const double chi = std::sqrt(chi2);
    const double yw = f.YWidth();
    const double zw = f.ZWidth();
    const double width_sum = yw + zw;
    const double lpe = std::log10(std::max(pe, 1.0));
    const double pe_ratio = pe / std::max(pe_max, 1.0);
    const double log10_pe_ratio = std::log10(std::max(pe_ratio, 1e-6));

    x[0] = static_cast<float>(lpe);
    x[1] = static_cast<float>(dy);
    x[2] = static_cast<float>(dz);
    x[3] = static_cast<float>(abs_dy);
    x[4] = static_cast<float>(abs_dz);
    x[5] = static_cast<float>(chi2);
    x[6] = static_cast<float>(chi);
    x[7] = static_cast<float>(yw);
    x[8] = static_cast<float>(zw);
    x[9] = static_cast<float>(width_sum);
    x[10] = cand.rank;
    x[11] = static_cast<float>(log10_pe_ratio);
    x[12] = static_cast<float>(n_cand);

    // Safeguard against NaN/inf (mimics np.nan_to_num)
    for (size_t fi = 0; fi < NUM_FEATURES; ++fi) {
      if (!std::isfinite(x[fi])) {
        x[fi] = 0.0f;
      }
    }

    const float score = fModel.PredictProba(x);

    if (score > best_score) {
      best_score = score;
      best_cand_idx = c;
      best_dy = dy;
      best_dz = dz;
      best_chi2 = chi2;
    }
  }

  if (best_cand_idx >= 0) {
    const auto& best_cand = candidates[best_cand_idx];
    const auto& best_flash = flashes[best_cand.orig_idx];

    result.flash_idx = best_cand.orig_idx;
    result.flash_time = best_flash.Time();
    result.bdt_score = static_cast<double>(best_score);
    result.pe = best_cand.pe;
    result.dy = best_dy;
    result.dz = best_dz;
    result.chi2 = best_chi2;

    if (best_score >= fConfig.min_bdt_score) {
      result.matched = true;
    }
  }

  return result;
}

} // namespace flashmatch

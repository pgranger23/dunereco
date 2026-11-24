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
#include <numeric>
#include <limits>
#include <cmath>
#include <utility>

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
  fConfig.light_weight_file = pset.get<std::string>("LightWeightFile", "");
  fConfig.cathode_x = pset.get<double>("CathodeX", -320.);
  fConfig.min_channels = pset.get<int>("MinChannels", 3);
  fConfig.rank_min_channels = pset.get<int>("RankMinChannels", 20);

  LoadModels();
}

void FlashMatchBDTAlg::Configure(const Config& config)
{
  fConfig = config;
  LoadModels();
}

void FlashMatchBDTAlg::LoadModels()
{
  if (fConfig.feature_set != "HD" && fConfig.feature_set != "VD") {
    throw cet::exception("FlashMatchBDTAlg") << "FeatureSet must be \"HD\" or \"VD\", got \""
                                             << fConfig.feature_set << "\"\n";
  }
  fModel.Load(fConfig.weight_file);
  if (!fConfig.light_weight_file.empty()) fLightModel.Load(fConfig.light_weight_file);
  if (IsVD()) {
    if (fModel.IsLoaded() && fModel.NumFeatures() != kNumVDVertex) {
      throw cet::exception("FlashMatchBDTAlg") << fConfig.weight_file << " has " << fModel.NumFeatures()
                                               << " features, the VD charge-position model needs "
                                               << kNumVDVertex << "\n";
    }
    if (fLightModel.IsLoaded() && fLightModel.NumFeatures() != kNumVDLight) {
      throw cet::exception("FlashMatchBDTAlg") << fConfig.light_weight_file << " has "
                                               << fLightModel.NumFeatures()
                                               << " features, the VD light-only model needs "
                                               << kNumVDLight << "\n";
    }
  }
}

namespace {

  /// Light observables of one flash, from its hits
  struct VDFlashSummary {
    double pe = 0., y = 0., z = 0., yw = 0., zw = 0.;
    double f_cath = 0., f_memL = 0., f_memR = 0., f_top1 = 0., f_top3 = 0., dur = 0.;
    int nch = 0, nhit = 0;
  };

  VDFlashSummary Summarise(std::vector<FlashHit> hits, double cathode_x)
  {
    VDFlashSummary s;
    s.nhit = static_cast<int>(hits.size());
    if (hits.empty()) return s;
    std::stable_sort(hits.begin(), hits.end(),
                     [](FlashHit const& a, FlashHit const& b) { return a.time < b.time; });
    double sy = 0., sz = 0., cath = 0., memL = 0., memR = 0.;
    std::vector<std::pair<int, double>> chpe;   // (channel, PE)
    for (auto const& h : hits) {
      s.pe += h.pe;
      sy += h.pe * h.y;
      sz += h.pe * h.z;
      if (h.x < cathode_x) cath += h.pe;
      else if (h.y < 0.) memL += h.pe;
      else memR += h.pe;
      auto it = std::find_if(chpe.begin(), chpe.end(), [&h](auto const& c) { return c.first == h.channel; });
      if (it == chpe.end()) chpe.emplace_back(h.channel, h.pe);
      else it->second += h.pe;
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
    std::sort(chpe.begin(), chpe.end(), [](auto const& a, auto const& b) { return a.second > b.second; });
    s.f_top1 = chpe[0].second / s.pe;
    double top3 = 0.;
    for (size_t i = 0; i < std::min<size_t>(3, chpe.size()); ++i) top3 += chpe[i].second;
    s.f_top3 = top3 / s.pe;
    s.dur = hits.back().time - hits.front().time;
    return s;
  }

  /// PE rank (0 = brightest) and log10(PE / brightest) among the selected entries; 999 / -6 otherwise
  void Ranks(std::vector<double> const& pe, std::vector<bool> const& sel,
             std::vector<float>& rank, std::vector<float>& lrel)
  {
    const size_t n = pe.size();
    rank.assign(n, 999.f);
    lrel.assign(n, -6.f);
    std::vector<size_t> order;
    double mx = 0.;
    for (size_t i = 0; i < n; ++i) {
      if (!sel[i]) continue;
      order.push_back(i);
      mx = std::max(mx, pe[i]);
    }
    std::stable_sort(order.begin(), order.end(), [&pe](size_t a, size_t b) { return pe[a] > pe[b]; });
    for (size_t r = 0; r < order.size(); ++r) rank[order[r]] = static_cast<float>(r);
    for (size_t i : order) lrel[i] = static_cast<float>(std::log10(std::max(pe[i] / std::max(mx, 1.0), 1e-6)));
  }

} // namespace

std::vector<std::vector<float>> FlashMatchBDTAlg::ComputeVDFeatures(const std::vector<FlashHits>& flashes,
                                                                    double charge_y,
                                                                    double charge_z,
                                                                    bool with_charge,
                                                                    std::vector<int>& cand) const
{
  cand.clear();
  std::vector<VDFlashSummary> sum;
  for (size_t i = 0; i < flashes.size(); ++i) {
    VDFlashSummary s = Summarise(flashes[i].hits, fConfig.cathode_x);
    if (s.nch < fConfig.min_channels || s.pe < fConfig.pe_threshold) continue;
    cand.push_back(static_cast<int>(i));
    sum.push_back(s);
  }
  const size_t n = sum.size();
  std::vector<double> pe(n);
  std::vector<bool> all(n, true), wide(n);
  for (size_t c = 0; c < n; ++c) {
    pe[c] = sum[c].pe;
    wide[c] = sum[c].nch >= fConfig.rank_min_channels;
  }
  std::vector<float> rank, lrel, rank20, lrel20;
  Ranks(pe, all, rank, lrel);
  Ranks(pe, wide, rank20, lrel20);

  std::vector<std::vector<float>> X(n);
  for (size_t c = 0; c < n; ++c) {
    auto const& s = sum[c];
    auto& x = X[c];
    x = {static_cast<float>(std::log10(s.pe)), static_cast<float>(s.nch), static_cast<float>(s.nhit),
         static_cast<float>(s.y), static_cast<float>(s.z), static_cast<float>(s.yw), static_cast<float>(s.zw),
         static_cast<float>(s.f_cath), static_cast<float>(s.f_memL), static_cast<float>(s.f_memR),
         static_cast<float>(s.f_top1), static_cast<float>(s.f_top3), static_cast<float>(s.dur),
         rank[c], lrel[c], rank20[c], lrel20[c], static_cast<float>(n)};
    if (with_charge) {
      const float dy = static_cast<float>(s.y - charge_y);
      const float dz = static_cast<float>(s.z - charge_z);
      x.insert(x.end(), {dy, dz, std::abs(dy), std::abs(dz), std::hypot(dy / 100.f, dz / 100.f)});
    }
  }
  return X;
}

MatchResult FlashMatchBDTAlg::MatchVD(const std::vector<FlashHits>& flashes,
                                      double charge_y,
                                      double charge_z,
                                      bool has_charge) const
{
  MatchResult result;
  const bool with_charge = has_charge && std::isfinite(charge_y) && std::isfinite(charge_z);
  const BDTModel& model = with_charge ? fModel : fLightModel;
  if (!model.IsLoaded()) return result;

  std::vector<int> cand;
  const auto X = ComputeVDFeatures(flashes, charge_y, charge_z, with_charge, cand);
  result.n_candidates = static_cast<int>(cand.size());

  float best_score = -1.f;
  int best = -1;
  for (size_t c = 0; c < X.size(); ++c) {
    const float score = model.PredictProba(X[c].data());
    if (score > best_score) {
      best_score = score;
      best = static_cast<int>(c);
    }
  }
  if (best < 0) return result;

  auto const& x = X[best];
  result.flash_idx = flashes[cand[best]].flash_idx;
  result.bdt_score = best_score;
  result.pe = std::pow(10., x[0]);
  if (with_charge) {
    result.dy = x[kNumVDLight];
    result.dz = x[kNumVDLight + 1];
    result.chi2 = x[kNumVDLight + 4] * x[kNumVDLight + 4];
  }
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

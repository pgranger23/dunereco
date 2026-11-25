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

namespace {

  enum class Source { Flash, Hits, Event, Charge };

  struct FeatureDef {
    Source source;
    FlashMatchBDTAlg::FeatureFn value;
  };

  using In = FlashMatchBDTAlg::FeatureInput;

  /// Available model inputs. A model lists the ones it was trained with (Features), in its order.
  const std::map<std::string, FeatureDef>& FeatureTable()
  {
    static const std::map<std::string, FeatureDef> table = {
      // recob::OpFlash
      {"log10_pe",            {Source::Flash, [](In const& i) { return std::log10(std::max(i.c.pe, 1.0)); }}},
      {"y",                   {Source::Flash, [](In const& i) { return i.c.y; }}},
      {"z",                   {Source::Flash, [](In const& i) { return i.c.z; }}},
      {"y_width",             {Source::Flash, [](In const& i) { return i.c.y_width; }}},
      {"z_width",             {Source::Flash, [](In const& i) { return i.c.z_width; }}},
      {"width_sum",           {Source::Flash, [](In const& i) { return i.c.y_width + i.c.z_width; }}},
      // OpHits of the flash (at the centre of their PD)
      {"n_channels",          {Source::Hits, [](In const& i) { return double(i.c.n_channels); }}},
      {"n_hits",              {Source::Hits, [](In const& i) { return double(i.c.n_hits); }}},
      {"hit_y_width",         {Source::Hits, [](In const& i) { return i.c.hit_y_width; }}},
      {"hit_z_width",         {Source::Hits, [](In const& i) { return i.c.hit_z_width; }}},
      {"frac_cathode",        {Source::Hits, [](In const& i) { return i.c.frac_cathode; }}},
      {"frac_membrane_neg_y", {Source::Hits, [](In const& i) { return i.c.frac_membrane_neg_y; }}},
      {"frac_membrane_pos_y", {Source::Hits, [](In const& i) { return i.c.frac_membrane_pos_y; }}},
      {"frac_top1_channel",   {Source::Hits, [](In const& i) { return i.c.frac_top1_channel; }}},
      {"frac_top3_channels",  {Source::Hits, [](In const& i) { return i.c.frac_top3_channels; }}},
      {"duration",            {Source::Hits, [](In const& i) { return i.c.duration; }}},
      // candidates of the event
      {"pe_rank",             {Source::Event, [](In const& i) { return i.c.pe_rank; }}},
      {"log10_pe_ratio",      {Source::Event, [](In const& i) { return i.c.log10_pe_ratio; }}},
      {"pe_rank_wide",        {Source::Hits, [](In const& i) { return i.c.pe_rank_wide; }}},
      {"log10_pe_ratio_wide", {Source::Hits, [](In const& i) { return i.c.log10_pe_ratio_wide; }}},
      {"n_candidates",        {Source::Event, [](In const& i) { return double(i.n_candidates); }}},
      // charge position (vertex or charge centroid)
      {"dy",                  {Source::Charge, [](In const& i) { return i.dy; }}},
      {"dz",                  {Source::Charge, [](In const& i) { return i.dz; }}},
      {"abs_dy",              {Source::Charge, [](In const& i) { return std::abs(i.dy); }}},
      {"abs_dz",              {Source::Charge, [](In const& i) { return std::abs(i.dz); }}},
      {"chi2",                {Source::Charge, [](In const& i) {
                                 const double a = i.dy / i.sigma, b = i.dz / i.sigma;
                                 return a * a + b * b; }}},
      {"chi",                 {Source::Charge, [](In const& i) {
                                 const double a = i.dy / i.sigma, b = i.dz / i.sigma;
                                 return std::sqrt(a * a + b * b); }}},
    };
    return table;
  }

  /// Hit-based observables of a flash
  void SummariseHits(std::vector<FlashHit> hits, double cathode_x, Candidate& c)
  {
    c.n_hits = static_cast<int>(hits.size());
    if (hits.empty()) return;
    std::stable_sort(hits.begin(), hits.end(), [](FlashHit const& a, FlashHit const& b) { return a.time < b.time; });
    double pe = 0., sy = 0., sz = 0., cath = 0., memL = 0., memR = 0.;
    std::map<int, double> chpe;
    for (auto const& h : hits) {
      pe += h.pe;
      sy += h.pe * h.y;
      sz += h.pe * h.z;
      if (h.x < cathode_x) cath += h.pe;
      else if (h.y < 0.) memL += h.pe;
      else memR += h.pe;
      chpe[h.channel] += h.pe;
    }
    const double my = sy / pe, mz = sz / pe;
    double vy = 0., vz = 0.;
    for (auto const& h : hits) {
      vy += h.pe * (h.y - my) * (h.y - my);
      vz += h.pe * (h.z - mz) * (h.z - mz);
    }
    c.hit_y_width = std::sqrt(vy / pe);
    c.hit_z_width = std::sqrt(vz / pe);
    c.frac_cathode = cath / pe;
    c.frac_membrane_neg_y = memL / pe;
    c.frac_membrane_pos_y = memR / pe;
    c.n_channels = static_cast<int>(chpe.size());
    std::vector<double> top;
    for (auto const& ch : chpe) top.push_back(ch.second);
    std::sort(top.begin(), top.end(), std::greater<>());
    c.frac_top1_channel = top[0] / pe;
    c.frac_top3_channels = (top[0] + (top.size() > 1 ? top[1] : 0.) + (top.size() > 2 ? top[2] : 0.)) / pe;
    c.duration = hits.back().time - hits.front().time;
  }

  /// PE rank (0 = brightest) and log10(PE / brightest) among the selected candidates
  void Rank(std::vector<Candidate>& cands, std::vector<bool> const& sel,
            double Candidate::*rank, double Candidate::*ratio)
  {
    std::vector<size_t> order;
    for (size_t i = 0; i < cands.size(); ++i)
      if (sel[i]) order.push_back(i);
    if (order.empty()) return;
    std::stable_sort(order.begin(), order.end(), [&cands](size_t a, size_t b) { return cands[a].pe > cands[b].pe; });
    const double pe_max = std::max(cands[order[0]].pe, 1.0);
    for (size_t r = 0; r < order.size(); ++r) {
      Candidate& c = cands[order[r]];
      c.*rank = static_cast<double>(r);
      c.*ratio = std::log10(std::max(c.pe / pe_max, 1e-6));
    }
  }

} // namespace

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
  const Config def;
  fConfig.weight_file = pset.get<std::string>("WeightFile", def.weight_file);
  fConfig.features = pset.get<std::vector<std::string>>("Features", def.features);
  fConfig.pe_threshold = pset.get<double>("PEThreshold", def.pe_threshold);
  fConfig.min_channels = pset.get<int>("MinChannels", def.min_channels);
  fConfig.spatial_sigma = pset.get<double>("SpatialSigma", def.spatial_sigma);
  fConfig.wide_min_channels = pset.get<int>("WideMinChannels", def.wide_min_channels);
  fConfig.cathode_x = pset.get<double>("CathodeX", def.cathode_x);
  fConfig.min_bdt_score = pset.get<double>("MinBDTScore", def.min_bdt_score);
  fConfig.fallback_to_charge = pset.get<bool>("FallbackToCharge", def.fallback_to_charge);

  Setup();
}

void FlashMatchBDTAlg::Configure(const Config& config)
{
  fConfig = config;
  Setup();
}

void FlashMatchBDTAlg::Setup()
{
  fFeatures.clear();
  fUsesHits = fConfig.min_channels > 0;
  fUsesCharge = false;
  for (auto const& name : fConfig.features) {
    auto const it = FeatureTable().find(name);
    if (it == FeatureTable().end()) {
      cet::exception e("FlashMatchBDTAlg");
      e << "Unknown feature \"" << name << "\". Available:";
      for (auto const& f : FeatureTable()) e << " " << f.first;
      throw e << "\n";
    }
    fFeatures.push_back(it->second.value);
    fUsesHits |= it->second.source == Source::Hits;
    fUsesCharge |= it->second.source == Source::Charge;
  }

  fModel.Load(fConfig.weight_file);
  if (fModel.IsLoaded() && fModel.NumFeatures() != fFeatures.size()) {
    throw cet::exception("FlashMatchBDTAlg") << fConfig.weight_file << " has " << fModel.NumFeatures()
                                             << " features, but " << fFeatures.size() << " are configured\n";
  }
}

std::vector<Candidate> FlashMatchBDTAlg::SelectCandidates(const std::vector<recob::OpFlash>& flashes,
                                                          const std::vector<std::vector<FlashHit>>* hits) const
{
  if (fUsesHits && !(hits && hits->size() == flashes.size())) {
    throw cet::exception("FlashMatchBDTAlg") << "the configured features need the hits of every flash\n";
  }
  std::vector<Candidate> cands;
  for (size_t i = 0; i < flashes.size(); ++i) {
    const auto& f = flashes[i];
    if (f.TotalPE() < fConfig.pe_threshold) continue;
    Candidate c;
    c.flash_idx = static_cast<int>(i);
    c.time = f.Time();
    c.pe = f.TotalPE();
    c.y = f.YCenter();
    c.z = f.ZCenter();
    c.y_width = f.YWidth();
    c.z_width = f.ZWidth();
    if (fUsesHits) SummariseHits((*hits)[i], fConfig.cathode_x, c);
    if (c.n_channels < fConfig.min_channels) continue;
    cands.push_back(c);
  }
  Rank(cands, std::vector<bool>(cands.size(), true), &Candidate::pe_rank, &Candidate::log10_pe_ratio);
  if (fUsesHits) {
    std::vector<bool> wide(cands.size());
    for (size_t i = 0; i < cands.size(); ++i) wide[i] = cands[i].n_channels >= fConfig.wide_min_channels;
    Rank(cands, wide, &Candidate::pe_rank_wide, &Candidate::log10_pe_ratio_wide);
  }
  return cands;
}

MatchResult FlashMatchBDTAlg::Match(const std::vector<recob::OpFlash>& flashes,
                                    double charge_y,
                                    double charge_z,
                                    bool has_charge,
                                    const std::vector<std::vector<FlashHit>>* hits) const
{
  return Match(SelectCandidates(flashes, hits), charge_y, charge_z, has_charge);
}

MatchResult FlashMatchBDTAlg::Match(const std::vector<Candidate>& candidates,
                                    double charge_y,
                                    double charge_z,
                                    bool has_charge) const
{
  MatchResult result;

  if (!fModel.IsLoaded()) return result;
  if (fUsesCharge && (!has_charge || !std::isfinite(charge_y) || !std::isfinite(charge_z))) return result;

  const int n_cand = static_cast<int>(candidates.size());
  result.n_candidates = n_cand;

  float best_score = -1.0f;
  int best = -1;
  std::vector<float> x(fFeatures.size());
  for (int k = 0; k < n_cand; ++k) {
    const Candidate& c = candidates[k];
    const In in{c, fUsesCharge ? c.y - charge_y : 0., fUsesCharge ? c.z - charge_z : 0., fConfig.spatial_sigma, n_cand};
    for (size_t j = 0; j < fFeatures.size(); ++j) {
      x[j] = static_cast<float>(fFeatures[j](in));
      if (!std::isfinite(x[j])) x[j] = 0.0f;   // mimics np.nan_to_num
    }
    const float score = fModel.PredictProba(x.data());
    if (score > best_score) {
      best_score = score;
      best = k;
    }
  }

  if (best >= 0) {
    const Candidate& c = candidates[best];
    result.flash_idx = c.flash_idx;
    result.flash_time = c.time;
    result.bdt_score = static_cast<double>(best_score);
    result.pe = c.pe;
    if (fUsesCharge) {
      result.dy = c.y - charge_y;
      result.dz = c.z - charge_z;
      const double a = result.dy / fConfig.spatial_sigma, b = result.dz / fConfig.spatial_sigma;
      result.chi2 = a * a + b * b;
    }
    result.matched = best_score >= fConfig.min_bdt_score;
  }

  return result;
}

} // namespace flashmatch

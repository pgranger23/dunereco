/**
 * \file FlashMatchBDT_module.cc
 *
 * \brief LArSoft art::EDProducer module for charge-light matching using
 *        XGBoost BDT models.
 *
 * Reconstructs interaction T0 and establishes associations between
 * recob::PFParticle, anab::T0, and recob::OpFlash.
 *
 * FeatureSet "HD" (default): 13 features from the recob::OpFlash objects and the
 * charge position of each primary PFParticle (vertex, or SpacePoint centroid).
 *
 * FeatureSet "VD": features computed from the OpHits of each flash (via the
 * OpFlash-OpHit associations of the flash producer), see FlashMatchBDTAlg::MatchVD.
 * Primary PFParticles without a charge position, and events without PFParticles
 * (MatchWithoutPFParticle), are matched with the light-only model; in the latter
 * case the T0 is associated with the flash only.
 *
 * \author Pierre Granger (pierre.granger@cern.ch)
 * \date 2026
 */

#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art/Persistency/Common/PtrMaker.h"
#include "canvas/Persistency/Common/FindManyP.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/ParameterSet.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

#include "larcore/Geometry/WireReadout.h"
#include "larcorealg/Geometry/OpDetGeo.h"
#include "lardataobj/AnalysisBase/T0.h"
#include "lardataobj/RecoBase/OpFlash.h"
#include "lardataobj/RecoBase/OpHit.h"
#include "lardataobj/RecoBase/PFParticle.h"
#include "lardataobj/RecoBase/Vertex.h"
#include "lardataobj/RecoBase/SpacePoint.h"

#include "FlashMatchBDTAlg.h"

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace flashmatch {

class FlashMatchBDT : public art::EDProducer {
public:
  explicit FlashMatchBDT(fhicl::ParameterSet const& p);

  FlashMatchBDT(FlashMatchBDT const&) = delete;
  FlashMatchBDT(FlashMatchBDT&&) = delete;
  FlashMatchBDT& operator=(FlashMatchBDT const&) = delete;
  FlashMatchBDT& operator=(FlashMatchBDT&&) = delete;

  void produce(art::Event& evt) override;

private:
  std::vector<FlashHits> GetFlashHits(art::Event const& evt,
                                      art::Handle<std::vector<recob::OpFlash>> const& flashHandle);

  art::InputTag fFlashModuleLabel;
  art::InputTag fPFParticleModuleLabel;
  art::InputTag fSpacePointModuleLabel;
  bool fSelectPrimaryOnly;
  bool fMatchWithoutPFParticle;   ///< VD: light-only T0 for events without primary PFParticles

  FlashMatchBDTAlg fBDTAlg;
  std::map<int, std::array<double, 3>> fChannelCentre;   ///< VD: optical channel -> PD centre
};

FlashMatchBDT::FlashMatchBDT(fhicl::ParameterSet const& p)
  : EDProducer{p}
  , fFlashModuleLabel(p.get<art::InputTag>("FlashModuleLabel", "opflash"))
  , fPFParticleModuleLabel(p.get<art::InputTag>("PFParticleModuleLabel", "pandora"))
  , fSpacePointModuleLabel(p.get<art::InputTag>("SpacePointModuleLabel", "pandora"))
  , fSelectPrimaryOnly(p.get<bool>("SelectPrimaryOnly", true))
  , fMatchWithoutPFParticle(p.get<bool>("MatchWithoutPFParticle", false))
  , fBDTAlg(p.get<fhicl::ParameterSet>("BDTAlg", fhicl::ParameterSet{}))
{
  produces<std::vector<anab::T0>>();
  produces<art::Assns<recob::PFParticle, anab::T0>>();
  produces<art::Assns<anab::T0, recob::OpFlash>>();
}

std::vector<FlashHits> FlashMatchBDT::GetFlashHits(art::Event const& evt,
                                                   art::Handle<std::vector<recob::OpFlash>> const& flashHandle)
{
  art::FindManyP<recob::OpHit> flashToHits(flashHandle, evt, fFlashModuleLabel);
  auto const& wireReadout = art::ServiceHandle<geo::WireReadout const>()->Get();
  std::vector<FlashHits> out(flashHandle->size());
  for (size_t i = 0; i < flashHandle->size(); ++i) {
    out[i].flash_idx = static_cast<int>(i);
    for (auto const& hit : flashToHits.at(i)) {
      const int ch = hit->OpChannel();
      auto it = fChannelCentre.find(ch);
      if (it == fChannelCentre.end()) {
        auto const c = wireReadout.OpDetGeoFromOpChannel(ch).GetCenter();
        it = fChannelCentre.emplace(ch, std::array<double, 3>{c.X(), c.Y(), c.Z()}).first;
      }
      out[i].hits.push_back({hit->PE(), hit->PeakTime(), it->second[0], it->second[1], it->second[2], ch});
    }
  }
  return out;
}

void FlashMatchBDT::produce(art::Event& evt)
{
  auto t0Col = std::make_unique<std::vector<anab::T0>>();
  auto pfpT0Assn = std::make_unique<art::Assns<recob::PFParticle, anab::T0>>();
  auto t0FlashAssn = std::make_unique<art::Assns<anab::T0, recob::OpFlash>>();
  auto putAll = [&]() {
    evt.put(std::move(t0Col));
    evt.put(std::move(pfpT0Assn));
    evt.put(std::move(t0FlashAssn));
  };

  // Retrieve OpFlashes
  art::Handle<std::vector<recob::OpFlash>> flashHandle;
  if (!evt.getByLabel(fFlashModuleLabel, flashHandle) || !flashHandle.isValid() || flashHandle->empty()) {
    mf::LogDebug("FlashMatchBDT") << "No OpFlashes found with label " << fFlashModuleLabel.encode();
    putAll();
    return;
  }

  const bool vd = fBDTAlg.IsVD();
  std::vector<FlashHits> flashHits;
  if (vd) flashHits = GetFlashHits(evt, flashHandle);

  art::PtrMaker<anab::T0> makeT0Ptr{evt};
  std::optional<MatchResult> lightResult;   // VD light-only match, the same for every PFParticle

  auto runMatch = [&](double cy, double cz, bool has_charge) -> MatchResult {
    if (!vd) return fBDTAlg.Match(*flashHandle, cy, cz, has_charge);
    if (!has_charge) {
      if (!lightResult) lightResult = fBDTAlg.MatchVD(flashHits, 0., 0., false);
      return *lightResult;
    }
    return fBDTAlg.MatchVD(flashHits, cy, cz, true);
  };

  // Store a T0 for a match, associated with the flash and, if given, the PFParticle
  auto storeT0 = [&](MatchResult const& res, art::Ptr<recob::PFParticle> const* pfpPtr) {
    if (!(res.matched && res.flash_idx >= 0 && res.flash_idx < static_cast<int>(flashHandle->size()))) return;
    const double t0_ns = (*flashHandle)[res.flash_idx].Time() * 1.0e3; // convert us to ns
    const int t0_id = static_cast<int>(t0Col->size());
    t0Col->emplace_back(t0_ns, 1, res.flash_idx, t0_id, res.bdt_score);
    art::Ptr<anab::T0> t0Ptr = makeT0Ptr(t0_id);
    art::Ptr<recob::OpFlash> flashPtr(flashHandle, res.flash_idx);
    if (pfpPtr) pfpT0Assn->addSingle(*pfpPtr, t0Ptr);
    t0FlashAssn->addSingle(t0Ptr, flashPtr);
    mf::LogInfo("FlashMatchBDT") << "Matched " << (pfpPtr ? "PFParticle #" + std::to_string((*pfpPtr)->Self()) : "event")
                                 << " to OpFlash #" << res.flash_idx << " (T0 = " << t0_ns
                                 << " ns, BDT = " << res.bdt_score << ", PE = " << res.pe
                                 << ", chi2 = " << res.chi2 << ")";
  };

  // Retrieve PFParticles
  art::Handle<std::vector<recob::PFParticle>> pfpHandle;
  const bool hasPfp = evt.getByLabel(fPFParticleModuleLabel, pfpHandle) && pfpHandle.isValid() && !pfpHandle->empty();
  bool anyPrimary = false;

  if (hasPfp) {
    // Look up Vertex associations
    art::FindManyP<recob::Vertex> pfpToVertex(pfpHandle, evt, fPFParticleModuleLabel);

    // Look up SpacePoint associations (for centroid fallback if configured)
    std::unique_ptr<art::FindManyP<recob::SpacePoint>> pfpToSpacePoint;
    if (fBDTAlg.GetConfig().fallback_to_charge) {
      pfpToSpacePoint = std::make_unique<art::FindManyP<recob::SpacePoint>>(pfpHandle, evt, fSpacePointModuleLabel);
    }

    for (size_t iPfp = 0; iPfp < pfpHandle->size(); ++iPfp) {
      const auto& pfp = (*pfpHandle)[iPfp];

      if (fSelectPrimaryOnly && !pfp.IsPrimary()) {
        continue;
      }
      anyPrimary = true;

      double charge_y = 0.0;
      double charge_z = 0.0;
      bool has_charge = false;

      // 1. Try vertex position
      if (pfpToVertex.isValid()) {
        const auto& vtxList = pfpToVertex.at(iPfp);
        if (!vtxList.empty() && vtxList.front()->isValid()) {
          charge_y = vtxList.front()->position().Y();
          charge_z = vtxList.front()->position().Z();
          has_charge = true;
        }
      }

      // 2. Try SpacePoint centroid fallback if vertex missing or invalid
      if (!has_charge && pfpToSpacePoint && pfpToSpacePoint->isValid()) {
        const auto& spList = pfpToSpacePoint->at(iPfp);
        if (!spList.empty()) {
          double sum_y = 0.0;
          double sum_z = 0.0;
          size_t count = 0;
          for (const auto& sp : spList) {
            if (sp) {
              sum_y += sp->XYZ()[1];
              sum_z += sp->XYZ()[2];
              ++count;
            }
          }
          if (count > 0) {
            charge_y = sum_y / static_cast<double>(count);
            charge_z = sum_z / static_cast<double>(count);
            has_charge = true;
          }
        }
      }

      // 3. VD: light-only model when there is no charge position
      if (!has_charge && !vd) {
        continue;
      }

      art::Ptr<recob::PFParticle> pfpPtr(pfpHandle, iPfp);
      storeT0(runMatch(charge_y, charge_z, has_charge), &pfpPtr);
    }
  }
  else {
    mf::LogDebug("FlashMatchBDT") << "No PFParticles found with label " << fPFParticleModuleLabel.encode();
  }

  // VD: events without (primary) PFParticles get a light-only T0, associated with the flash only
  if (vd && fMatchWithoutPFParticle && !anyPrimary) {
    storeT0(runMatch(0., 0., false), nullptr);
  }

  putAll();
}

} // namespace flashmatch

DEFINE_ART_MODULE(flashmatch::FlashMatchBDT)

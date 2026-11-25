/**
 * \file FlashMatchBDT_module.cc
 *
 * \brief LArSoft art::EDProducer module for charge-light matching using an
 *        XGBoost BDT model (see FlashMatchBDTAlg).
 *
 * Reconstructs interaction T0 and establishes associations between
 * recob::PFParticle, anab::T0, and recob::OpFlash.
 *
 * What is read follows from the configured features: the OpHits of each flash
 * (OpFlash-OpHit associations of the flash producer) only if hit-based features
 * are used. If the features use a charge position, each primary PFParticle is
 * matched with its vertex (or SpacePoint centroid). Otherwise the match is the same
 * for all PFParticles: one T0 per event, associated with every primary PFParticle,
 * and kept without PFParticle if MatchWithoutPFParticle is set.
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

#include <memory>
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
  std::vector<std::vector<FlashHit>> GetFlashHits(art::Event const& evt,
                                                  art::Handle<std::vector<recob::OpFlash>> const& flashHandle) const;
  void StoreEventT0(art::Event const& evt,
                    art::Handle<std::vector<recob::OpFlash>> const& flashHandle,
                    MatchResult const& res,
                    std::vector<anab::T0>& t0Col,
                    art::Assns<recob::PFParticle, anab::T0>& pfpT0Assn,
                    art::Assns<anab::T0, recob::OpFlash>& t0FlashAssn) const;

  art::InputTag fFlashModuleLabel;
  art::InputTag fPFParticleModuleLabel;
  art::InputTag fSpacePointModuleLabel;
  bool fSelectPrimaryOnly;
  bool fMatchWithoutPFParticle;   ///< Without charge features: keep the T0 of events without PFParticle

  FlashMatchBDTAlg fBDTAlg;
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

void FlashMatchBDT::produce(art::Event& evt)
{
  auto t0Col = std::make_unique<std::vector<anab::T0>>();
  auto pfpT0Assn = std::make_unique<art::Assns<recob::PFParticle, anab::T0>>();
  auto t0FlashAssn = std::make_unique<art::Assns<anab::T0, recob::OpFlash>>();

  // Retrieve OpFlashes
  art::Handle<std::vector<recob::OpFlash>> flashHandle;
  if (!evt.getByLabel(fFlashModuleLabel, flashHandle) || !flashHandle.isValid() || flashHandle->empty()) {
    mf::LogDebug("FlashMatchBDT") << "No OpFlashes found with label " << fFlashModuleLabel.encode();
    evt.put(std::move(t0Col));
    evt.put(std::move(pfpT0Assn));
    evt.put(std::move(t0FlashAssn));
    return;
  }

  // Candidate flashes; the hits are only read if the configured features need them
  std::vector<std::vector<FlashHit>> hits;
  if (fBDTAlg.UsesHits()) hits = GetFlashHits(evt, flashHandle);
  const std::vector<Candidate> candidates = fBDTAlg.SelectCandidates(*flashHandle, &hits);

  // Features without charge position: the match is the same for every PFParticle
  if (!fBDTAlg.UsesCharge()) {
    StoreEventT0(evt, flashHandle, fBDTAlg.Match(candidates, 0., 0., false), *t0Col, *pfpT0Assn, *t0FlashAssn);
    evt.put(std::move(t0Col));
    evt.put(std::move(pfpT0Assn));
    evt.put(std::move(t0FlashAssn));
    return;
  }

  // Retrieve PFParticles
  art::Handle<std::vector<recob::PFParticle>> pfpHandle;
  if (!evt.getByLabel(fPFParticleModuleLabel, pfpHandle) || !pfpHandle.isValid() || pfpHandle->empty()) {
    mf::LogDebug("FlashMatchBDT") << "No PFParticles found with label " << fPFParticleModuleLabel.encode();
    evt.put(std::move(t0Col));
    evt.put(std::move(pfpT0Assn));
    evt.put(std::move(t0FlashAssn));
    return;
  }

  // Look up Vertex associations
  art::FindManyP<recob::Vertex> pfpToVertex(pfpHandle, evt, fPFParticleModuleLabel);

  // Look up SpacePoint associations (for centroid fallback if configured)
  std::unique_ptr<art::FindManyP<recob::SpacePoint>> pfpToSpacePoint;
  if (fBDTAlg.GetConfig().fallback_to_charge) {
    pfpToSpacePoint = std::make_unique<art::FindManyP<recob::SpacePoint>>(pfpHandle, evt, fSpacePointModuleLabel);
  }

  art::PtrMaker<anab::T0> makeT0Ptr{evt};

  for (size_t iPfp = 0; iPfp < pfpHandle->size(); ++iPfp) {
    const auto& pfp = (*pfpHandle)[iPfp];

    if (fSelectPrimaryOnly && !pfp.IsPrimary()) {
      continue;
    }

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

    if (!has_charge) {
      continue;
    }

    // Run matching algorithm
    MatchResult res = fBDTAlg.Match(candidates, charge_y, charge_z, true);

    if (res.matched && res.flash_idx >= 0 && res.flash_idx < static_cast<int>(flashHandle->size())) {
      const double t0_ns = res.flash_time * 1.0e3; // convert us to ns
      const int t0_id = static_cast<int>(t0Col->size());

      t0Col->emplace_back(t0_ns, 1, res.flash_idx, t0_id, res.bdt_score);

      art::Ptr<anab::T0> t0Ptr = makeT0Ptr(t0_id);
      art::Ptr<recob::PFParticle> pfpPtr(pfpHandle, iPfp);
      art::Ptr<recob::OpFlash> flashPtr(flashHandle, res.flash_idx);

      pfpT0Assn->addSingle(pfpPtr, t0Ptr);
      t0FlashAssn->addSingle(t0Ptr, flashPtr);

      mf::LogInfo("FlashMatchBDT") << "Matched PFParticle #" << pfp.Self()
                                   << " to OpFlash #" << res.flash_idx
                                   << " (T0 = " << t0_ns << " ns, BDT = " << res.bdt_score
                                   << ", PE = " << res.pe << ", chi2 = " << res.chi2 << ")";
    }
  }

  evt.put(std::move(t0Col));
  evt.put(std::move(pfpT0Assn));
  evt.put(std::move(t0FlashAssn));
}

std::vector<std::vector<FlashHit>> FlashMatchBDT::GetFlashHits(art::Event const& evt,
                                                               art::Handle<std::vector<recob::OpFlash>> const& flashHandle) const
{
  // OpHits of each flash, placed at the centre of their photon detector
  art::FindManyP<recob::OpHit> flashToHits(flashHandle, evt, fFlashModuleLabel);
  auto const& wireReadout = art::ServiceHandle<geo::WireReadout const>()->Get();
  std::vector<std::vector<FlashHit>> hits(flashHandle->size());
  for (size_t i = 0; i < flashHandle->size(); ++i) {
    for (auto const& hit : flashToHits.at(i)) {
      auto const c = wireReadout.OpDetGeoFromOpChannel(hit->OpChannel()).GetCenter();
      hits[i].push_back({hit->PE(), hit->PeakTime(), c.X(), c.Y(), c.Z(), hit->OpChannel()});
    }
  }
  return hits;
}

void FlashMatchBDT::StoreEventT0(art::Event const& evt,
                                 art::Handle<std::vector<recob::OpFlash>> const& flashHandle,
                                 MatchResult const& res,
                                 std::vector<anab::T0>& t0Col,
                                 art::Assns<recob::PFParticle, anab::T0>& pfpT0Assn,
                                 art::Assns<anab::T0, recob::OpFlash>& t0FlashAssn) const
{
  if (!res.matched) return;

  std::vector<art::Ptr<recob::PFParticle>> pfps;
  art::Handle<std::vector<recob::PFParticle>> pfpHandle;
  if (evt.getByLabel(fPFParticleModuleLabel, pfpHandle)) {
    for (size_t iPfp = 0; iPfp < pfpHandle->size(); ++iPfp) {
      if (!fSelectPrimaryOnly || (*pfpHandle)[iPfp].IsPrimary()) pfps.emplace_back(pfpHandle, iPfp);
    }
  }
  if (pfps.empty() && !fMatchWithoutPFParticle) return;

  const double t0_ns = res.flash_time * 1.0e3; // convert us to ns
  t0Col.emplace_back(t0_ns, 1, res.flash_idx, 0, res.bdt_score);
  art::Ptr<anab::T0> t0Ptr = art::PtrMaker<anab::T0>{evt}(0);
  t0FlashAssn.addSingle(t0Ptr, art::Ptr<recob::OpFlash>(flashHandle, res.flash_idx));
  for (auto const& pfp : pfps) {
    pfpT0Assn.addSingle(pfp, t0Ptr);
  }

  mf::LogInfo("FlashMatchBDT") << "Matched OpFlash #" << res.flash_idx << " (T0 = " << t0_ns
                               << " ns, BDT = " << res.bdt_score << ", PE = " << res.pe << ") to "
                               << pfps.size() << " PFParticle(s)";
}

} // namespace flashmatch

DEFINE_ART_MODULE(flashmatch::FlashMatchBDT)

////////////////////////////////////////////////////////////////////////
// \file    RegCNNMapper_module.cc
// \brief   Producer module for creating RegCNN PixelMap objects modified from CVNMapper_module.cc
// \author  Ilsoo Seong - iseong@uci.edu
//
// Modifications to interface for numu energy estimation
//  - Wenjie Wu - wenjieww@uci.edu
////////////////////////////////////////////////////////////////////////

// C/C++ includes
#include <iostream>
#include <sstream>

// ROOT includes
#include "TTree.h"
#include "TH2F.h"

// Framework includes
#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art_root_io/TFileDirectory.h"
#include "art_root_io/TFileService.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "fhiclcpp/ParameterSet.h"
#include "messagefacility/MessageLogger/MessageLogger.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "canvas/Persistency/Common/Assns.h"
#include "canvas/Persistency/Common/FindManyP.h"

// LArSoft includes
#include "lardataobj/RecoBase/Hit.h"
#include "lardata/DetectorInfoServices/DetectorPropertiesService.h"
#include "lardataobj/RawData/ExternalTrigger.h"
#include "larpandora/LArPandoraInterface/LArPandoraHelper.h"

#include "lardata/Utilities/AssociationUtil.h"
#include "lardataobj/RecoBase/Wire.h"
#include "lardataobj/RecoBase/Vertex.h"
#include "lardataobj/RecoBase/PFParticle.h"
#include "lardataobj/RecoBase/SpacePoint.h"
#include "nusimdata/SimulationBase/MCNeutrino.h"
#include "nusimdata/SimulationBase/MCParticle.h"
#include "nusimdata/SimulationBase/MCTruth.h"

#include "dune/RegCNN/art/RegPixelMapProducer.h"
#include "dune/RegCNN/art/RegPixelMap3DProducer.h"
#include "dune/RegCNN/func/RegPixelMap.h"
#include "dune/RegCNN/func/RegPixelMap3D.h"
#include "dune/RegCNN/func/RegCNNResult.h"

namespace lar_pandora{class LArPandoraHelper;}

namespace cnn {

  class RegCNNMapper : public art::EDProducer {
  public:
    explicit RegCNNMapper(fhicl::ParameterSet const& pset);
    ~RegCNNMapper();

    void produce(art::Event& evt);
    void beginJob();
    void endJob();


  private:
    /// Module lablel for input clusters
    std::string    fHitsModuleLabel;

    /// Instance lablel for cluster pixelmaps
    std::string    fClusterPMLabel;

    /// Minimum number of hits for cluster to be converted to pixel map
    unsigned short fMinClusterHits;

    /// Width of pixel map in tdcs
    unsigned short fTdcWidth;

    /// Length of pixel map in wires
    unsigned short fWireLength;

    // resolution of ticks
    double fTimeResolution;
    // resolution of wires
    double fWireResolution;

    /// Maximum gap in wires at front of cluster to prevent pruning of upstream
    /// hits
    //unsigned int fMaxWireGap;

    /// Use unwrapped pixel maps?
    bool fUnwrappedPixelMap;

    /// select which global wire method
    int fGlobalWireMethod;
    /// select how to choose center of pixel map
    int fUseRecoVertex;
    /// choose to create 3D pixel maps
    int fUseThreeDMap;

    std::string fVertexModuleLabel;
    std::string fPFParticleModuleLabel;
    std::string fPandoraNuVertexModuleLabel;

    std::string fRegCNNResultLabel;
    std::string fRegCNNModuleLabel;

    /// PixelMapProducer does the work for us
    RegPixelMapProducer fProducer;
    RegPixelMap3DProducer fProducer3D;

  };


  //.......................................................................
  RegCNNMapper::RegCNNMapper(fhicl::ParameterSet const& pset):
  EDProducer(pset),
  fHitsModuleLabel  (pset.get<std::string>         ("HitsModuleLabel")),
  fClusterPMLabel   (pset.get<std::string>         ("ClusterPMLabel")),
  fMinClusterHits   (pset.get<unsigned short>      ("MinClusterHits")),
  fTdcWidth         (pset.get<unsigned short>      ("TdcWidth")),
  fWireLength       (pset.get<unsigned short>      ("WireLength")),
  fTimeResolution   (pset.get<unsigned short>      ("TimeResolution")),
  fWireResolution   (pset.get<unsigned short>      ("WireResolution")),
  fGlobalWireMethod (pset.get<int>                 ("GlobalWireMethod")),
  fUseRecoVertex    (pset.get<int>                 ("UseRecoVertex")),
  fUseThreeDMap     (pset.get<int>                 ("UseThreeDMap")),
  fVertexModuleLabel(pset.get<std::string>         ("VertexModuleLabel")),
  fPFParticleModuleLabel(pset.get<std::string>     ("PFParticleModuleLabel")),
  fPandoraNuVertexModuleLabel(pset.get<std::string>("PandoraNuVertexModuleLabel")),
  fRegCNNResultLabel (pset.get<std::string>        ("RegCNNResultLabel")),
  fRegCNNModuleLabel (pset.get<std::string>        ("RegCNNModuleLabel")),
  fProducer(RegPixelMapProducer(fWireLength, fWireResolution, fTdcWidth, fTimeResolution, fGlobalWireMethod)),
  fProducer3D(RegPixelMap3DProducer(32, 0., 160., 32, 0., 160, 32, 0., 320.))
    { 
        if (fUseThreeDMap==0) {
            produces< std::vector<cnn::RegPixelMap>   >(fClusterPMLabel);
        } else if (fUseThreeDMap==1) {
            produces< std::vector<cnn::RegPixelMap3D>   >(fClusterPMLabel);
        } else {
            mf::LogError("RegCNNMapper::RegCNNMapper")<<"RegCNNMapper accepts 0 or 1 for fUseThreeDMap"<<std::endl;
        }
    }

  //......................................................................
  RegCNNMapper::~RegCNNMapper()
  {
    //======================================================================
    // Clean up any memory allocated by your module
    //======================================================================
  }

  //......................................................................
  void RegCNNMapper::beginJob()
  {  }

  //......................................................................
  void RegCNNMapper::endJob()
  {
  }

  //......................................................................
  void RegCNNMapper::produce(art::Event& evt)
  {

    art::Handle< std::vector< recob::Hit > > hitListHandle;
    std::vector< art::Ptr< recob::Hit > > hitlist;
    if (evt.getByLabel(fHitsModuleLabel, hitListHandle))
      art::fill_ptr_vector(hitlist, hitListHandle);

    art::FindManyP<recob::Wire> fmwire(hitListHandle, evt, fHitsModuleLabel);

    unsigned short nhits = hitlist.size();

    // Get the vertex out of the event record
    art::Handle<std::vector<recob::Vertex> > vertexHandle;
    std::vector<art::Ptr<recob::Vertex> > vertex_list;
    if (evt.getByLabel(fVertexModuleLabel,vertexHandle))
        art::fill_ptr_vector(vertex_list, vertexHandle);
    art::FindMany<recob::PFParticle> fmPFParticle(vertexHandle, evt, fPFParticleModuleLabel);

    art::FindManyP<recob::SpacePoint> fmSPFromHits(hitListHandle, evt, fPFParticleModuleLabel);

    //Declaring containers for things to be stored in event
    std::unique_ptr< std::vector<cnn::RegPixelMap> >
      pmCol(new std::vector<cnn::RegPixelMap>);

    std::unique_ptr< std::vector<cnn::RegPixelMap3D> >
        pm3DCol(new std::vector<cnn::RegPixelMap3D>);

    auto const clockData = art::ServiceHandle<detinfo::DetectorClocksService const>()->DataFor(evt);
    auto const detProp = art::ServiceHandle<detinfo::DetectorPropertiesService const>()->DataFor(evt, clockData);
    if(nhits>fMinClusterHits){
        if (fUseThreeDMap==0) {
            RegPixelMap pm;
            if (fUseRecoVertex==0){
                // create pixel map based on mean of wire and ticks
                pm = fProducer.CreateMap(clockData, detProp, hitlist, fmwire);
            } else if (fUseRecoVertex==1) {
              // create pixel map based on the reconstructed vertex
              // Get RegCNN Results
              art::Handle<std::vector<cnn::RegCNNResult>> cnnresultListHandle;
              evt.getByLabel(fRegCNNModuleLabel, fRegCNNResultLabel, cnnresultListHandle);
                  std::vector<float> vtx(3, -99999);
              if (!cnnresultListHandle.failedToGet())
              {
                  if (!cnnresultListHandle->empty())
                  {
                      const std::vector<float>& v = (*cnnresultListHandle)[0].fOutput;
                      for (unsigned int ii = 0; ii < 3; ii++){
                           vtx[ii] = v[ii];
                           std::cout<<"vertex "<<ii<<" : "<<vtx[ii]<<std::endl;
                      }
                  }
              }
              pm = fProducer.CreateMap(clockData, detProp, hitlist, fmwire, vtx);
            }
            else if (fUseRecoVertex==2) {
                // create pixel map based on pandora vertex
                lar_pandora::PFParticleVector particleVector;
                lar_pandora::LArPandoraHelper::CollectPFParticles(evt, fPandoraNuVertexModuleLabel, particleVector);
                lar_pandora::VertexVector vertexVector;
                lar_pandora::PFParticlesToVertices particlesToVertices;
                lar_pandora::LArPandoraHelper::CollectVertices(evt, fPandoraNuVertexModuleLabel, vertexVector, particlesToVertices);

                int n_prims = 0;
                int n_pand_particles = particleVector.size();
                //std::cout<<"n_pand_particles "<<n_pand_particles<<std::endl;
                double xyz_temp[3] = {0.0, 0.0, 0.0};
                std::vector<float> vtx(3, -99999);
                for (int ipfp = 0; ipfp< n_pand_particles; ipfp++) {
                    const art::Ptr<recob::PFParticle> particle = particleVector.at(ipfp);
                    if (!particle->IsPrimary()) continue;
                    n_prims++;
                    // Particles <-> Vertices
                    lar_pandora::PFParticlesToVertices::const_iterator vIter = particlesToVertices.find(particle);
                    if (particlesToVertices.end() != vIter) {
                        const lar_pandora::VertexVector& vertexVector = vIter->second;
                        if (!vertexVector.empty()) {
                            if (vertexVector.size()!=1)
                                std::cout<<" Warning: Found particle with more than one associated vertex "<<std::endl;

                            const art::Ptr<recob::Vertex> vertex_pfp = *(vertexVector.begin());
                            vertex_pfp->XYZ(xyz_temp);
                            for (unsigned int ii=0; ii<3; ++ii) {
                                vtx[ii] = xyz_temp[ii];
                            }
                            //std::cout<<"Pandora vertex "<<vtx[0]<<", "<<vtx[1]<<", "<<vtx[2]<<std::endl;
                        }
                    }
                } 
                pm = fProducer.CreateMap(clockData, detProp, hitlist, fmwire, vtx);
            }
            else {
              // create pixel map based on the reconstructed vertex on wire and tick coordinate
              // Get RegCNN Results
              art::Handle<std::vector<cnn::RegCNNResult>> cnnresultListHandle;
              evt.getByLabel(fRegCNNModuleLabel, fRegCNNResultLabel, cnnresultListHandle);
                  std::vector<float> vtx(6, -99999);
              if (!cnnresultListHandle.failedToGet())
              {
                  if (!cnnresultListHandle->empty())
                  {
                      const std::vector<float>& v = (*cnnresultListHandle)[0].fOutput;
                      for (unsigned int ii = 0; ii < 6; ii++){
                           vtx[ii] = v[ii];
                      }
                  }
              }
              pm = fProducer.CreateMap(clockData, detProp, hitlist, fmwire, vtx);
            }
            // skip if PixelMap is empty
            if (pm.fInPM) pmCol->push_back(pm);
        }
        else if (fUseThreeDMap==1) {
            RegPixelMap3D pm;
            if (fUseRecoVertex==2) {
                // create pixel map based on pandora vertex
                lar_pandora::PFParticleVector particleVector;
                lar_pandora::LArPandoraHelper::CollectPFParticles(evt, fPandoraNuVertexModuleLabel, particleVector);
                lar_pandora::VertexVector vertexVector;
                lar_pandora::PFParticlesToVertices particlesToVertices;
                lar_pandora::LArPandoraHelper::CollectVertices(evt, fPandoraNuVertexModuleLabel, vertexVector, particlesToVertices);

                int n_prims = 0;
                int n_pand_particles = particleVector.size();
                //std::cout<<"n_pand_particles "<<n_pand_particles<<std::endl;
                double xyz_temp[3] = {0.0, 0.0, 0.0};
                std::vector<float> vtx(3, -99999);
                for (int ipfp = 0; ipfp< n_pand_particles; ipfp++) {
                    const art::Ptr<recob::PFParticle> particle = particleVector.at(ipfp);
                    if (!particle->IsPrimary()) continue;
                    n_prims++;
                    // Particles <-> Vertices
                    lar_pandora::PFParticlesToVertices::const_iterator vIter = particlesToVertices.find(particle);
                    if (particlesToVertices.end() != vIter) {
                        const lar_pandora::VertexVector& vertexVector = vIter->second;
                        if (!vertexVector.empty()) {
                            if (vertexVector.size()!=1)
                                std::cout<<" Warning: Found particle with more than one associated vertex "<<std::endl;

                            const art::Ptr<recob::Vertex> vertex_pfp = *(vertexVector.begin());
                            vertex_pfp->XYZ(xyz_temp);
                            for (unsigned int ii=0; ii<3; ++ii) {
                                vtx[ii] = xyz_temp[ii];
                            }
                            //std::cout<<"Pandora vertex "<<vtx[0]<<", "<<vtx[1]<<", "<<vtx[2]<<std::endl;
                        }
                    }
                } 
                pm = fProducer3D.Create3DMap(clockData, detProp, hitlist, fmSPFromHits, vtx);
            }
            if (pm.fInPM) pm3DCol->push_back(pm);
        }
        
      //pm.Print();
    }
    //Boundary bound = pm.Bound();
    //}
    if (fUseThreeDMap==0) {
        evt.put(std::move(pmCol), fClusterPMLabel);
    }
    else if (fUseThreeDMap==1) {
        evt.put(std::move(pm3DCol), fClusterPMLabel);
    }
    std::cout<<"Map Complete!"<<std::endl;
  }

  //----------------------------------------------------------------------

DEFINE_ART_MODULE(cnn::RegCNNMapper)
} // end namespace cnn
////////////////////////////////////////////////////////////////////////
